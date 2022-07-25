import numpy as np
import keras
import cv2
from keras.utils import data_utils

class DataGenerator(data_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, IDs, X_dir, Y_dir, batch_size=32, input_dim=(320,320,3), out_dim=(20,20,8), scale_by=256, shuffle=True):
        'Initialization'
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.batch_size = batch_size

        self.Y_dir = Y_dir
        self.X_dir = X_dir

        self.list_IDs = IDs
        self.scale_by = scale_by

        self.dx = input_dim[1] / out_dim[1]
        self.dy = input_dim[0] / out_dim[0]
        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.input_dim))
        y = np.zeros((self.batch_size, *self.out_dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = cv2.imread(self.X_dir + str(self.list_IDs[ID]) + '.png')/self.scale_by
            y[i] = self.get_y(self.list_IDs[ID])

        return X, y

    def get_y(self, IDname):
        Y = np.zeros(self.out_dim)
        with open(self.Y_dir + IDname + '.txt', 'r') as f:
            labels = f.readlines()

        for label in labels:
            cat, xmid, ymid, w = [float(x) for x in label.split(' ')[:-1]]

            x = int(xmid * self.input_dim[1] / self.dx)
            y = int(ymid * self.input_dim[0] / self.dy)

            rx = (int(xmid * self.input_dim[1]) % self.dx) / self.dx
            ry = (int(ymid * self.input_dim[0]) % self.dy) / self.dy

            Y[y, x, :4] = np.asarray([1, rx, ry, w])
            Y[y, x, 4+int(cat)] = 1

        return Y