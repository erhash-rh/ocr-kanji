import numpy as np
import pickle as pkl
from keras.callbacks import LearningRateScheduler

from tensorflow.keras import optimizers

import datetime
import time
import cv2
import random
 
from model1 import build_model, lossFun, save_model
from os import listdir
from os.path import isfile, join

from dataset import DataGenerator

# Setup training and test data IDs
images_dir = './imgs/'
labels_dir = './labels/'
model_dir = './models/'

test_size = 0.1

IDs = [f[:-4] for f in listdir(labels_dir) if isfile(join(labels_dir, f))]
split_index = int(len(IDs) * (1-test_size))
random.shuffle(IDs)

IDs_train, IDs_test = IDs[:split_index], IDs[split_index:]

# Setup training parameters
input_h = 512
input_w = 512

in_dim = (input_h, input_w, 3)
out_classes = 3 # N5, N4 and N3
out_dim = (32,32,4+out_classes)
train_parameters = {
	'batch_size': 16,
	'input_dim': in_dim,
	'out_dim': out_dim,
	'scale_by': 256,
	'shuffle': True
	}

epochs = 32
learning_rate = 0.0005

# Setup data generators
train_generator = DataGenerator(IDs_train, images_dir, labels_dir, **train_parameters)
test_generator = DataGenerator(IDs_test, images_dir, labels_dir, **train_parameters)

# Build model
model = build_model(input_h, input_w)
adam = optimizers.Adam(learning_rate = learning_rate)
model.compile(optimizer = adam, loss=lossFun, run_eagerly=True)
model.summary()

# Train & save model
history = model.fit(train_generator, epochs = epochs)
model_name = 'kdet' 
save_model(model, model_name, model_dir, history)


# View some test samples
test_samples = 4
dx = in_dim[1] / out_dim[1]
dy = in_dim[0] / out_dim[0] 

colors = [(255,0,0), (0,255,0), (0,0,255)]
NS = ['N5', 'N4', 'N3']

obj_thr = 0.5
for p in range(test_samples):
	prediction = model.predict(np.asarray([test_generator[0][0][p]]))
	image = np.asarray(test_generator[0][0][p]*256, dtype=np.uint8)

	for c in range(out_dim[1]):
		for r in range(out_dim[0]):

			if prediction[0,r,c,0] > obj_thr:
				rx, ry, side = prediction[0,r,c,1:4]
				cat = np.argmax(prediction[0,r,c,4:])
				xmid, ymid = (c+rx)*dx, (r+ry)*dy

				x1 = int(xmid - side*in_dim[1]/2)
				x2 = int(xmid + side*in_dim[1]/2)
				y1 = int(ymid - side*in_dim[0]/2)
				y2 = int(ymid + side*in_dim[0]/2)

				cv2.rectangle(image, (x1,y1), (x2,y2),(255,0,0), 1)
				cv2.putText(image, str(NS[cat]), (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[cat], 1, cv2.LINE_AA)

	cv2.imshow(str(p), image)
cv2.waitKey(0)