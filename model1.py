from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, BatchNormalization, Flatten, Dropout, Input, Concatenate, AveragePooling2D, Layer, Permute, Conv2DTranspose, UpSampling2D, Add
from keras.models import model_from_json, Model
from keras.activations import softmax
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
import time
import pickle as pkl

def scheduler(epoch, lr):
	lr = 0.00001 / (1 + epoch)
	print("New learning rate:", lr)
	return lr


def lossFun(y_true, y_pred):
	loss = 0

	loss_obj =  K.mean(y_true[...,0] * K.square(y_pred[...,0] - y_true[...,0]))
	loss_nobj = K.mean(K.abs(y_true[...,0]-1) * K.square(y_pred[...,0] - y_true[...,0]))
	loss_mid = K.mean(y_true[...,0] * K.mean(K.square(y_pred[...,1:3] - y_true[...,1:3])))
	loss_box = K.mean(y_true[...,0] * K.mean(K.square(y_pred[...,3] - y_true[...,3])))
	loss_class = K.mean(y_true[...,0] * K.mean(K.square(y_pred[...,4:] - y_true[...,4:])))

	loss += loss_obj*2
	loss += loss_nobj
	loss += loss_mid
	loss += loss_box
	loss += loss_class

	return loss

def build_model(input_h, input_w, out_classes = 3):
	inp = Input(shape=(input_h,input_w,3))
	
	x = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(inp)

	channels = [32, 64, 96, 128]
	for n_channel in channels:
		x = Conv2D(n_channel, kernel_size=(3,3), strides=(2,2), padding="same", activation='mish')(x)
		x = Conv2D(n_channel, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(x)
		x = Conv2D(n_channel, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(x)

	x = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(x)

	xobj = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(x)
	xpos = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(x)
	xw = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(x)
	xclass = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation='mish')(x)

	xobj = Conv2D(1, kernel_size=(3,3), strides=(1,1), padding="same", activation='linear')(xobj)
	xpos = Conv2D(2, kernel_size=(3,3), strides=(1,1), padding="same", activation='linear')(xpos)
	xw = Conv2D(1, kernel_size=(3,3), strides=(1,1), padding="same", activation='linear')(xw)
	xclass = Conv2D(out_classes, kernel_size=(3,3), strides=(1,1), padding="same", activation='linear')(xclass)

	x = Concatenate()([xobj, xpos, xw, xclass])
	model = Model(inputs=inp, outputs=x)

	return model


def save_model(model, model_name, model_dir, history):
	model_json = model.to_json()

	stamp = str(int(time.time()))
	print('Saved with stamp: ', stamp)
	with open(model_dir+model_name+"_"+stamp+".json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(model_dir+model_name+"_"+stamp+".h5")

	with open(model_dir+model_name+"_"+stamp+".pkl", 'wb') as f:
		pkl.dump(history.history, f)

def load_model(directory, model_file, weights_file):
	# load json and create model
	json_file = open(directory+model_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(directory+weights_file)
	print("Loaded model from disk.")
	input_shape = loaded_model.layers[0].input_shape[0]
	
	input_h, input_w = input_shape[1:3]

	return loaded_model, input_h, input_w