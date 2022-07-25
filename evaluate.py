import numpy as np
import pickle as pkl
from tensorflow.keras import optimizers
import datetime
import time
import cv2
import copy
from os import listdir
from os.path import isfile, join
import random
import json
import sys
import copy

from model1 import build_model, lossFun, load_model

def split_sample(image, in_dim):
	width = in_dim[1]
	height = in_dim[0]
	n_x = int(image.shape[1]/width)
	n_y = int(image.shape[0]/height)
	test_samples = []
	for i in range(n_x):
		for j in range(n_y):
			sample = np.asarray((image[j*height:(j+1)*height,i*width:(i+1)*width,:]), dtype=np.uint8)/256
			test_samples.append(sample)

	return test_samples

# ############################################################
model_dir = './models/'
# ############################################################

in_dim = (512,512,3)
out_dim = (32,32,4)


model_time_stamp = '1658750940'

print("Loading model...")
model_file = 'kdet_' + model_time_stamp + '.json'
weights_file = 'kdet_' + model_time_stamp + '.h5'

model, input_h, input_w = load_model(model_dir, model_file, weights_file)
model.summary()
print("Done")
# ############################################################

file_name = 'pic02.jpg'
sample_dir = './test_samples/'

image = cv2.imread(sample_dir + file_name)

test_samples = split_sample(image, in_dim)


dx = in_dim[1] / out_dim[1] 
dy = in_dim[0] / out_dim[0] 

obj_thr = 0.5

colors = [(255,0,0), (0,255,0), (0,0,255)]
NS = ['N5', 'N4', 'N3']

for s, sample in enumerate(test_samples):
	sample_in = np.resize(sample, (1, *sample.shape))

	prediction = model.predict(sample_in)

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

				cv2.rectangle(sample, (x1,y1), (x2,y2),(255,0,0), 1)
				cv2.putText(sample, str(NS[cat]), (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[cat], 1, cv2.LINE_AA)

	cv2.imshow(str(s), sample)

cv2.waitKey(0)
