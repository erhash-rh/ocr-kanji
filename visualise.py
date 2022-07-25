import cv2

# Pick image index from ./imgs/ to view the labels
idx = 1
img_path = './imgs/'
label_path = './labels/'

image = cv2.imread(img_path + '{}.png'.format(idx))

with open(label_path + '{}.txt'.format(idx), 'r') as f:
	labels = f.readlines()

colors = [(255,0,0), (0,255,0), (0,0,255)]
NS = ['N5', 'N4', 'N3']

for i, label in enumerate(labels):
	cat, xmid, ymid, side = [float(x) for x in label.split(' ')[:-1]]
	cat = int(cat)

	x1 = int((xmid-side/2) * 512)
	x2 = int((xmid+side/2) * 512)
	y1 = int((ymid-side/2) * 512)
	y2 = int((ymid+side/2) * 512)

	cv2.rectangle(image, (x1,y1), (x2,y2), colors[cat], 1)
	cv2.putText(image, str(NS[cat]), (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[cat], 1, cv2.LINE_AA)


cv2.imshow('test', image)
cv2.waitKey(0)
