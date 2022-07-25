import numpy as np 
import cv2
from PIL import ImageFont, ImageDraw, Image
from kanji_lists import JLPT

save_dir = './generated/'

kanjis = [list(JLPT.N5), list(JLPT.N4), list(JLPT.N3)]

print(len(kanjis))

fonts = ["./simsun.ttc"]

def color():
	color = tuple(np.random.randint(10, size=(3)))
	return color

def draw_word(draw, pos, space, n_chars, font, color, width, height):
	# Draws a single word on one line with random kanjis
	classes = np.random.randint(0, 3, n_chars)
	kanji_list = []
	for i in range(n_chars):
		kanji_list.append(np.random.choice(kanjis[classes[i]]))

	x, y = pos
	labels = []

	half_size = font_size/2
	font_size_r = font_size/width

	for i, kanji in enumerate(kanji_list):
		draw.text((x,y), kanji, font=font, fill=color)
		labels.append([classes[i], (x + half_size)/width, (y + half_size)/height, font_size_r])
		x += font_size + space
		if x+font_size > width:
			break
	return labels

# Setup paths
fontpath = "./simsun.ttc"
img_dir = './imgs/'
label_dir = './labels/'

# Setup input image size
height = 512
width = 512

# Setup no of images in dataset
N_PICS = 10

# Setup data generator parameters
NOISE_LEVEL = 5

FONT_SIZE_MIN = 24
FONT_SIZE_MAX = 56

NO_LINES_MIN = 10
NO_LINES_MAX = 32

X_START_MAX = 10
Y_START_MAX = 50

N_CHARS_MAX = 30
X_SPACE_MAX = 10

for i in range(N_PICS):
	background = np.random.randint(200,256-NOISE_LEVEL)
	noise = np.random.randint(NOISE_LEVEL, size=(height,width,3), dtype=np.uint8)
	image = np.ones((height,width,3), np.uint8) * background + noise

	# prepare PIL in order to draw kanji characters
	img_pil = Image.fromarray(image)
	draw = ImageDraw.Draw(img_pil)

	# randomise kanji size, no of lines and spacings
	font_size = np.random.randint(FONT_SIZE_MIN, FONT_SIZE_MAX) 
	font = ImageFont.truetype(fontpath, font_size)

	n_lines = np.random.randint(NO_LINES_MIN, NO_LINES_MAX)
	dy = np.random.randint(font_size, font_size+50)
	y = np.random.randint(Y_START_MAX)

	# generate newspaper like image and labels
	labels = []
	for j in range(n_lines):
		x = np.random.randint(X_START_MAX)
		pos = (x,y)
		n_chars = np.random.randint(N_CHARS_MAX)
		space = np.random.randint(X_SPACE_MAX)
		labels += draw_word(draw, pos, space, n_chars, font, color(), width, height)
		y += dy
		if y + font_size > height:
			break

	image = np.array(img_pil)
	
	# Save X data
	cv2.imwrite(img_dir + '{}.png'.format(str(i)), image)

	# Save Y data
	with open(label_dir + '{}.txt'.format(str(i)), 'w') as f:
		for label in labels:
			for item in label:
				f.write(str(item)+ ' ')
			f.write('\n')


cv2.waitKey(0)