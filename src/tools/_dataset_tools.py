import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

import tools._my_tools as mt

# If tuple is passed, returns it unchanged.
# If int is passed, returns a tuple of two such ints.
def parseTuple(tup):
	if type(tup) is tuple:
		return tup
	else:
		return (tup,tup)

# Crops rectangle of size from img in coordinates [x,y].
def cropImg(img, x, y, size):
	W,H = parseTuple(size)

	return img[y:y+H, x:x+W]

# Performs sliding window crops of size, with stride, from every image in
# folder in_f and saves it to folder out_f/crop_n where crop_n is 
# number of current crop.
# Sufficient number of folders named out_f/XX needs to be created
# beforehand.
# Function assumes that every image in in_f has the same size.
def cropFolder(in_f, out_f, size, stride):
	images = sorted(os.listdir(in_f))
	# -----------------------------------------------------
	W,H = parseTuple(size)

	tmp = cv.imread(in_f+images[0], cv.IMREAD_GRAYSCALE)
	w_times = tmp.shape[1] // stride - W // stride + 1
	h_times = tmp.shape[0] // stride - H // stride + 1
	# -----------------------------------------------------
	for i in images:
		img = cv.imread(in_f+i, cv.IMREAD_GRAYSCALE)

		for j in range(h_times):
			for k in range(w_times):
				x = k*stride
				y = j*stride
				crop_n = j*w_times+k
				cv.imwrite(out_f+str(crop_n).zfill(2)+"/"+i, cropImg(img,x,y,(W,H)))
		
		print (i)

# Takes all images from folder in_f and removes images which have more than 95%
# of the area without precipitation or have only precipitation level 1 of 16. Also,
# removes previous images so that there are always left three consecutive images.
# Function assumes that every image in in_f has the same size.
def findTriplets(in_f):
	images = sorted(os.listdir(in_f))
	# -----------------------------------------------------
	tmp = cv.imread(in_f+images[0], cv.IMREAD_GRAYSCALE)
	threshold = int(0.95 * tmp.shape[0] * tmp.shape[1])
	# -----------------------------------------------------
	cnt = 0
	last = []

	for i in range(len(images)):
		img = cv.imread(in_f+images[i],cv.IMREAD_GRAYSCALE)
		unique, counts = np.unique(img, return_counts=True)

		if counts[0] > threshold or np.max(unique) <= 17:
			os.remove(in_f+images[i])
			for j in last:
				os.remove(in_f+j)
			last = []
		else:
			last.append(images[i])
			if len(last) == 3:
				last = []
	# -----------------------------------------------------
	images = sorted(os.listdir(in_f))
	print("In folder \"" + str(in_f) + "\" where left " + str(len(images)) + " images.")


def loadToNPA(in_f):
	X = []
	y = []
	# -----------------------------------------------------	
	images = sorted(os.listdir(in_f))
	
	for i in range(0,len(images),3):
		img = [cv.imread(in_f+images[j],cv.IMREAD_GRAYSCALE) for j in range(i,i+3)]
		x = np.stack([img[0],img[2]],axis=0)
		X.append(x)
		y.append(img[1])

	X = np.array(X)
	y = np.array(y)

	return X,y