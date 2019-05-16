import getopt

import cv2 as cv
import numpy as np
import sys
import torch
import torch.nn as nn

from models.extrapolation import Model

# Device for running computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Not computing gradients for better computationl performance
torch.set_grad_enabled(False)

# Parse script arguments
arg_weights = "data/extrapolation80.pth"
arg_frame1 = "examples/extrapolation/1/0.png"
arg_frame2 = "examples/extrapolation/1/1.png"
arg_frame3 = "examples/extrapolation/1/2.png"
arg_out = "examples/extrapolation/out.png"
arg_steps = 1
arg_inf = None

for opt, arg in getopt.getopt(sys.argv[1:], '', [ param[2:] + '=' for param in sys.argv[1::2] ])[0]:
	if opt == '--model' and arg != '': arg_weights = arg
	if opt == '--first' and arg != '': arg_frame1 = arg
	if opt == '--second' and arg != '': arg_frame2 = arg
	if opt == '--third' and arg != '': arg_frame3 = arg
	if opt == '--out' and arg != '': arg_out = arg
	if opt == '--steps' and arg != '': arg_steps = int(arg)
	if opt == '--inf' and arg != '': arg_inf = arg

if arg_inf is not None:
	arg_frame1 = arg_inf + "0.png"
	arg_frame2 = arg_inf + "1.png"
	arg_frame3 = arg_inf + "2.png"

#######################################

def prepInput(arg_frame1, arg_frame2, arg_frame3):
	# Read input images and check dimensions
	img1 = cv.imread(arg_frame1, cv.IMREAD_GRAYSCALE).astype('float32') / 255.
	img2 = cv.imread(arg_frame2, cv.IMREAD_GRAYSCALE).astype('float32') / 255.
	img3 = cv.imread(arg_frame3, cv.IMREAD_GRAYSCALE).astype('float32') / 255.

	assert img1.shape == img2.shape == img3.shape
	shape = img1.shape

	img1 = img1.reshape((1,1,shape[0],shape[1]))
	img2 = img2.reshape((1,1,shape[0],shape[1]))
	img3 = img3.reshape((1,1,shape[0],shape[1]))

	# Create input tensor and compute output tensor
	tensor_in = torch.tensor( np.concatenate((img1,img2,img3),axis=1) ).to(device)

	return tensor_in

def predict(arg_frame1, arg_frame2, arg_frame3, arg_out, arg_steps):
	tensor_in = prepInput(arg_frame1, arg_frame2, arg_frame3)
	tensor_out = model.predict(tensor_in, arg_steps)

	# Save output image from the output tensor
	img_out = (tensor_out.cpu().detach().numpy() * 255).astype('int')
	out_name = arg_out.rsplit('.',1)[0]
	for i in range(img_out.shape[1]):
		cv.imwrite(out_name+"_"+str(i)+".png", img_out[0,i])

#######################################

# Create model for interpolation
model = Model().to(device)
model.load_state_dict(torch.load(arg_weights, map_location=device))
model.eval()

#######################################

if __name__ == '__main__':
	predict(arg_frame1, arg_frame2, arg_frame3, arg_out, arg_steps)