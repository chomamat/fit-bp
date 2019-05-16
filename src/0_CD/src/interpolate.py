import getopt

import cv2 as cv
import numpy as np
import sys
import torch
import torch.nn as nn

from models.interpolation import Model

# Device for running computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Not computing gradients for better computationl performance
torch.set_grad_enabled(False)

# Parse script arguments
arg_weights = "data/interpolation85.pth"
arg_frame1 = "examples/interpolation/03_1.png"
arg_frame2 = "examples/interpolation/03_3.png"
arg_out = "examples/interpolation/out.png"

for opt, arg in getopt.getopt(sys.argv[1:], '', [ param[2:] + '=' for param in sys.argv[1::2] ])[0]:
	if opt == '--model' and arg != '': arg_weights = arg
	if opt == '--first' and arg != '': arg_frame1 = arg
	if opt == '--second' and arg != '': arg_frame2 = arg
	if opt == '--out' and arg != '': arg_out = arg

#######################################

def interpolate(arg_frame1, arg_frame2, arg_out):
    # Read input images and check dimensions
    img1 = cv.imread(arg_frame1, cv.IMREAD_GRAYSCALE).astype('float32') / 255.
    img2 = cv.imread(arg_frame2, cv.IMREAD_GRAYSCALE).astype('float32') / 255.

    assert img1.shape == img2.shape
    shape = img1.shape

    img1 = img1.reshape((1,1,shape[0],shape[1]))
    img2 = img2.reshape((1,1,shape[0],shape[1]))

    # Create input tensor and compute output tensor
    tensor_in = torch.tensor( np.concatenate((img1,img2),axis=1) ).to(device)
    tensor_out = model(tensor_in)

    # Save output image from the output tensor
    img_out = (tensor_out[0,0].cpu().detach().numpy() * 255).astype('int')
    cv.imwrite(arg_out, img_out)

#######################################

# Create model for interpolation
model = Model().to(device)
model.load_state_dict(torch.load(arg_weights, map_location=device))
model.eval()

#######################################

if __name__ == '__main__':
    interpolate(arg_frame1, arg_frame2, arg_out)