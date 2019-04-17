import getopt

import cv2 as cv
import numpy as np
import sys
import torch
import torch.nn as nn

# Device for running computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Not computing gradients for better computationl performance
torch.set_grad_enabled(False)

# Parse script arguments
arg_weights = "model_cl_2e-5.pytorch"
arg_frame1 = "examples/03_1.png"
arg_frame2 = "examples/03_3.png"
arg_out = "examples/out.png"

for opt, arg in getopt.getopt(sys.argv[1:], '', [ param[2:] + '=' for param in sys.argv[1::2] ])[0]:
	if opt == '--model' and arg != '': arg_weights = arg
	if opt == '--first' and arg != '': arg_frame1 = arg
	if opt == '--second' and arg != '': arg_frame2 = arg
	if opt == '--out' and arg != '': arg_out = arg

#######################################

class Model(nn.Module):
    def __init__(self, weights=None):
        super(Model, self).__init__()
        
        self.activation = nn.PReLU()
        
        self.conv_setup = {
            'kernel' : (3,3),
            'stride' : (1,1),
            'padding' : 1,
            'activation' : self.activation
        }
        self.pooling_setup = {
            'kernel_size' : (2,2),
            'stride' : (2,2)
        }
        self.upsample_setup = {
            'scale_factor' : 2,
            'mode' : 'bilinear',
            'align_corners' : True
        }

        self.pooling_layer = nn.AvgPool2d(**self.pooling_setup)
        self.upsample_layer = nn.Upsample(**self.upsample_setup)
        
        self.conv32 = self._convBlock(2, 32, **self.conv_setup)
        self.conv64 = self._convBlock(32, 64, **self.conv_setup)
        self.conv128 = self._convBlock(64, 128, **self.conv_setup)
        self.conv256 = self._convBlock(128, 256, **self.conv_setup)
        self.conv256_256 = self._convBlock(256, 256, **self.conv_setup)


        self.upsample256 = self._upsampleBlock(self.upsample_layer, 256, 256, **self.conv_setup)
        self.deconv128 = self._convBlock(256, 128, **self.conv_setup)
        self.upsample128 = self._upsampleBlock(self.upsample_layer, 128, 128, **self.conv_setup)
        self.deconv64 = self._convBlock(128, 64, **self.conv_setup)
        self.upsample64 = self._upsampleBlock(self.upsample_layer, 64, 64, **self.conv_setup)
        self.deconv32 = self._convBlock(64, 32, **self.conv_setup)
        self.upsample32 = self._upsampleBlock(self.upsample_layer, 32, 32, **self.conv_setup)
        self.deconv1 = self._convBlock(32, 1, kernel=(3,3), stride=(1,1), padding=1, activation=None)
        
        if weights is not None:
        	self.load_state_dict(torch.load(weights, map_location=device))

    def forward(self, x):
        x32 = self.conv32(x)
        x32_p = self.pooling_layer(x32)
        x64 = self.conv64(x32_p)
        x64_p = self.pooling_layer(x64)
        x128 = self.conv128(x64_p)
        x128_p = self.pooling_layer(x128)
        x256 = self.conv256(x128_p)
        x256_p = self.pooling_layer(x256)

        x = self.conv256_256(x256_p)

        # expansion

        x = self.upsample256(x)
        x += x256
        x = self.deconv128(x)

        x = self.upsample128(x)
        x += x128
        x = self.deconv64(x)

        x = self.upsample64(x)
        x += x64
        x = self.deconv32(x)
        
        x = self.upsample32(x)
        x += x32
        x = self.deconv1(x)
        
        return x
    
    @staticmethod
    def _convBlock(in_channels, out_channels, kernel, stride, padding, activation):
        net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel, stride, padding), nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel, stride, padding), nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        )
        if activation is not None:
            net = nn.Sequential(
                net, 
                nn.BatchNorm2d(out_channels),
                activation
            )
        return net
    @staticmethod
    def _upsampleBlock(upsample, in_channels, out_channels, kernel, stride, padding, activation):
        return nn.Sequential(
            upsample,
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding), activation
        )

#######################################

# Create model for interpolation
model = Model(arg_weights).to(device).eval()

# Read input images and check dimensions
img1 = cv.imread(arg_frame1, cv.IMREAD_GRAYSCALE).astype('float32') / 255.
img2 = cv.imread(arg_frame2, cv.IMREAD_GRAYSCALE).astype('float32') / 255.

assert img1.shape == img2.shape == (96,96)

img1 = img1.reshape((1,1,96,96))
img2 = img2.reshape((1,1,96,96))

# Create input tensor and compute output tensor
tensor_in = torch.tensor( np.concatenate((img1,img2),axis=1) )
tensor_out = model(tensor_in)

# Save output image from the output tensor
img_out = (tensor_out[0,0].cpu().detach().numpy() * 255).astype('int')
cv.imwrite(arg_out, img_out)
