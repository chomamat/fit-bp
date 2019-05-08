import torch
import torch.nn as nn
import torchvision
from skimage.measure import compare_ssim as ssim
import numpy as np

class VggLoss(nn.Module):
    def __init__(self, output_layer=-10, vgg_factor=0.00002):
        super(VggLoss, self).__init__()

        self.vgg_factor = vgg_factor
        model = torchvision.models.vgg19(pretrained=True).cuda()

        self.features = nn.Sequential(
            # stop at relu4_4 [:-10]
            *list(model.features.children())[:output_layer]
        )

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        O = self.upChannel(output)
        T = self.upChannel(target)
        outputFeatures = self.features(O)
        targetFeatures = self.features(T)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # review this norm, what does that mean?
        loss = torch.norm(outputFeatures - targetFeatures, 2)

        return self.vgg_factor * loss

    # VGG19 works with 3 channels, but radar images are only grayscale.
    # It empirically looks to be OK, to just concatenate the image in the channels dimension.
    def upChannel(self,x):
        out = x

        while out.shape[1] < 3:
            out = torch.cat((out,x),1)

        return out

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, output, target) -> torch.Tensor:
        assert output.shape[0] == target.shape[0] == 1
        assert output.shape == target.shape
        def_shape = output.shape
        
        O = output.view(def_shape[2],def_shape[3],-1).cpu().detach().numpy()
        T = target.view(def_shape[2],def_shape[3],-1).cpu().detach().numpy()
        
        if output.shape[1] == 3:
            multichannel=False
            return ssim(O, T, multichannel=True)
        else:
            return ssim(np.squeeze(O), np.squeeze(T))
    
class CombinedLoss(nn.Module):
    def __init__(self, vgg_layer=-18, vgg_factor=0.00002):
        super(CombinedLoss, self).__init__()
        self.vgg = VggLoss(vgg_layer, vgg_factor)
        self.l1 = nn.L1Loss()

    def forward(self, output, target) -> torch.Tensor:
        return self.vgg(output, target) + self.l1(output, target)