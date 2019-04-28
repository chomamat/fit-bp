import torch
import torch.nn as nn
import torchvision

class VggLoss(nn.Module):
    def __init__(self, vgg_factor):
        super(VggLoss, self).__init__()

        self.vgg_factor = vgg_factor
        model = torchvision.models.vgg19(pretrained=True).cuda()

        self.features = nn.Sequential(
            # stop at relu4_4 [:-10]
            *list(model.features.children())[:-10]
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
    def upChannel(x):
        out = x

        while out.shape[1] < 3:
            out = torch.cat((out,x),1)

        return out

class CombinedLoss(nn.Module):
    def __init__(self, vgg_factor):
        super(CombinedLoss, self).__init__()
        self.vgg = VggLoss(vgg_factor)
        self.l1 = nn.L1Loss()

    def forward(self, output, target) -> torch.Tensor:
        return self.vgg(output, target) + self.l1(output, target)