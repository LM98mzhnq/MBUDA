import torch
from torch import nn
import torch.nn.functional as F

class DUAL(nn.Module):
    def __init__(self, in_channels,ndf=512):
        super(DUAL, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out1 = self.branch1(x)
        out1 += identity
        out1 = self.relu1(out1)

        identity = out1
        out2 = self.branch2(out1)
        out2 += identity
        out2 = self.relu2(out2)


        return out2