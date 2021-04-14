import torch
from torch import nn
import torchvision 
torchvision.models.resnet50

def conv3x3(input_channels, out_channels, stride=2, groups=1, dilation=1):
    """
        3x3 convolution with padding
    """
    return nn.Conv2d(input_channels, out_channels, kernel_size=3,stride=stride, padding=dilation
                    groups=groups, dilation=dilation, bias=False)

def conv1x1(input_channels, output_channels, stride=1):
    """
        1x1 convolution
    """
    return nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):
    def __init__(self, blocks_args, global_params)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, downsample=None, groups=1, 
                base_width=64, dilation=1, norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResidualBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResidualBlock")
        
        self.conv1 = conv3x3(input_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out