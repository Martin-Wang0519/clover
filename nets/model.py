import torch
import torch.nn as nn
from collections import OrderedDict


class CBL(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False
                              )
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class ResUnit(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResUnit, self).__init__()
        self.down_sample = None
        self.stride = 1
        if in_channel != out_channel:
            self.down_sample = nn.Sequential(nn.Conv2d(in_channel,
                                                       out_channel,
                                                       kernel_size=1,
                                                       stride=2,
                                                       bias=False),
                                             nn.BatchNorm2d(out_channel)
                                             )
            self.stride = 2

        self.CBL1 = CBL(in_channel, out_channel, (3, 3), (self.stride, self.stride), 1)
        self.CBL2 = CBL(out_channel, out_channel, (3, 3), (1, 1), 1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)

        x = self.CBL1(x)
        x = self.CBL2(x)
        x = x + identity
        x = self.relu(x)
        return x


# 神经网络结构
class ResNet(torch.nn.Module):
    def __init__(self, ResUnit_nums, classes_num):
        super(ResNet, self).__init__()

        self.layers_out_filters = [32, 64, 128, 256, 512]
        self.conv1 = nn.Conv2d(3,
                               self.layers_out_filters[0],
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.layers_out_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.current_channel = self.layers_out_filters[0]

        self.layer1 = self.make_layers(ResUnit_nums[0], self.layers_out_filters[1])
        self.layer2 = self.make_layers(ResUnit_nums[1], self.layers_out_filters[2])
        self.layer3 = self.make_layers(ResUnit_nums[2], self.layers_out_filters[3])
        self.layer4 = self.make_layers(ResUnit_nums[3], self.layers_out_filters[4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.layers_out_filters[-1], classes_num)

    def make_layers(self, ResUnit_Num, out_channel):
        layers = []
        layers.append(ResUnit(self.current_channel, out_channel))
        self.current_channel = out_channel

        for _ in range(1, ResUnit_Num):
            layers.append(ResUnit(out_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def ResNet34(classes_num):
    return ResNet([3, 4, 6, 3], classes_num)
