import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import math
import numpy as np
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)
def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=True)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        # out += residual
        out = self.relu(out)
        return out

class cnn(nn.Module):
    def __init__(self, inplanes, planes):
        super(cnn, self).__init__()
        hidden = 100
        self.conv1 = conv3x3(inplanes, hidden, stride=1)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.layer1 = self._make_layer(BasicBlock, hidden, 2)
        self.convlast = conv3x3(hidden, planes, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, planes, blocks):
        layers = []
        for i in range(0, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.convlast(x)
        return x

class cnn2_pad(nn.Module):
    def __init__(self, inplanes, planes):
        super(cnn2_pad, self).__init__()
        self.inplanes = inplanes
        if inplanes==122:
            #self.part0_29 = cnn(30, 10)
            #self.part30_59 = cnn(30, 10)
            #self.part60_89 = cnn(30, 10)
            #self.part90_119 = cnn(30, 10)
            #self.part120 = cnn(1, 10)
            #self.part121 = cnn(1, 10)
            #insize = 60
            self.part1 = cnn(122,256)
            insize = 256
        elif inplanes==62:
            self.part0_29 = cnn(30, 20)
            self.part30_59 = cnn(30, 20)
            self.part120 = cnn(1, 20)
            self.part121 = cnn(1, 20)
            insize = 80
        #self.conv1 = conv3x3(insize, 100, stride=1)
        self.conv1 = conv3x3(insize, 512, stride=1)
        self.conv2 = conv3x3(512, planes, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, planes, blocks):
        layers = []
        for i in range(0, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        #x1 = self.part0_29(x[:, 0:30])
        #x1 = self.part0_29(x[:, 0:30])
        #x2 = self.part30_59(x[:, 30:60])
        #x5 = self.part120(x[:, 60:61])
        #x6 = self.part121(x[:, 61:62])
        #x5 = self.part120(x[:, 120:121])
        #x6 = self.part121(x[:, 121:122])
        if self.inplanes==62:
            x = self.relu(torch.cat([x1,x2,x5,x6],1))
        else:
            #x3 = self.part60_89(x[:, 60:90])
            #x4 = self.part90_119(x[:, 90:120])
            #x = self.relu(torch.cat([x1, x2, x3, x4, x5, x6], 1))
            #x = self.relu(torch.cat([x1, x2, x3, x4, x5, x6], 1))
            #x = self.relu(torch.cat([x1, x2, x3, x4, x5, x6], 1))
            pass
        x = self.part1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x[:, :, :, 20:-20]
        return x


if __name__ == '__main__':
    x = torch.randn(1,122,96,184)
    cnn_f = cnn2_pad(122,30)
    y = cnn_f(x)
    print(y.size())

    # 122->512->*7->30
