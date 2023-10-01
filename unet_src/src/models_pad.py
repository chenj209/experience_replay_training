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

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
#         print(x.shape,'x.shape')#torch.Size([2, 64, 96, 184])
        c1 = self.conv1(x)
#         print(c1.shape,'c1.shape')#torch.Size([2, 64, 96, 184])
        p1 = self.pool1(c1)
#         print(p1.shape,'p1.shape')#torch.Size([2, 64, 48, 92])
        c2 = self.conv2(p1)
#         print(c2.shape,'c2.shape')#torch.Size([2, 64, 48, 92])
        p2 = self.pool2(c2)
#         print(p2.shape,'p2.shape')#torch.Size([2, 128, 24, 46])
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)#torch.Size([2, 128, 12, 23])
        
        up_8 = self.up8(c3)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out
    
    
class cnn2_pad(nn.Module):
    def __init__(self, inplanes, planes):
        super(cnn2_pad, self).__init__()
        self.inplanes = inplanes
        if inplanes==122:
            self.part0_29 = cnn(30, 10)
            self.part30_59 = cnn(30, 10)
            self.part60_89 = cnn(30, 10)
            self.part90_119 = cnn(30, 10)
            self.part120 = cnn(1, 10)
            self.part121 = cnn(1, 10)
            insize = 60
        elif inplanes==62:
            self.part0_29 = cnn(30, 20)
            self.part30_59 = cnn(30, 20)
            self.part120 = cnn(1, 20)
            self.part121 = cnn(1, 20)
            insize = 80
        self.conv1 = conv3x3(insize, 64, stride=1)
        self.conv2 = conv3x3(64, planes, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.unet = Unet(64,64)

    def _make_layer(self, block, planes, blocks):
        layers = []
        for i in range(0, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.part0_29(x[:, 0:30])
        x2 = self.part30_59(x[:, 30:60])
        x5 = self.part120(x[:, 120:121])
        x6 = self.part121(x[:, 121:122])
        if self.inplanes==62:
            x = self.relu(torch.cat([x1,x2,x5,x6],1))
        else:
            x3 = self.part60_89(x[:, 60:90])
            x4 = self.part90_119(x[:, 90:120])
            x = self.relu(torch.cat([x1, x2, x3, x4, x5, x6], 1))
        #x shape torch.Size([2, 60, 96, 184])
        x = self.conv1(x)
        x = self.relu(x)
#         x = self.conv2(x)
        print#(x.shape) batch planes 96 184
        self.unet(x)
        x = self.conv2(x)
#         print(x.shape)
        x = x[:, :, :, 20:-20]
        return x


if __name__ == '__main__':
    x = torch.randn(1,122,96,184)
    unet_f = unet_pad(122,30)
    y = unet_f(x)
    print(y.size())
    # 两次max pooling keep lon lat dimesion
    # classic unet
