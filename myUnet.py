from torchsummary import summary
import torch
import torchvision.models as models
import torch.nn as nn

from unet import UNetWithResnet50Encoder
from fastai_Unet import build_res_unet


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride,padding=padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv(x)+self.res(x)

        return out

start_fm = 32

class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()

        # Input 128x128x1

        # Contracting Path

        # (Double) Convolution 1
        self.double_conv1 = double_conv(3, start_fm, 3, 1, 1)
        # Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.double_conv2 = double_conv(start_fm, start_fm * 2, 3, 1, 1)
        # Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Convolution 3
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4, 3, 1, 1)
        # Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # Convolution 4
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8, 3, 1, 1)
        # Max Pooling 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Convolution 5
        self.double_conv5 = double_conv(start_fm * 8, start_fm * 16, 3, 1, 1)
        # Max Pooling 5
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        # Convolution 6
        self.double_conv6 = double_conv(start_fm * 16, start_fm * 32, 3, 1, 1)
        # Max Pooling 6
        self.maxpool6 = nn.MaxPool2d(kernel_size=2)


        # Convolution 7 (Bridge)
        self.double_conv7 = double_conv(start_fm * 32, start_fm * 64, 3, 1, 1)


        # Transposed Convolution 6
        self.t_conv6 = nn.ConvTranspose2d(start_fm * 64, start_fm * 32, 2, 2)
        # Expanding Path Convolution 6
        self.ex_double_conv6 = double_conv(start_fm * 64, start_fm * 32, 3, 1, 1)

        # Transposed Convolution 5
        self.t_conv5 = nn.ConvTranspose2d(start_fm * 32, start_fm * 16, 2, 2)
        # Expanding Path Convolution 5
        self.ex_double_conv5 = double_conv(start_fm * 32, start_fm * 16, 3, 1, 1)

        # Transposed Convolution 4
        self.t_conv4 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, 2, 2)
        # Expanding Path Convolution 4
        self.ex_double_conv4 = double_conv(start_fm * 16, start_fm * 8, 3, 1, 1)

        # Transposed Convolution 3
        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)
        # Convolution 3
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4, 3, 1, 1)

        # Transposed Convolution 2
        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)
        # Convolution 2
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2, 3, 1, 1)

        # Transposed Convolution 1
        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)
        # Convolution 1
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm, 3, 1, 1)

        # One by One Conv
        self.one_by_one = nn.Conv2d(start_fm, 3, 1, 1, 0)
        self.final_act = nn.Sigmoid()

    def forward(self, inputs):
        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.double_conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        conv5 = self.double_conv5(maxpool4)
        maxpool5= self.maxpool5(conv5)

        conv6 = self.double_conv6(maxpool5)
        maxpool6 = self.maxpool6(conv6)

        #Bridge
        conv7 = self.double_conv7(maxpool6)

        # Expanding Path
        t_conv6 = self.t_conv6(conv7)
        cat6 = torch.cat([conv6, t_conv6], 1)
        ex_conv6 = self.ex_double_conv6(cat6)

        t_conv5 = self.t_conv5(ex_conv6)
        cat5 = torch.cat([conv5, t_conv5], 1)
        ex_conv5 = self.ex_double_conv5(cat5)

        t_conv4 = self.t_conv4(ex_conv5)
        cat4 = torch.cat([conv4, t_conv4], 1)
        ex_conv4 = self.ex_double_conv4(cat4)

        t_conv3 = self.t_conv3(ex_conv4)
        cat3 = torch.cat([conv3, t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        t_conv2 = self.t_conv2(ex_conv3)
        cat2 = torch.cat([conv2, t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)

        output = self.one_by_one(ex_conv1)
        output = self.final_act(output)

        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet()
print(model)
model = model.to(device)
summary(model, (3,128,128))