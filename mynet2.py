from torchsummary import summary
import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),

            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),

        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride),
        )

        self.sig = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out) * x2
        out = self.sig(out)

        return out


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

start_fm = 128

class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()

        # down scale
        # Convolution 1
        self.double_conv1 = double_conv(3, start_fm, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.double_conv2 = double_conv(start_fm, start_fm * 2, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Convolution 3
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # Convolution 4
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Convolution 5 (Bridge)
        self.double_conv7 = double_conv(start_fm * 8, start_fm * 16, 3, 1, 1)

        # Expanding Path Convolution 5 #NOT NEEDED IN TRAINING ERROR IN LAYER
        self.ex_double_conv5 = double_conv(start_fm * 32, start_fm * 16, 3, 1, 1)

        # Transposed Convolution 4
        self.attn2 = AttentionBlock(start_fm * 8, start_fm * 16, 3, 1, 1)

        self.t_conv4 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, 2, 2)
        # Expanding Path Convolution 4
        self.ex_double_conv4 = double_conv(start_fm * 16, start_fm * 8, 3, 1, 1)

        # Transposed Convolution 3
        self.attn3 = AttentionBlock(start_fm * 4, start_fm * 8, 3, 1, 1)

        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)
        # Convolution 3
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4, 3, 1, 1)

        # Transposed Convolution 2
        self.attn4 = AttentionBlock(start_fm * 2, start_fm * 4, 3, 1, 1)

        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)
        # Convolution 2
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2, 3, 1, 1)

        # Transposed Convolution 1
        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)
        # Convolution 1
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm, 3, 1, 1)

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

        #Bridge
        conv7 = self.double_conv7(maxpool4)

        # Expanding Path
        attn2 = self.attn2(conv4, conv7)
        t_conv4 = self.t_conv4(attn2)
        cat4 = torch.cat([conv4, t_conv4], 1)
        ex_conv4 = self.ex_double_conv4(cat4)

        attn3 = self.attn3(conv3, ex_conv4)
        t_conv3 = self.t_conv3(attn3)
        cat3 = torch.cat([conv3, t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        attn4 = self.attn4(conv2, ex_conv3)
        t_conv2 = self.t_conv2(attn4)
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