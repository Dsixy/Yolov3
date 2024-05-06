import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class YoloBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YoloBlock, self).__init__()
        self.conv1 = CNNBlock(in_channels, out_channels, 1, 1)
        self.conv2 = CNNBlock(out_channels, out_channels * 2, 3, 1)
        self.conv3 = CNNBlock(out_channels * 2, out_channels, 1, 1)
        self.conv4 = CNNBlock(out_channels, out_channels * 2, 3, 1)
        self.conv5 = CNNBlock(out_channels * 2, out_channels, 1, 1)

    def forward(self, x):
        return self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = CNNBlock(channels, channels // 2, 1, 1)
        self.conv2 = CNNBlock(channels // 2, channels, 3, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ResidualOperator(nn.Module):
    def __init__(self, in_channels, repeat):
        super(ResidualOperator, self).__init__()
        self.layers = nn.ModuleList()
        self.repeat = repeat
        self.conv = CNNBlock(in_channels, in_channels * 2, 3, 2)
        for i in range(self.repeat):
            self.layers += [
                ResidualBlock(in_channels * 2)
            ]

    def forward(self, x):
        x = self.conv(x)
        for layer in self.layers:
            x = layer(x)
        return x


class DarkNet53(nn.Module):
    def __init__(self, in_channels):
        super(DarkNet53, self).__init__()
        self.conv = CNNBlock(in_channels, 32, 3, 1)
        self.res1 = ResidualOperator(32, 1)
        self.res2 = ResidualOperator(64, 2)
        self.res3 = ResidualOperator(128, 8)
        self.res4 = ResidualOperator(256, 8)
        self.res5 = ResidualOperator(512, 4)

    def forward(self, x):
        out1 = self.res3(self.res2(self.res1(self.conv(x))))
        out2 = self.res4(out1)
        out3 = self.res5(out2)
        return out3, out2, out1


class YOLOv3(nn.Module):
    def __init__(self, in_channels, class_num):
        super(YOLOv3, self).__init__()
        self.darknet = DarkNet53(in_channels)
        self.yolo_block1 = YoloBlock(1024, 512)
        self.yolo_block2 = YoloBlock(768, 256)
        self.yolo_block3 = YoloBlock(384, 128)
        self.conv11 = CNNBlock(512, 1024, 3, 1)
        self.conv12 = CNNBlock(1024, 3 * (5 + class_num), 1, 1)
        self.conv21 = CNNBlock(256, 512, 3, 1)
        self.conv22 = CNNBlock(512, 3 * (5 + class_num), 1, 1)
        self.conv31 = CNNBlock(128, 256, 3, 1)
        self.conv32 = CNNBlock(256, 3 * (5 + class_num), 1, 1)
        self.up_sample1 = nn.Upsample(scale_factor=2)
        self.up_sample2 = nn.Upsample(scale_factor=2)
        self.conv1 = CNNBlock(512, 256, 1, 1)
        self.conv2 = CNNBlock(256, 128, 1, 1)

    def forward(self, x):
        x1, x2, x3 = self.darknet(x)
        out1 = self.yolo_block1(x1)
        out2 = torch.cat((self.up_sample1(self.conv1(out1)), x2), dim=1)
        out2 = self.yolo_block2(out2)
        out3 = self.yolo_block3(torch.cat((self.up_sample2(self.conv2(out2)), x3), dim=1))
        out1 = self.conv12(self.conv11(out1))
        out2 = self.conv22(self.conv21(out2))
        out3 = self.conv32(self.conv31(out3))
        return out1, out2, out3

