import Worms_Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/milesial/Pytorch-UNet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels # todo: why to do mid channels?
        self.double_conv = nn.Sequential(
            # todo:NOT IMPORTANT why 3x3 conv?(besides its in the article)
            # todo: it named kernel because a kernel trick ?
            # (The trick is to replace this inner product with another one, without even knowing Φ)
            # todo: how do we calculate our bias ?
            # todo: what more feature do i need to calculate and how (EX VX etc) and if BatchNorm2d do it for me?
            # todo: why we use BatchNorm2d isnt relu Regularizes the values(wasnt on the papaer architecture)? “change in the distribution of network activations due to the change in network parameters during training” (Ioffe & Szegedy, 2015).,"reduces the internal covariate shif"?
            # todo: all this sequential is equivalent to two blue right arrows at the papers graph ?
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

    # todo: this equivalent to one red arrow down and two blue in the papers graph?
    # todo: why to split the architecture into separated class?

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # todo: whats bilinear? is it bilinear interpolation? the image are single channel what channels is being reduced the features map?
        # todo:  downsample is for avoid overfiting? by deleting some part of our data isnt that give us bias? why not putting random values insted?
        #  downsampleis this the regulaztion term discussed in 231n?
        # todo: why different usage in Upsampling and convtranspose when what and why?
        # Upsampling is the process of inserting zero-valued samples between original samples to increase the sampling rate
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # todo: is this the two gray arrows in papers graph ? in article he didnt pad valid conv value here
        # he do different from the paper or only for concat?
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    # todo: do we have last fully coneceted layer? is it here? this is the last feature or the middle one?
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#########################################################################################


""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        # todo: what we do here in classes we dont have labels
        self.n_classes = n_classes
        # todo: again whats bilinear?
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        #todo: factor?
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # todo: go over this with the graph
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    # todo: what i need to change here?
    # todo: how will this connect to my wormdataset?
    # todo: chossing regularization?
    # todo:gradient checking?
    # todo: what initialization we choose for filters and why?
    # todo: to low can make gradint colapse to zero to high will stick to supreme and minima cs231n says np.random(n)*sqrt(2/n)
    # todo: why relu and not other?(softmax etc)