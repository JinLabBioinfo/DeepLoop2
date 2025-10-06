import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, first_conv_kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=first_conv_kernel_size, padding='same', bias=True),
            #nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


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
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
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
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
#@torch.compile

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_filters=32, fan=2, bilinear=False, **vq_kwargs):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_filters = init_filters
        self.fan = fan
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, init_filters, first_conv_kernel_size=7))
        self.vq1 = VectorQuantize(init_filters, accept_image_fmap = True, **vq_kwargs)
        self.down1 = (Down(init_filters, init_filters * fan))
        self.vq2 = VectorQuantize(init_filters * fan, accept_image_fmap = True, **vq_kwargs)
        self.down2 = (Down(init_filters * fan, init_filters * fan * 2))
        self.vq3 = VectorQuantize(init_filters * fan * 2, accept_image_fmap = True, **vq_kwargs)
        self.down3 = (Down(init_filters * fan * 2, init_filters * fan * 4))
        self.vq4 = VectorQuantize(init_filters * fan * 4, accept_image_fmap = True, **vq_kwargs)

        factor = 2 if bilinear else 1
        self.down4 = (Down(init_filters * fan * 4, init_filters * fan * 8 // factor))
        self.vq = VectorQuantize(init_filters * fan * 8 // factor, accept_image_fmap = True, **vq_kwargs)
        self.up1 = (Up(init_filters * fan * 8, init_filters * fan * 4 // factor, bilinear))
        self.up2 = (Up(init_filters * fan * 4, init_filters * fan * 2 // factor, bilinear))
        self.up3 = (Up(init_filters * fan * 2, init_filters * fan // factor, bilinear))
        self.up4 = (Up(init_filters * fan, init_filters, bilinear))
        self.outc = (OutConv(init_filters, n_classes))

    def forward(self, x):
        x1, idx1, c1 = self.vq1(self.inc(x))
        x2, idx2, c2 = self.vq2(self.down1(x1))
        x3, idx3, c3 = self.vq3(self.down2(x2))
        x4, idx4, c4 = self.vq4(self.down3(x3))
        x5, idx5, c5 = self.vq(self.down4(x4))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        batch = logits.size(0)
        all_idx = torch.cat([
            idx1.view(batch, -1),
            idx2.view(batch, -1),
            idx3.view(batch, -1),
            idx4.view(batch, -1),
            idx5.view(batch, -1),
        ], dim=1)
        total_cmt = c1 + c2 + c3 + c4 + c5
        #indices = [indices1, indices2, indices3, indices4, indices]
        #indices = torch.concat([idxs.view(x.shape[0], -1) for idxs in indices], dim=1)
        # return logits, indices, cmt_loss + cmt_loss1 + cmt_loss2 + cmt_loss3 + cmt_loss4
        return logits, all_idx, total_cmt
