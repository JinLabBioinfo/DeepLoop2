import torch
import torch.nn as nn


def conv9(in_ch, out_ch):
    # 9x9 with padding=4 keeps H,W unchanged
    return nn.Conv2d(in_ch, out_ch, kernel_size=9, padding='same', bias=True)


class DenseUNet(nn.Module):
    """
    Encoder (dense inputs, downsampled each level):
      level0: x (1) -> e1 (4); dense_in2 = cat(e1, x)            => 5   @ 128
      level1: pool(dense_in2)=h2 (5) -> e2 (8); dense_in3=cat(e2,h2) => 13 @ 64
      level2: pool(dense_in3)=h3 (13)-> e3 (16); dense_in4=cat(e3,h3)=> 29 @ 32
      level3: pool(dense_in4)=h4 (29)-> e4 (32); dense_in5=cat(e4,h4)=> 61 @ 16
      level4: pool(dense_in5)=h5 (61)-> e5 (64)                    @ 8

    Decoder:
      d4 @16: up(e5)=64 + dense_in5=61  => 125 -> 32 -> (32+61=93)->32 ->32
      d3 @32: up(d4)=32 + dense_in4=29  => 61  -> 16 -> (16+29=45)->16 ->16
      d2 @64: up(d3)=16 + dense_in3=13  => 29  -> 8  -> (8+13=21)->8  ->8
      d1 @128: up(d2)=8 + dense_in2=5   => 13  -> 4  -> (4+5=9)->4   ->4

    Final:
      out_in = cat(d2 (8@64), dense_in2 pooled once (5@64)) => 13 -> 1x1 -> out
    """

    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.act = nn.LeakyReLU(inplace=True)

        # Encoder
        # stage1: 1 -> 4 -> 4
        self.e1a = conv9(1, 4)     # conv2d_1
        self.e1b = conv9(4, 4)     # conv2d_2

        # stage2: 5 -> 8 -> 8
        self.e2a = conv9(5, 8)     # conv2d_3
        self.e2b = conv9(8, 8)     # conv2d_4

        # stage3: 13 -> 16 -> 16
        self.e3a = conv9(13, 16)   # conv2d_5
        self.e3b = conv9(16, 16)   # conv2d_6

        # stage4: 29 -> 32 -> 32
        self.e4a = conv9(29, 32)   # conv2d_7
        self.e4b = conv9(32, 32)   # conv2d_8

        # stage5: 61 -> 64 -> 64
        self.e5a = conv9(61, 64)   # conv2d_9
        self.e5b = conv9(64, 64)   # conv2d_10

        # Decoder
        # d4 @16: 125 -> 32 ; 93 -> 32 ; 32 -> 32
        self.d4a = conv9(125, 32)  # conv2d_11
        self.d4b = conv9(93, 32)   # conv2d_12
        self.d4c = conv9(32, 32)   # conv2d_13

        # d3 @32: 61 -> 16 ; 45 -> 16 ; 16 -> 16
        self.d3a = conv9(61, 16)   # conv2d_14
        self.d3b = conv9(45, 16)   # conv2d_15
        self.d3c = conv9(16, 16)   # conv2d_16

        # d2 @64: 29 -> 8 ; 21 -> 8 ; 8 -> 8
        self.d2a = conv9(29, 8)    # conv2d_17
        self.d2b = conv9(21, 8)    # conv2d_18
        self.d2c = conv9(8, 8)     # conv2d_19

        # d1 @128: 13 -> 4 ; 9 -> 4 ; 4 -> 4
        self.d1a = conv9(13, 4)    # conv2d_20
        self.d1b = conv9(9, 4)     # conv2d_21
        self.d1c = conv9(4, 4)     # conv2d_22

        # final 13 -> out_channels with 1x1
        self.out = nn.Conv2d(13, out_channels, kernel_size=1, bias=True)  # conv2d_23

    def forward(self, x):
        # Encoder level 0 (H)
        e1 = self.act(self.e1a(x))
        e1 = self.act(self.e1b(e1))                 # 4 @ H
        dense_in2 = torch.cat([e1, x], dim=1)       # 5 @ H

        #Encoder level 1 (H/2)
        h2 = self.pool(dense_in2)                   # 5 @ H/2
        e2 = self.act(self.e2a(h2))
        e2 = self.act(self.e2b(e2))                 # 8 @ H/2
        dense_in3 = torch.cat([e2, h2], dim=1)      # 13 @ H/2

        #Encoder level 2 (H/4)
        h3 = self.pool(dense_in3)                   # 13 @ H/4
        e3 = self.act(self.e3a(h3))
        e3 = self.act(self.e3b(e3))                 # 16 @ H/4
        dense_in4 = torch.cat([e3, h3], dim=1)      # 29 @ H/4

        #Encoder level 3 (H/8)
        h4 = self.pool(dense_in4)                   # 29 @ H/8
        e4 = self.act(self.e4a(h4))
        e4 = self.act(self.e4b(e4))                 # 32 @ H/8
        dense_in5 = torch.cat([e4, h4], dim=1)      # 61 @ H/8

        #Encoder level 4 (H/16) bottom 
        h5 = self.pool(dense_in5)                   # 61 @ H/16
        e5 = self.act(self.e5a(h5))
        e5 = self.act(self.e5b(e5))                 # 64 @ H/16


        #Decoder level 3 (H/8)
        u4 = self.up(e5)                            # 64 @ H/8
        d4_in = torch.cat([u4, dense_in5], dim=1)   # 125 @ H/8
        d4 = self.act(self.d4a(d4_in))              # 32 @ H/8
        d4 = self.act(self.d4b(torch.cat([d4, dense_in5], dim=1)))  # 93 -> 32 @ H/8
        d4 = self.act(self.d4c(d4))                 # 32 @ H/8

        #Decoder level 2 (H/4)
        u3 = self.up(d4)                            # 32 @ H/4
        d3_in = torch.cat([u3, dense_in4], dim=1)   # 61 @ H/4
        d3 = self.act(self.d3a(d3_in))              # 16 @ H/4
        d3 = self.act(self.d3b(torch.cat([d3, dense_in4], dim=1)))  # 45 -> 16 @ H/4
        d3 = self.act(self.d3c(d3))                 # 16 @ H/4

        #Decoder level 1 (H/2)
        u2 = self.up(d3)                            # 16 @ H/2
        d2_in = torch.cat([u2, dense_in3], dim=1)   # 29 @ H/2
        d2 = self.act(self.d2a(d2_in))              # 8 @ H/2
        d2 = self.act(self.d2b(torch.cat([d2, dense_in3], dim=1)))  # 21 -> 8 @ H/2
        d2 = self.act(self.d2c(d2))                 # 8 @ H/2

        #Decoder level 0 (H)
        u1 = self.up(d2)                            # 8 @ H
        d1_in = torch.cat([u1, dense_in2], dim=1)   # 13 @ H
        d1 = self.act(self.d1a(d1_in))              # 4 @ H
        d1 = self.act(self.d1b(torch.cat([d1, dense_in2], dim=1)))  # 9 -> 4 @ H
        d1 = self.act(self.d1c(d1))                 # 4 @ H

        # Final (13 -> 1)
        dense_in2_h2 = self.pool(dense_in2)         # 5 @ H/2
        out_in = torch.cat([d2, dense_in2_h2], dim=1)  # 8 + 5 = 13 @ H/2
        logits = self.out(out_in)                   # 1 @ H/2
        logits = self.up(logits)                    # back to H

        return logits, None, 0
