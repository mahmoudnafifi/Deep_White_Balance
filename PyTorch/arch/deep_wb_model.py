"""
 Constructs network architecture
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

from .deep_wb_blocks import *


class deepWBNet(nn.Module):
    def __init__(self):
        super(deepWBNet, self).__init__()
        self.n_channels = 3
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_down2 = DownBlock(48, 96)
        self.encoder_down3 = DownBlock(96, 192)
        self.encoder_bridge_down = BridgeDown(192, 384)
        self.awb_decoder_bridge_up = BridgeUP(384, 192)
        self.awb_decoder_up1 = UpBlock(192, 96)
        self.awb_decoder_up2 = UpBlock(96, 48)
        self.awb_decoder_up3 = UpBlock(48, 24)
        self.awb_decoder_out = OutputBlock(24, self.n_channels)
        self.tungsten_decoder_bridge_up = BridgeUP(384, 192)
        self.tungsten_decoder_up1 = UpBlock(192, 96)
        self.tungsten_decoder_up2 = UpBlock(96, 48)
        self.tungsten_decoder_up3 = UpBlock(48, 24)
        self.tungsten_decoder_out = OutputBlock(24, self.n_channels)
        self.shade_decoder_bridge_up = BridgeUP(384, 192)
        self.shade_decoder_up1 = UpBlock(192, 96)
        self.shade_decoder_up2 = UpBlock(96, 48)
        self.shade_decoder_up3 = UpBlock(48, 24)
        self.shade_decoder_out = OutputBlock(24, self.n_channels)

    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        x_awb = self.awb_decoder_bridge_up(x5)
        x_awb = self.awb_decoder_up1(x_awb, x4)
        x_awb = self.awb_decoder_up2(x_awb, x3)
        x_awb = self.awb_decoder_up3(x_awb, x2)
        awb = self.awb_decoder_out(x_awb, x1)
        x_t = self.tungsten_decoder_bridge_up(x5)
        x_t = self.tungsten_decoder_up1(x_t, x4)
        x_t = self.tungsten_decoder_up2(x_t, x3)
        x_t = self.tungsten_decoder_up3(x_t, x2)
        t = self.tungsten_decoder_out(x_t, x1)
        x_s = self.shade_decoder_bridge_up(x5)
        x_s = self.shade_decoder_up1(x_s, x4)
        x_s = self.shade_decoder_up2(x_s, x3)
        x_s = self.shade_decoder_up3(x_s, x2)
        s = self.shade_decoder_out(x_s, x1)
        return torch.cat((awb, t, s), dim=1)
