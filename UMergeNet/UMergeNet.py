#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from enum import Enum

#jupyter nbconvert --to script UMergeNet.ipynb

class AdjustChannels(nn.Module):
    """
    Adjusts the number of channels of a tensor:
    -If in_ch < out_ch: repeat channels until reaching out_ch
    -If in_ch > out_ch: keep (out_ch//2) channels and reduce the rest with a conv1x1 to (out_ch -keep)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        if in_ch > out_ch:
            self.keep = out_ch // 2
            self.reduced_out = out_ch - self.keep
            self.reduce = nn.Conv2d(in_ch - self.keep, self.reduced_out, kernel_size=1)
        else:
            self.reduce = None

    def forward(self, x):
        b, c, h, w = x.shape

        if c == self.out_ch:
            return x

        elif c < self.out_ch:
            repeat_factor = -(-self.out_ch // c)  # ceil(out_ch /c)
            return x.repeat(1, repeat_factor, 1, 1)[:, :self.out_ch]

        else:  # c > out_ch
            part1 = x[:, :self.keep]
            excess = x[:, self.keep:]
            reduced = self.reduce(excess)
            return torch.cat([part1, reduced], dim=1)


class AxialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation = 1, groups=1, bias=True, padding='same'):
        super().__init__()
        self.adjust = AdjustChannels(in_channels, out_channels)

        self.groups       = groups
        self.out_channels = out_channels
        if groups == out_channels: #Dw
            self.dw_h   = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=padding, groups=groups, dilation=dilation, bias=bias)
            self.dw_w   = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), padding=padding, groups=groups, dilation=dilation, bias=bias)
        else:    
            self.dw_h   = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=padding, groups=groups, dilation=dilation, bias=bias)
            self.dw_w   = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=padding, groups=groups, dilation=dilation, bias=bias)

    def forward(self, x):
        if self.groups == self.out_channels:
            # (If it is DepthWise)
            x = self.adjust(x)
            x = x + self.dw_h(x) + self.dw_w(x)
        else:
            x = self.adjust(x) + self.dw_h(x) + self.dw_w(x)
        return x

class ConvType(Enum):
    Axial    = 0
    Atrous   = 1
    Standard = 2
    Normal   = 2

def conv(type, in_channels, out_channels, kernel_size, dilation=1, padding='same', groups=1):
    if type == ConvType.Axial:
        return AxialConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups)
    if type == ConvType.Atrous:
        #7 becomes 5
        kernel_size -= 2
        #1 becomes 2
        dilation    += 1
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding='same', groups=groups)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding='same', groups=groups)

class EncoderTwoLanesBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type, groups=1):
        super().__init__()
        middle_channels = out_channels//2

        if groups == 'dw':
            groups = middle_channels

        self.wide   = nn.Sequential()
        self.narrow = nn.Sequential()

        self.wide.append(conv(conv_type, in_channels, middle_channels, kernel_size=7, padding='same', groups=groups))
        self.narrow.append(nn.Conv2d(    in_channels, middle_channels, kernel_size=3, padding='same', groups=groups))

        self.bn = nn.BatchNorm2d(out_channels)
        self.pw = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same')
        self.act = nn.GELU()

    def forward(self, x):
        x = torch.cat([self.wide(x), self.narrow(x)], dim=1)
        x = self.act(self.pw(self.bn(x)))
        return x

class DecoderTwoLanesBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type, groups=1):
        super().__init__()
        middle_channels = out_channels//2

        if groups == 'dw':
            groups = middle_channels


        self.pw1    = nn.Conv2d(in_channels, middle_channels, kernel_size=1, padding='same')
        self.wide   = nn.Sequential()
        self.narrow = nn.Sequential()

        self.wide.append(conv(conv_type, middle_channels, middle_channels, kernel_size=7, padding='same', groups=groups))
        self.narrow.append(    nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding='same', groups=groups))

        self.bn = nn.BatchNorm2d(out_channels)
        self.pw2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same')
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([self.wide(x), self.narrow(x)], dim=1)
        x = self.act(self.pw2(self.bn(x)))
        return x

class MergerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, conv_type, groups=4):
        super().__init__()

        if groups == 'dw':
            groups = out_channels


        self.pw1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.convs = nn.Sequential()
        for i in range(1):
            self.convs.append(conv(conv_type, out_channels, out_channels, kernel_size=7, padding='same', groups=groups))
            self.convs.append(nn.BatchNorm2d(out_channels))
            self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same'))
            self.convs.append(nn.GELU())

        for i in range(num_layers):
            self.convs.append(conv(conv_type, out_channels, out_channels, kernel_size=7, padding='same', groups=groups))
            self.convs.append(nn.BatchNorm2d(out_channels))
            self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same'))
            self.convs.append(nn.GELU())

    def forward(self, x):
        x = self.pw1(x)
        x = self.convs(x)
        return x


class UMergeNet(nn.Module):

    def __init__(self, in_channels, out_channels, layer1=16, layer2=32, layer3=64, layer4=128, layer5=256,
                       encoder_groups=4, merger_groups=4, decoder_groups=4, conv_type=ConvType.Axial):
        super().__init__()

        self.pool         = nn.MaxPool2d(kernel_size=2)
        self.upsample     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # encoder
        self.enc1 = EncoderTwoLanesBlock(in_channels,  layer1,                        conv_type=conv_type)
        self.enc2 = EncoderTwoLanesBlock(layer1,       layer2, groups=encoder_groups, conv_type=conv_type)
        self.enc3 = EncoderTwoLanesBlock(layer2,       layer3, groups=encoder_groups, conv_type=conv_type)
        self.enc4 = EncoderTwoLanesBlock(layer3,       layer4, groups=encoder_groups, conv_type=conv_type)
        self.enc5 = EncoderTwoLanesBlock(layer4,       layer5, groups=encoder_groups, conv_type=conv_type)

        # Mergers
        self.merge1 = MergerBlock(layer1 + layer2 + layer3 + in_channels, layer3,   num_layers=0, groups=merger_groups, conv_type=conv_type)
        self.merge2 = MergerBlock(layer3 + layer4 + in_channels, layer4,            num_layers=1, groups=merger_groups, conv_type=conv_type)
        self.merge3 = MergerBlock(layer4 + layer5 + in_channels, layer5,            num_layers=2, groups=merger_groups, conv_type=conv_type)

        # Decoder
        self.dec5 = DecoderTwoLanesBlock(layer5*2,           layer5, groups=decoder_groups, conv_type=conv_type)
        self.dec4 = DecoderTwoLanesBlock(layer5+layer4*2,    layer4, groups=decoder_groups, conv_type=conv_type)
        self.dec3 = DecoderTwoLanesBlock(layer4+layer3*2,    layer3, groups=decoder_groups, conv_type=conv_type)
        self.dec2 = DecoderTwoLanesBlock(layer3+layer2,      layer2, groups=decoder_groups, conv_type=conv_type)
        self.dec1 = DecoderTwoLanesBlock(layer2+layer1,      layer1, groups=decoder_groups, conv_type=conv_type)

        # Final Layer
        self.final    = nn.Conv2d(layer1, out_channels, kernel_size=1)



    def forward(self, x):

        ## Encoder
        lo1 = self.enc1(x)
        lo1_resized = self.pool(lo1)

        x1  = self.pool(x)
        lo2 = self.enc2(lo1_resized)
        lo2_resized = self.pool(lo2)

        x2  = self.pool(x1)
        lo3 = self.enc3(lo2_resized)

        x3  = self.pool(x2)
        lo4 = self.enc4(self.pool(lo3))

        x4  = self.pool(x3)
        lo5 = self.enc5(self.pool(lo4))

        # Mergers
        lo_features = torch.cat((x2, self.pool(lo1_resized), lo2_resized, lo3), dim=1)
        lx3 = self.merge1(lo_features)

        lo_features = torch.cat((x3, self.pool(lx3), lo4), dim=1)
        lx4 = self.merge2(lo_features)

        lo_features = torch.cat((x4, self.pool(lx4), lo5), dim=1)
        lx5 = self.merge3(lo_features)

        ## Decoder
        out = torch.cat((lx5, lo5), dim=1)
        out = self.dec5(out)

        out = self.upsample(out)
        out = torch.cat((out, lx4, lo4), dim=1)
        out = self.dec4(out)

        out = self.upsample(out)
        out = torch.cat((out, lx3, lo3), dim=1)
        out = self.dec3(out)

        out = self.upsample(out)
        out = torch.cat((out, lo2), dim=1)
        out = self.dec2(out)

        out = self.upsample(out)
        out = torch.cat((out, lo1), dim=1)
        out = self.dec1(out)


        return self.final(out)

if __name__ == '__main__':
    print("Axial")
    model = UMergeNet(in_channels=3, out_channels=1)

    print("Axial-DW")
    model = UMergeNet(in_channels=3, out_channels=1, merger_groups='dw', encoder_groups='dw', decoder_groups='dw')

    print("Atrous")
    model = UMergeNet(in_channels=3, out_channels=1, conv_type=ConvType.Atrous)

    print("Atrous-DW")
    model = UMergeNet(in_channels=3, out_channels=1, conv_type=ConvType.Atrous, merger_groups='dw', encoder_groups='dw', decoder_groups='dw')

    print("Normal")
    model = UMergeNet(in_channels=3, out_channels=1, conv_type=ConvType.Standard)

    print("Normal-DW")
    model = UMergeNet(in_channels=3, out_channels=1, conv_type=ConvType.Standard, merger_groups='dw', encoder_groups='dw', decoder_groups='dw')


    #REFERENCE -UMergeNetMergerMoreLayersPerLevel-2
    #2,446,673
    #GFLOPS: 1.35
    # Dice: 0.9412 mIoU: 0.8891
    # FPS: 907 Time per image: 1,101 ms

