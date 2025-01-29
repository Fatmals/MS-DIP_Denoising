import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import * 

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class UNet(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                 feature_scale=4, more_layers=0, concat_x=False,
                 upsample_mode='deconv', pad='zero', norm_layer=nn.InstanceNorm2d,
                 need_sigmoid=True, need_bias=True):
        super(UNet, self).__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        # Using Dilated Convolutions in deeper layers for a larger receptive field
        self.start = unetConv2(num_input_channels, filters[0], norm_layer, need_bias, pad)
        self.down1 = unetDown(filters[0], filters[1], norm_layer, need_bias, pad)
        self.down2 = unetDown(filters[1], filters[2], norm_layer, need_bias, pad, dilation=2)
        self.down3 = unetDown(filters[2], filters[3], norm_layer, need_bias, pad, dilation=4)
        self.down4 = unetDown(filters[3], filters[4], norm_layer, need_bias, pad, dilation=8)

        # Multi-Scale Feature Fusion
        self.fuse_conv = nn.Conv2d(filters[1] + filters[2] + filters[3] + filters[4], filters[3], kernel_size=1, bias=False)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad)

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)
        if need_sigmoid:
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, inputs):
        in64 = self.start(inputs)
        down1 = self.down1(in64)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        # Multi-Scale Feature Fusion
        fused_features = torch.cat([F.interpolate(down1, size=down4.shape[2:], mode='bilinear', align_corners=True),
                                    F.interpolate(down2, size=down4.shape[2:], mode='bilinear', align_corners=True),
                                    F.interpolate(down3, size=down4.shape[2:], mode='bilinear', align_corners=True),
                                    down4], dim=1)
        down4 = self.fuse_conv(fused_features)

        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()

        print(pad)
        if norm_layer is not None:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs= self.conv1(inputs)
        outputs= self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown, self).__init__()
        self.conv= unetConv2(in_size, out_size, norm_layer, need_bias, pad)
        self.down= nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs= self.down(inputs)
        outputs= self.conv(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                   conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up= self.up(inputs1)
        
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2 
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2 
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output= self.conv(torch.cat([in1_up, inputs2_], 1))

        return output
