from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet, MultiScaleUNet  # Import MultiScaleUNet

import torch.nn as nn

def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU',
            skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride',
            scales=None, norm_layer=nn.BatchNorm2d):  # Default norm_layer
    if scales is None:
        scales = [2.5, 2, 1.5]  # Default scales
    
    if net_type == 'MultiScaleUNet':
            if scales is None:
            scales = [2.5, 2, 1.5]  # Default scales
            
            # Ensure norm_layer is explicitly provided, otherwise use InstanceNorm2d as default
            if norm_layer is None:
                        norm_layer = nn.InstanceNorm2d
            
            net = MultiScaleUNet(
                        num_input_channels=input_depth,
                        num_output_channels=3,
                        feature_scale=4,
                        scales=scales,
                        upsample_mode=upsample_mode,
                        pad=pad,
                        norm_layer=norm_layer,  # Pass the correct normalization layer
                        need_sigmoid=True,
                        need_bias=True
        )

    elif NET_TYPE == 'UNet':
        net = UNet(
            num_input_channels=input_depth,
            num_output_channels=n_channels,
            feature_scale=4,
            more_layers=0,
            concat_x=False,
            upsample_mode=upsample_mode,
            pad=pad,
            norm_layer=norm_layer,  # Explicitly pass norm_layer here
            need_sigmoid=True,
            need_bias=True
        )
    # Other NET_TYPE cases remain unchanged
    return net



