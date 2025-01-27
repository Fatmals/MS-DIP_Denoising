from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet
from .unet import MultiScaleUNet  # Ensure this is correctly imported

import torch.nn as nn



def get_net(input_depth, NET_TYPE, pad, upsample_mode='bilinear', norm_layer=None, num_scales=5, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4):
    # Set default normalization layer if not provided
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    
    if NET_TYPE == 'ResNet':
        net = ResNet(input_depth, 3, 10, 16, 1, norm_layer, False)
    elif NET_TYPE == 'skip':
        net = skip(input_depth, n_channels, num_channels_down=[skip_n33d]*num_scales, 
                   num_channels_up=[skip_n33u]*num_scales,
                   num_channels_skip=[skip_n11]*num_scales, 
                   upsample_mode=upsample_mode, need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)
    elif NET_TYPE == 'MultiScaleUNet':
        return MultiScaleUNet(num_input_channels=input_depth, num_output_channels=3,
                              scales=[1, 0.5, 0.25], feature_scale=4, more_layers=0,
                              concat_x=False, upsample_mode=upsample_mode,
                              pad=pad, norm_layer=norm_layer if norm_layer is not None else nn.BatchNorm2d,
                              need_sigmoid=True, need_bias=True)
    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, ratios=[32, 16, 8, 4, 2, 1], fill_noise=False, pad=pad)
    elif NET_TYPE == 'identity':
        net = nn.Sequential()
    else:
        raise ValueError(f"Unsupported NET_TYPE: {NET_TYPE}. Please choose from 'ResNet', 'skip', 'MultiScaleUNet', 'texture_nets', or 'identity'.")

    return net

