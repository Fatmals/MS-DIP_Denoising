import torch
import torch.nn as nn
from .common import *

def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], 
        num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)  # Number of scales based on the depth of the network

    # Adding multi-scale output layers
    output_layers = []

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(n_scales):
        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            skip.add_module(str(i), Concat(1, skip, deeper))
        else:
            model_tmp.add_module(str(i), deeper)
        
        deeper.add_module(str(i), conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add_module('bn' + str(i), bn(num_channels_down[i]))
        deeper.add_module('act' + str(i), act(act_fun))

        if i == n_scales - 1:
            # The deepest layer: no downsampling
            deeper.add_module('upsample' + str(i), nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        else:
            # Intermediate layers
            model_tmp = deeper

        model_tmp.add_module('conv_up' + str(i), conv(num_channels_skip[i] + num_channels_down[i], num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add_module('bn_up' + str(i), bn(num_channels_up[i]))
        model_tmp.add_module('act_up' + str(i), act(act_fun))

        # Multi-scale outputs for each scale
        output_layers.append(nn.Conv2d(num_channels_up[i], num_output_channels, 1, bias=need_bias, pad=pad))

    # Adding all multi-scale outputs to the model
    model.add_module('output_layers', nn.ModuleList(output_layers))

    if need_sigmoid:
        model.add_module('sigmoid', nn.Sigmoid())

    return model

