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
    
    n_scales = len(num_channels_down)  # Number of scales based on the depth of the network

    # Ensure all parameters that are expected to be lists are handled correctly
    upsample_mode = [upsample_mode]*n_scales if not isinstance(upsample_mode, list) else upsample_mode
    downsample_mode = [downsample_mode]*n_scales if not isinstance(downsample_mode, list) else downsample_mode
    filter_size_down = [filter_size_down]*n_scales if not isinstance(filter_size_down, list) else filter_size_down
    filter_size_up = [filter_size_up]*n_scales if not isinstance(filter_size_up, list) else filter_size_up

    model = nn.Sequential()

    input_depth = num_input_channels
    for i in range(n_scales):
        deeper = nn.Sequential()
        skip = nn.Sequential()

        # Add skip connections if applicable
        if num_channels_skip[i] != 0:
            skip_conn = Concat(1, skip, deeper)
            model.add_module('skip' + str(i), skip_conn)

        # Creating down-sampling layers
        deeper.add_module('conv_down' + str(i), conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add_module('bn_down' + str(i), bn(num_channels_down[i]))
        deeper.add_module('act_down' + str(i), act(act_fun))

        # Intermediate layers within deeper
        if i < n_scales - 1:
            deeper.add_module('deeper' + str(i), nn.Sequential())

        # Creating up-sampling layers
        model.add_module('upsample' + str(i), nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        model.add_module('conv_up' + str(i), conv(num_channels_down[i], num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model.add_module('bn_up' + str(i), bn(num_channels_up[i]))
        model.add_module('act_up' + str(i), act(act_fun))

        # Prepare for the next iteration
        input_depth = num_channels_down[i]

    # Final convolution to output channels
    model.add_module('final_conv', conv(num_channels_up[-1], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add_module('sigmoid', nn.Sigmoid())

    return model


