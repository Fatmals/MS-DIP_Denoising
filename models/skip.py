import torch
import torch.nn as nn
from .common import *

def enhanced_multiscale_skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], 
        num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    # Ensuring list input for modes and filter sizes
    upsample_mode = [upsample_mode] * n_scales if not isinstance(upsample_mode, list) else upsample_mode
    downsample_mode = [downsample_mode] * n_scales if not isinstance(downsample_mode, list) else downsample_mode
    filter_size_down = [filter_size_down] * n_scales if not isinstance(filter_size_down, list) else filter_size_down
    filter_size_up = [filter_size_up] * n_scales if not isinstance(filter_size_up, list) else filter_size_up

    # Multi-scale output layers
    output_layers = []

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(n_scales):
        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == n_scales - 1:
            # The deepest layer
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        # Adding multi-scale outputs
        output_layers.append(conv(num_channels_up[i], num_output_channels, 1, bias=need_bias, pad=pad))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(nn.ModuleList(output_layers))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
