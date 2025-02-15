from .skip import skip
from .skip import multi_scale_skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet, MultiScaleUNet  # ✅ Import MultiScaleUNet from unet.py

import torch.nn as nn


def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='ReLU', 
            skip_n33d=128, skip_n33u=128, skip_n11=4, scales=[1.0], num_scales=5, downsample_mode='stride'):
    """
    Constructs a multi-scale version of the network.
    Initializes multiple networks (one per scale) and fuses results.
    """
    if NET_TYPE == 'skip':
        models = multi_scale_skip(input_depth, n_channels, scales, pad)
        net = MultiScaleUNet(models)  # ✅ Now using MultiScaleUNet from unet.py

    elif NET_TYPE == 'ResNet':
        net = ResNet(input_depth, 3, 10, 16, 1, nn.BatchNorm2d, False)

    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, num_output_channels=n_channels, pad=pad, act_fun=act_fun)

    else:
        raise ValueError(f"Unsupported NET_TYPE: {NET_TYPE}")

    return net  # ✅ Now `net` is a single model instead of a list




