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
#######################
# NEW 
#######################
class DIPGenerator(nn.Module):
    def __init__(self, num_channels=64):
        super(DIPGenerator, self).__init__()
        # Encoder
        self.down1 = nn.Conv2d(3, num_channels, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(num_channels, num_channels * 2, 3, stride=2, padding=1)
        # Decoder
        self.up1 = nn.ConvTranspose2d(num_channels * 2, num_channels, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(num_channels, 3, 3, stride=2, padding=1, output_padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        # Encoding
        x = self.act(self.down1(x))
        x = self.act(self.down2(x))
        # Decoding
        x = self.act(self.up1(x))
        x = self.up2(x)
        return x

########################
# Multi-Scale
########################
class UNet(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
   def __init__(self, scales=[1, 0.5, 0.25]):
        super(UNet, self).__init__()
        self.scales = scales
        self.generators = nn.ModuleList([DIPGenerator() for _ in self.scales])

    def forward(self, x):
        outputs = []
        current_input = x
        for gen in self.generators:
            # Downsample image to current scale
            scaled_input = F.interpolate(current_input, scale_factor=self.scales[len(outputs)], mode='bilinear', align_corners=False)
            output = gen(scaled_input)
            # Upsample output to original size
            output = F.interpolate(output, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            outputs.append(output)
            current_input = output  # Use current output as input for the next scale

        # Multi-scale inference ensemble
        final_output = torch.mean(torch.stack(outputs), dim=0)
        return final_output

    def train_step(self, input_image, noisy_image):
        # Assume an optimizer and loss function are defined outside
        self.train()
        output = self(input_image)
        loss = torch.nn.MSELoss()(output, noisy_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()






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

