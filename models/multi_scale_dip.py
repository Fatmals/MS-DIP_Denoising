import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleNetwork(nn.Module):
    '''
    A Multi-Scale Deep Image Prior Model
    Processes input at multiple scales (full, half, quarter) and fuses their outputs.
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                 num_filters=[64, 128, 256], need_sigmoid=True, need_bias=True):
        super(MultiScaleNetwork, self).__init__()

        # Define separate encoders and decoders for different scales
        self.encoder_full = self._make_encoder(num_input_channels, num_filters, need_bias)
        self.encoder_half = self._make_encoder(num_input_channels, num_filters, need_bias)
        self.encoder_quarter = self._make_encoder(num_input_channels, num_filters, need_bias)

        self.decoder_full = self._make_decoder(num_filters, num_output_channels, need_bias)
        self.decoder_half = self._make_decoder(num_filters, num_output_channels, need_bias)
        self.decoder_quarter = self._make_decoder(num_filters, num_output_channels, need_bias)

        # Fusion layer to combine multi-scale outputs
        self.fusion = nn.Conv2d(num_output_channels * 3, num_output_channels, kernel_size=1, bias=need_bias)

        # Add final sigmoid layer if needed
        if need_sigmoid:
            self.final = nn.Sequential(self.fusion, nn.Sigmoid())
        else:
            self.final = self.fusion

    def _make_encoder(self, num_input_channels, num_filters, need_bias):
        '''
        Creates an encoder with a series of convolutional layers.
        '''
        layers = []
        in_channels = num_input_channels
        for out_channels in num_filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=need_bias))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))  # Downsampling
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_decoder(self, num_filters, num_output_channels, need_bias):
        '''
        Creates a decoder with upsampling and convolutional layers.
        '''
        layers = []
        in_channels = num_filters[-1]
        for out_channels in reversed(num_filters[:-1]):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=need_bias))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, num_output_channels, kernel_size=3, stride=1, padding=1, bias=need_bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward pass through the multi-scale network.
        '''
        # Downscale input for half and quarter resolutions
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

        # Pass through encoders
        enc_full = self.encoder_full(x)
        enc_half = self.encoder_half(x_half)
        enc_quarter = self.encoder_quarter(x_quarter)

        # Pass through decoders
        dec_full = self.decoder_full(enc_full)
        dec_half = F.interpolate(self.decoder_half(enc_half), scale_factor=2.0, mode='bilinear', align_corners=False)
        dec_quarter = F.interpolate(self.decoder_quarter(enc_quarter), scale_factor=4.0, mode='bilinear', align_corners=False)

        # Concatenate multi-scale outputs
        combined = torch.cat([dec_full, dec_half, dec_quarter], dim=1)

        # Fuse and produce final output
        output = self.final(combined)
        return output
