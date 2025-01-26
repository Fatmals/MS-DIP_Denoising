import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, channels, out_ch=3):
        super(UNet, self).__init__()
        self.down1 = self.contract_block(channels, 64)
        self.down2 = self.contract_block(64, 128)
        self.down3 = self.contract_block(128, 256)
        self.down4 = self.contract_block(256, 512)
        self.middle = self.contract_block(512, 1024)
        self.up4 = self.expand_block(1024, 512)
        self.up3 = self.expand_block(512, 256)
        self.up2 = self.expand_block(256, 128)
        self.up1 = self.expand_block(128, 64)
        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(F.max_pool2d(x1, 2))
        x3 = self.down3(F.max_pool2d(x2, 2))
        x4 = self.down4(F.max_pool2d(x3, 2))
        xm = self.middle(F.max_pool2d(x4, 2))
        x = self.up4(F.interpolate(xm, scale_factor=2, mode='nearest'))
        x = self.up3(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.up2(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.up1(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.final(x)
        return x

    def contract_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def expand_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

class UNetMultiScale(nn.Module):
    def __init__(self, num_scales, channels=3, out_ch=3):
        super(UNetMultiScale, self).__init__()
        self.num_scales = num_scales
        self.scales = nn.ModuleList([UNet(channels, out_ch) for _ in range(num_scales)])

    def forward(self, x):
        outputs = []
        for scale in self.scales:
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            output = scale(x)
            outputs.append(output)
            x = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
        return outputs

