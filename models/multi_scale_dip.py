import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleUNet(nn.Module):
    def __init__(self, unet_models):
        """
        A wrapper model that applies multiple U-Net models at different scales
        and fuses the results.
        """
        super(MultiScaleUNet, self).__init__()
        self.unet_models = nn.ModuleList(unet_models)  # Store multiple U-Nets

    def forward(self, noisy_images):
        """
        Forward pass for multi-scale denoising.
        noisy_images: list of tensors at different scales.
        Returns: Fused denoised output.
        """
        denoised_outputs = [model(img) for model, img in zip(self.unet_models, noisy_images)]

        # Resize all outputs to the size of the largest tensor
        max_size = denoised_outputs[0].shape[-2:]  # Get shape of first output (H, W)
        resized_outputs = [F.interpolate(out, size=max_size, mode="bilinear", align_corners=False) 
                           for out in denoised_outputs]

        final_output = torch.stack(resized_outputs, dim=0).mean(dim=0)  # Average fusion
        return final_output
