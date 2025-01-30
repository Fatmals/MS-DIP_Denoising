import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from unet import UNet  # Import original UNet
from common import get_image, crop_image, pil_to_np, get_noisy_image

# ---------------------------------------------
# 1. Multi-Scale Image Processing (Advanced Version)
# ---------------------------------------------
def create_scaled_images(fname, imsize, scales, sigma_):
    """Creates dynamically resized images and applies noise at different scales."""
    images = []
    noisy_images = []
    
    for scale in scales:
        # Resize image to the current scale
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_pil = img_pil.resize((int(img_pil.width * scale), int(img_pil.height * scale)), Image.LANCZOS)
        img_np = pil_to_np(img_pil)

        # Adjust noise level dynamically
        scaled_sigma = sigma_ * (1 / scale)
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, scaled_sigma)
        
        # Store scaled images
        images.append(img_np)
        noisy_images.append(img_noisy_np)

    return images, noisy_images

# ---------------------------------------------
# 2. Multi-Scale UNet Model
# ---------------------------------------------
class UNetMultiScale(nn.Module):
    """Modified UNet to handle multi-scale inputs."""
    def __init__(self, base_unet):
        super(UNetMultiScale, self).__init__()
        self.base_unet = base_unet
        self.fusion_layer = nn.Conv2d(6, 3, kernel_size=1)  # Learnable fusion

    def forward(self, pyramid_inputs):
        outputs = [self.base_unet(inp) for inp in pyramid_inputs]

        # Concatenate and use a learnable fusion layer
        for i in range(len(outputs) - 1, 0, -1):
            upsampled = F.interpolate(outputs[i], size=outputs[i - 1].shape[-2:], mode='bilinear')
            outputs[i - 1] = self.fusion_layer(torch.cat([outputs[i - 1], upsampled], dim=1))  # Learnable fusion

        return outputs[0]  # Final output

# ---------------------------------------------
# 3. Multi-Scale Denoising Function
# ---------------------------------------------
def multi_scale_denoising(fname, imsize, model, device, scales, sigma_):
    """Applies multi-scale denoising using dynamically resized images."""
    images, noisy_images = create_scaled_images(fname, imsize, scales, sigma_)
    pyramid_tensors = [torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) for img in noisy_images]
    
    denoised_pyramid = model(pyramid_tensors)  # Pass multi-scale inputs to model
    return denoised_pyramid.squeeze(0).permute(1, 2, 0).cpu().numpy()

# ---------------------------------------------
# 4. Inference Example (Testing the Multi-Scale Denoising)
# ---------------------------------------------
if __name__ == "__main__":
    # Define scales and noise level
    scales = [2, 1.5, 1]
    sigma_ = 25  # Example noise level
    fname = 'sample_noisy_image.png'
    imsize = (256, 256)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_unet = UNet(num_input_channels=3, num_output_channels=3).to(device)
    model = UNetMultiScale(base_unet).to(device)
    model.load_state_dict(torch.load("denoising_model.pth"))
    model.eval()

    # Apply multi-scale denoising
    denoised_image = multi_scale_denoising(fname, imsize, model, device, scales, sigma_)

    # Show results
    plt.figure(figsize=(10, 5))
    plt.imshow(np.clip(denoised_image, 0, 1))  # Clip values for display
    plt.title("Denoised Image (Multi-Scale)")
    plt.axis('off')
    plt.show()
