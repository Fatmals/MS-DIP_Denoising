import cv2
import numpy as np
import torch
from common_utils import get_image, crop_image, pil_to_np, get_noisy_image

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

def multi_scale_denoise(fname, imsize, model, device, scales, sigma_):
    """Applies multi-scale denoising using dynamically resized images."""
    images, noisy_images = create_scaled_images(fname, imsize, scales, sigma_)
    pyramid_tensors = [torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) for img in noisy_images]

    denoised_pyramid = model(pyramid_tensors)  # Pass multi-scale inputs to model
    return denoised_pyramid.squeeze(0).permute(1, 2, 0).cpu().numpy()


