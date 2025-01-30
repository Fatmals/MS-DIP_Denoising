import cv2
import numpy as np
import torch
import torch.nn.functional as F

def create_pyramid(image, num_scales=3):
    """Creates a Gaussian pyramid of the input image."""
    pyramid = [image]
    for _ in range(num_scales - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def upsample_and_merge(denoised_pyramid):
    """Upsamples and merges multi-scale outputs."""
    merged = denoised_pyramid[-1]  # Start with the coarsest scale
    for i in range(len(denoised_pyramid) - 2, -1, -1):
        merged = cv2.pyrUp(merged)
        merged = cv2.addWeighted(merged, 0.5, denoised_pyramid[i], 0.5, 0)
    return merged

def multi_scale_denoising(image, model, device, num_scales=3):
    """Applies multi-scale denoising."""
    image_pyramid = create_pyramid(image, num_scales)
    denoised_pyramid = []

    for scale_image in image_pyramid:
        scale_image_tensor = torch.from_numpy(scale_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        denoised = model(scale_image_tensor)
        denoised_pyramid.append(denoised.squeeze(0).permute(1, 2, 0).cpu().numpy())

    return upsample_and_merge(denoised_pyramid)

