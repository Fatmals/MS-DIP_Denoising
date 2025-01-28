import os
import cv2
from .common_utils import *


from torchvision.transforms.functional import resize

def get_noisy_image(img_np, sigma, scales=None):
    """
    Adds Gaussian noise to an image, optionally handling multiple scales.
    
    Args: 
        img_np: np.array with values from 0 to 1 (H, W, C)
        sigma: std of the noise
        scales: list of scales to resize the image to and apply noise
    
    Returns:
        A dictionary with noisy images and their scales:
            {
                'scales': [scale1, scale2, ...],
                'noisy_images': [noisy_img_np1, noisy_img_np2, ...],
                'noisy_pil': [noisy_img_pil1, noisy_img_pil2, ...]
            }
    """
    if scales is None:
        # Default single scale behavior
        img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        return img_noisy_pil, img_noisy_np

    # Multi-scale handling
    results = {
        'scales': [],
        'noisy_images': [],
        'noisy_pil': []
    }
    
    for scale in scales:
        # Resize the image to the current scale
        img_scaled = cv2.resize(img_np, (int(img_np.shape[1] * scale), int(img_np.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
        
        # Add Gaussian noise
        img_noisy_np = np.clip(img_scaled + np.random.normal(scale=sigma, size=img_scaled.shape), 0, 1).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        
        # Append results
        results['scales'].append(scale)
        results['noisy_images'].append(img_noisy_np)
        results['noisy_pil'].append(img_noisy_pil)
    
    return results
