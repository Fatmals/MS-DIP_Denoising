import os
from .common_utils import *


        
def get_noisy_image(img_np, sigma_list):
    """
    Adds synthetic noise at multiple scales.
    img_np: Original image (numpy array).
    sigma_list: List of noise levels corresponding to different scales.
    Returns a list of noisy images at different scales.
    """
    noisy_images = []
    for sigma in sigma_list:
        noise = np.random.normal(scale=sigma, size=img_np.shape)
        img_noisy = np.clip(img_np + noise, 0, 1)
        noisy_images.append(img_noisy)
    return noisy_images
