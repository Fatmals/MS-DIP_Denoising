import os
from .common_utils import *


        
from torchvision.transforms.functional import resize

def get_multi_scale_noisy_images(img_np, scales, sigma):
    img_noisy_pils = []
    img_noisy_nps = []
    for scale in scales:
        # Downsample the image
        h, w = img_np.shape[1], img_np.shape[2]
        scaled_img = resize(np_to_pil(img_np), size=(int(h * scale), int(w * scale)))
        scaled_img_np = pil_to_np(scaled_img)
        
        # Add Gaussian noise
        noisy_img_np = np.clip(scaled_img_np + np.random.normal(scale=sigma, size=scaled_img_np.shape), 0, 1).astype(np.float32)
        noisy_img_pil = np_to_pil(noisy_img_np)
        
        img_noisy_pils.append(noisy_img_pil)
        img_noisy_nps.append(noisy_img_np)
    
    return img_noisy_pils, img_noisy_nps

