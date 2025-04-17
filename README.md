from pathlib import Path

readme_content = """# Multi-Scale Deep Image Prior (MS-DIP) for Image Denoising

> **Author:** Fatma Al Saadi  
> **Project:** MSc Dissertation, Trinity College Dublin (2025)  
> **Supervisor:** [Your Supervisor's Name]  
> **Built on top of:** [Deep Image Prior (Ulyanov et al., CVPR 2018)](https://github.com/DmitryUlyanov/deep-image-prior)

---

## 📌 Overview

This repository contains the official code for my MSc dissertation project, which proposes a **Multi-Scale Deep Image Prior (MS-DIP)** framework for unsupervised image denoising. The core idea extends the original **Deep Image Prior (DIP)** method by incorporating:

- Multi-scale U-Net architectures
- Scale-adaptive Gaussian noise injection
- Output fusion across resolutions
- Custom loss functions including SSIM and Total Variation Loss

MS-DIP aims to improve reconstruction of fine details **and** global structure, particularly under varying noise conditions.

---

## 📂 Features

- 📈 Unsupervised training — no clean images required
- 🔁 Fusion of multiple DIP models at different scales
- 🧠 SSIM + MSE loss combination for perceptual and pixel accuracy
- ✨ Total Variation Loss for texture preservation
- 🔬 Evaluation metrics: PSNR, SSIM with early stopping
- ⚙️ Supports Adamax, SGD, RMSProp, Adam optimizers

---

## 🚀 Getting Started

### 🔧 Installation (Conda)

Install dependencies using the included `environment.yml`:

```bash
conda env create -f environment.yml
conda activate msdip
