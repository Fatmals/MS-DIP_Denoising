## 📚 Built On Top Of

This project is based on the original *Deep Image Prior* by:

**Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky**  
*CVPR 2018 – Deep Image Prior*  
[arXiv:1711.10925](https://arxiv.org/abs/1711.10925)

It builds upon Deep Image Prior by incorporating a **multi-scale strategy** and introducing **architectural and optimisation modifications** tailored specifically for denoising tasks.

> 🔗 [View original DIP repository (Ulyanov et al.)](https://github.com/DmitryUlyanov/deep-image-prior)

---

## 📌 Overview

This repository contains the official code for my MAI dissertation project, which proposes a **Multi-Scale Deep Image Prior (MS-DIP)** framework for unsupervised image denoising. The core idea extends the original **Deep Image Prior (DIP)** method by incorporating:

- Multi-scale U-Net architectures
- Scale-adaptive Gaussian noise injection
- Output fusion across resolutions
- Custom loss functions including SSIM

MS-DIP aims to improve reconstruction of fine details **and** global structure, particularly under varying noise conditions.

---

## 📂 Features

- 📈 Unsupervised training — no clean images required
- 🔁 Fusion of multiple DIP models at different scales
- 🧠 SSIM + MSE loss combination for perceptual and pixel accuracy
- 🔬 Evaluation metrics: PSNR, SSIM with early stopping
- ⚙️ Supports Adamax, SGD, RMSProp, Adam optimisers

---

## 🚀 Getting Started

### 🔧 Installation (Conda)

Install dependencies using the included `environment.yml`:

```bash
conda env create -f environment.yml
conda activate msdip
