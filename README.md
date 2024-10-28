# Conditional Denoising Diffusion Probabilistic Model (DDPM)

## Project Overview
This project implements a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** with enhancements through **Classifier-Free Guidance (CFG)** and **Exponential Moving Average (EMA)**. Diffusion models have emerged as powerful generative models, particularly for image generation tasks, by modeling the data distribution through iterative denoising. Here, we explore conditioning the DDPM to generate class-specific images and apply optimization techniques to improve sample quality and training stability.

### Papers:

[1. Conditional DDPM](https://arxiv.org/pdf/2105.05233)

[2. CFG](https://arxiv.org/pdf/2207.12598)

[3. EMA](https://arxiv.org/pdf/2102.09672)

## Key Features and Improvements
1. **Conditional Training**: The model is conditioned on class labels, enabling the generation of specific classes in datasets such as [CIFAR-10](https://www.kaggle.com/datasets/cifar-10).
2. **Classifier-Free Guidance (CFG)**: By integrating CFG to steer the generation process, we decrease posterior collaspe and increase the likelihood of samples being aligned with the desired class, and enhancing visual fidelity.
3. **Exponential Moving Average (EMA)**: EMA smoothing is applied to the model weights, stabilizing training and improving generated image quality by using averaged weights over recent steps.

---

### Setup Prerequisites

1. **Install Python 3.12.3**
2. **Install Poetry**
3. **Install Nvidia CUDA 12.1**
   - Note: The version of PyTorch in this project uses CUDA 12.1 for GPU computing.

---

### Steps to Run
1. **Clone this repository**
2. **Install Dependencies:**

   ```bash
   poetry install
   ```
3. **Enter the virtual environment:**

   ```bash
   poetry shell
   ```
4. **Conditional Training**
   - (Optional) Configure Hyperparameters in `train.py`
   - Set path to dataset in `train.py`
   - Run the training script:
   ```bash
   python train.py
   ```
5. **Sampling of Conditional Model**

   This model was trained on [CIFAR-10 64x64](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution) with 10 classes:

   - Classes: airplane:0, auto:1, bird:2, cat:3, deer:4, dog:5, frog:6, horse:7, ship:8, truck:9

   Example code to load and sample from the model:

   ```py
   n = 10
   device = "cuda"
   model = UNet_conditional(num_classes=10).to(device)
   ckpt = torch.load("conditional_ema_ckpt.pt")
   model.load_state_dict(ckpt)
   diffusion = Diffusion(img_size=64, device=device)
   y = torch.Tensor([6] * n).long().to(device)
   x = diffusion.sample(model, n, y, cfg_scale=3)
   utils.plot_images(x)
   ```
---

## Results
The conditional DDPM with CFG and EMA demonstrates relatively high-quality, class-specific image generation with improved stability and visual fidelity. The project’s potential applications range from image synthesis and super-resolution to data augmentation and beyond.