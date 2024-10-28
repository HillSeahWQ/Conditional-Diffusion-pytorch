### Conditional Training
- (Optional) Configure Hyperparameters in `ddpm_conditional.py`
- Set path to dataset in `ddpm_conditional.py`
- Run the training script:
```bash
python ddpm_conditional.py
```

### Conditional Model

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