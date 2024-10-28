from pathlib import Path
import os
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils import *
from modules import UNet, EMA

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # linear scheduler for Beta_t
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # get noisy images xt, and the actual noise added -> Xt = sqrt(cumprod_alphat) * x0 + sqrt(1-cumprod_alphat) * noise, noise~N(0,1)
    def get_noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    # sample random time steps t, 0 <= t <= T (uniform distribution)
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    # sample algorithm (see algorithm 2 - sampling for diffusion paper) also the reverse diffusion process (denoising)
    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():

            # get initial max noise images, from the standard normal distribution
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

            # for T time steps, starting from first time step of the reverse process T, all the way to 1 (final image)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                # get the corresponding time steps (same size as batch size) for feeding into the model for noise prediction (recap, model needs to know the time step information for each pass)
                t = (torch.ones(n) * i).long().to(self.device)

                # predicted noise from the model
                predicted_noise = model(x, t, labels)

                # for cfg, a modified predicted noise formula, see CFG paper (cfg_scale > 0 when we want to use cfg)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                # follow close formula for getting Xt-1 from Xt (denoising noisy samples)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # for z in the algorithm
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # formula for sampled image in algorithm
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()

        # transformation from denoised x, x0 tensor, to valid images
        x = (x.clamp(-1, 1) + 1) / 2 # clamp outputs to be [-1, 1], then +1 and /2 to transform to [0, 1]
        x = (x * 255).type(torch.uint8) # transform back to valid pixel range [0, 255]
        return x
    

def train(args):
    # Setup
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # Algorithm 1 - training algorithm
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # CFG
            if np.randm.randm() <= 0.1:
                labels=None

            # algo till get loss
            t = diffusion.sample_timesteps(images.shape[0]).to(device) # sample timesteps uniformly
            x_t, noise = diffusion.get_noise_images(images, t) # sample random noise images at random uniform time step t, get the noisy image + actual noise
            predicted_noise = model(x_t, t, labels) # predict the noise for each noisy image at their own timestep t
            loss = mse(noise, predicted_noise) # loss function is a simple mse of actual and predicted noise

            # normal back prop + optimization steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model) # for EMA

            # Logs
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

            # Eval
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

            if epoch % 10 == 0:
                labels = torch.arange(10).long().to(device)
                sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
                ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
                plot_images(sampled_images)
                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
                save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
                torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
                torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # Hyperparameters
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 14 # change to lower if facing gpu memory issues
    args.image_size = 64 
    args.num_classes = 10 # update to your dataset's number of classes
    args.dataset_path = Path().cwd().parent / "data" / "cifar10" / "cifar10-64" / "train"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()