import os
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data_setup import create_dataloader
from model_builder import Generator, Discriminator, initialize_weights
from engine import train_dcgan

# Configuration
class Config:
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_NAME = "celeb_A"
    image_folder_dir = Path().cwd() / "data" / DATASET_NAME
    model_save_path = Path().cwd() / "models" / f"DCGAN_{DATASET_NAME}"

    # to change if require checkpoint resume training
    start_epoch=5
    generator_checkpoint=Path().cwd() / "models" / f"DCGAN_{DATASET_NAME}" / f"generator_epoch_{start_epoch}.pth"
    discriminator_checkpoint=Path().cwd() / "models" / f"DCGAN_{DATASET_NAME}" / f"discriminator_epoch_{start_epoch}.pth"

    # Hyper parameters from the DCGAN paper
    BATCH_SIZE = 128
    IMG_SIZE = 64
    IMG_CHANNELS = 3
    Z_DIM = 100
    LEARNING_RATE = 2e-4
    B1 = 0.5
    

# Create transforms
def get_transforms():
    return transforms.Compose([
        transforms.Resize(size=(Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(Config.IMG_CHANNELS)], # ensure image pixels normalised to [-1, 1]
            [0.5 for _ in range(Config.IMG_CHANNELS)]
        )
    ])

def main():
    image_folder_dir, model_save_path = Config.image_folder_dir, Config.model_save_path

    if not image_folder_dir.exists():
        print(f"Directory {image_folder_dir} does not exist!")
        return
    
    # Import the data
    dataloader = create_dataloader(
        root=image_folder_dir, 
        transforms=get_transforms(), 
        batch_size=Config.BATCH_SIZE, 
        num_workers=os.cpu_count(),
        shuffle=True, 
        pin_memory=True
    )

    # Instantiate and initialize the models
    generator = Generator(noise_channels=Config.Z_DIM, img_channels=Config.IMG_CHANNELS).to(Config.DEVICE)
    discriminator = Discriminator(img_channels=Config.IMG_CHANNELS).to(Config.DEVICE)

    initialize_weights(generator)
    initialize_weights(discriminator)

    # Create loss function and optimizer
    criterion = nn.BCELoss()
    optimizer_D = optim.Adam(params=discriminator.parameters(), lr=Config.LEARNING_RATE, betas=(Config.B1, 0.999))
    optimizer_G = optim.Adam(params=generator.parameters(), lr=Config.LEARNING_RATE, betas=(Config.B1, 0.999))

    # For evaluation
    fixed_noise = torch.randn(Config.BATCH_SIZE, Config.Z_DIM, 1, 1).to(Config.DEVICE)

    # Train
    train_dcgan(
        discriminator=discriminator,
        generator=generator,
        dataloader=dataloader,
        criterion=criterion,
        optimizer_D=optimizer_D,
        optimizer_G=optimizer_G,
        fixed_noise=fixed_noise,
        z_dim=Config.Z_DIM,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        device=Config.DEVICE,
        summary_writer_real_logs_path=Path(__file__).parent / "runs" / f"real_{datetime.now().strftime("%Y-%m-%d")}",
        summary_writer_fake_logs_path=Path(__file__).parent / "runs" / f"fake_{datetime.now().strftime("%Y-%m-%d")}",
        model_save_path=model_save_path,
        start_epoch=Config.start_epoch,
        generator_checkpoint=Config.generator_checkpoint,
        discriminator_checkpoint=Config.discriminator_checkpoint
    )

if __name__ == "__main__":
    main()
