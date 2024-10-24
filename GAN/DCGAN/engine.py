from tqdm import tqdm
from pathlib import Path
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


def train_dcgan(
        discriminator,
        generator,
        dataloader,
        criterion,
        optimizer_D,
        optimizer_G,
        fixed_noise,
        z_dim=100,  # as per paper
        batch_size=128,
        epochs=10,  # as per paper
        device="cuda" if torch.cuda.is_available() else "cpu",
        summary_writer_real_logs_path=Path(__file__).parent / "runs" / "real",
        summary_writer_fake_logs_path=Path(__file__).parent / "runs" / "fake",
        model_save_path=None,
        start_epoch=0,  # New parameter to continue training from a specific epoch
        generator_checkpoint=None,  # Paths to load saved checkpoints
        discriminator_checkpoint=None
):  
    # Load the model checkpoints if provided
    if generator_checkpoint and discriminator_checkpoint:
        generator.load_state_dict(torch.load(generator_checkpoint), weights_only=True)
        discriminator.load_state_dict(torch.load(discriminator_checkpoint), weights_only=True)
        print(f"Resuming training from epoch {start_epoch + 1}...")

    # For tensorboard tracking
    step = 0
    writer_real = SummaryWriter(summary_writer_real_logs_path)
    writer_fake = SummaryWriter(summary_writer_fake_logs_path)

    for epoch in tqdm(range(start_epoch, epochs)):

        for batch_idx, (x, _) in enumerate(dataloader):
            
            # Get x, z, G(z), D(x), D(G(z))
            x = x.to(device)
            noise = torch.randn(size=(x.shape[0], z_dim, 1, 1)).to(device)

            g_z = generator(noise)  # G(z)
            d_x = discriminator(x).reshape(-1)  # D(x), reshape from 1*1*1 to 1
            d_g_z = discriminator(g_z).reshape(-1)  # D(G(z)), reshape from 1*1*1 to 1

            ### Train the Discriminator
            loss_real_D = criterion(d_x, torch.ones_like(d_x))  # -log(D(X))
            loss_fake_D = criterion(d_g_z, torch.zeros_like(d_g_z))  # -log(1-D(G(z)))
            loss_D = (loss_fake_D + loss_real_D) / 2

            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            ### Train the Generator
            d_g_z_next = discriminator(g_z).reshape(-1)  # after training the disc, new D(G(z))
            loss_G = criterion(d_g_z_next, torch.ones_like(d_g_z_next))  # -log(D(G(z)))

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Print losses occasionally and log to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch {batch_idx}/{len(dataloader)} :- Discriminator Loss: {loss_D:.4f} | Generator Loss: {loss_G:.4f}")

                with torch.no_grad():
                    fake = generator(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(x[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1

        if model_save_path:
            torch.save(generator.state_dict(), model_save_path / f"generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), model_save_path / f"discriminator_epoch_{epoch + 1}.pth")


# def train_dcgan(
#         discriminator,
#         generator,
#         dataloader,
#         criterion,
#         optimizer_D,
#         optimizer_G,
#         fixed_noise,
#         z_dim = 100,  # as per paper
#         batch_size = 128,
#         epochs = 10, # as per paper
#         device = "cuda" if torch.cuda.is_available() else "cpu",
#         summary_writer_real_logs_path = Path(__file__).parent / "runs" / "real",
#         summary_writer_fake_logs_path = Path(__file__).parent / "runs" / "fake",
#         model_save_path = None

# ):  
#     # For tensorboard tracking
#     step = 0
#     writer_real = SummaryWriter(summary_writer_real_logs_path)
#     writer_fake = SummaryWriter(summary_writer_fake_logs_path)

#     for epoch in tqdm(range(epochs)):

#         for batch_idx, (x, _) in enumerate(dataloader):
            
#             # Get x, z, G(z), D(x), D(G(z))
#             x = x.to(device)
#             noise = torch.randn(size = (batch_size, z_dim, 1, 1)).to(device)

#             g_z = generator(noise) # G(z)
#             d_x = discriminator(x).reshape(-1) # D(x), reshape from 1*1*1 to 1
#             d_g_z = discriminator(g_z).reshape(-1) # D(G(z)), reshape from 1*1*1 to 1

#             ### Train the Discriminator: Min -(log(D(x)) + log(1-D(G(Z)))) <---> Max log(D(x)) + log(1-D(G(Z)))

#             loss_real_D = criterion(d_x, torch.ones_like(d_x)) # -log(D(X))
#             loss_fake_D = criterion(d_g_z, torch.zeros_like(d_g_z)) # -log(1-D(G(z)))
#             loss_D = (loss_fake_D + loss_real_D)/2 #-(log(D(x)) + log(1-D(G(Z)))) note: dont have to divide by 2 but taking the average of 2 lossess as each loss [0, 1], so that it makes sense when printed

#             optimizer_D.zero_grad()

#             loss_D.backward(retain_graph=True)

#             optimizer_D.step()

#             ### Train the Generator: Min -log(D(G(z)) <---> Max log(D(G(z))) <---> Min log(1-D(G(z)))
#             d_g_z_next = discriminator(g_z).reshape(-1) # after training the disc, new D(G(z)), reshape from 1*1*1 to 1
#             loss_G = criterion(d_g_z_next, torch.ones_like(d_g_z_next)) # -log(D(G(z)))

#             optimizer_G.zero_grad()

#             loss_G.backward()

#             optimizer_G.step()

#             # Print losses occasionally and print to tensorboard
#             # For tensorboard tracking
#             if batch_idx % 100 == 0:
#                 print(
#                     f"Epoch [{epoch}/{epochs}], Batch {batch_idx}/{len(dataloader)} :- Discriminator Loss: {loss_D:.4f} | Generator Loss: {loss_G:.4f}")

#                 with torch.no_grad():
#                     fake = generator(fixed_noise)
#                     # take out (up to) 32 examples
#                     img_grid_real = torchvision.utils.make_grid(x[:32], normalize=True)
#                     img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

#                     writer_real.add_image("Real", img_grid_real, global_step=step)
#                     writer_fake.add_image("Fake", img_grid_fake, global_step=step)

#                 step += 1

#         if model_save_path:
#             torch.save(generator.state_dict(), model_save_path / f"generator_epoch_{epoch + 1}.pth")
#             torch.save(discriminator.state_dict(), model_save_path / f"discriminator_epoch_{epoch + 1}.pth")