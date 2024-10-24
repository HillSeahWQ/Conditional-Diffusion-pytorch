import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, noise_channels=100, img_channels=3):
        super().__init__()
        self.conv_layers = nn.Sequential(

            # 1st fractional strided convolution layer (upsample from 1*1 -> 4*4)
            # Projection layer, to convert the z of 100 inputs to 1024 * 4 * 4 (noise_channels = z_dim)
            # Each input (z) will be actually reshaped to 100 * 1 * 1 (100 channels)
            # (to ensure from 1x1 -> 4x4, with stride = 2 and kernal = 4, we need padding = 0 now (for a x4 increase))
            self._block(in_channels=noise_channels, out_channels=1024, kernel_size=4, stride=2, padding=0),

            # 2nd fractional strided convolution layer (upsample from 4*4 -> 8*8)
            self._block(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),

            # 3rd fractional strided convolution layer (upsample from 8*8 -> 16*16)
            self._block(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            
            # 4th fractional strided convolution layer (upsample from 16*16 -> 32*32)
            self._block(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),

            # Output fractional strided convolution layer (upsample from 32*32 -> 64*64)
            nn.ConvTranspose2d(in_channels=128, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):

        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ) if batch_norm else nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.conv_layers(z)
    

class Discriminator(nn.Module):

    def __init__(self, img_channels=3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            
            # 1st fractional strided convolution layer (downsample from 64*64 -> 32*32)
            self._block(in_channels=img_channels, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=False),

            # 2nd fractional strided convolution layer (downsample from 32*32 -> 16*16)
            self._block(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            
            # 3rd fractional strided convolution layer (downsample from 16*16 -> 8*8)
            self._block(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),

            # Output fractional strided convolution layer (downsample from 8*8 -> 4*4)
            self._block(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            
            # Classifier
            # No fully connected layer for DCGAN, use another way (instead of nn.Flatten(), nn.Linear(in_features=1024*4*4, out_features=1))
            # Use another convolutional layer (to ensure from 4x4 to 1x1, with stride = 2 and kernal = 4, we need padding = 0 now (for a x4 reduction))
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid() # ensure prediction is within [0, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        ) if batch_norm else nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        return self.conv_layers(x)
    

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02) # As per the DCGAN paper, Weights are initialized from Normal Distribution with mean = 0; standard deviation = 0.02.



# Notes:

# Fractional Strided Convolutional Layers for the Generator
# - Configuration as per figure: stride = 2, kernal size = 4
# - Use formula to calcuate the padding required:
#       H[out] = (H[in] - 1) * stride - 2 * padding + kernal_size + output_padding
#   => output_padding - 2*padding = -2 for all cases 
#   => Let output_padding = 0, padding = 1
#   => This config will x2 to the image dim (img_height and img width) for each convultional layer

# Generator:
# 1. Input: 100-dimensional uniform distribution (Z)
# 2. Projection layer: to 1024*4*4 (to be reshaped before sending to f-s convolutional layers)
# 3. A series of four fractionally-strided convolutions  (stride = 2, kernal size = 4) 
#   3.1 f-s conv: Output Chanels = 512, img_dim from 4*4 to 8*8
#   3.2 f-s conv: Output Chanels = 256, img_dim from 8*8 to 16*16
#   3.3 f-s conv: Output Chanels = 128, img_dim from 4*4 to 32*32
#   3.4. [Output]: f-s conv: Output Chanels = 3, img_dim from 32*32 to 64*64
# *Batch Norm to be applied (except last layer of generator)
# *ReLU all layers (except Output layer)
# *TanH for output layer

# Strided Convolutional Layer for the Discriminator
# - Configuration as per figure: stride = 2, kernal size = 4
# - Use formula to calcuate the padding required:
#       H[out] = [ (H[in] - kernal_size + 2 * padding) / stride ] + 1

#   => padding = 1 for all cases 
#   => Let padding = 1
#   => This config will x2 to the image dim (img_height and img width) for each convultional layer

# Discriminator: (just the mirror opposite of the configuration, with strided convolutional layers instead of fractional strided convolutional layers)
# 1. Input: (3 by 64 by 64) images
# 2. A series of four strided convolutions  (stride = 2, kernal size = 4) 
#   2.1 f-s conv: Output Chanels = 128, img_dim from 64*64 to 32*32
#   2.2 f-s conv: Output Chanels = 256, img_dim from 32*32 to 16*16 
#   2.3 f-s conv: Output Chanels = 512, img_dim from 16*16 to 8*8
#   2.4. [Output]: f-s conv: Output Chanels = 1024, img_dim from 8*8 to 4*4
# *Batch Norm to be applied (except first layer for the discriminator)
# *LeakyReLU all layers, slope set to 0.2


# class Generator(nn.Module):

#     def __init__(self, noise_channels=100, img_channels=3):
#         super().__init__()
        
#         # Projection layer, to convert the z of 100 inputs to 1024 * 4 * 4 (noise_channels = z_dim)
#         # Each input (z) will be actually reshaped to 100 * 1 * 1 (100 channels)
#         # (to ensure from 1x1 to 4x4, with stride = 2 and kernal = 4, we need padding = 0 now (for a x4 increase))
#         self.projection = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=noise_channels, out_channels=1024, kernel_size=4, stride=2, padding=0),
#             nn.BatchNorm2d(1024),
#             nn.ReLU()
#         )

#         self.conv_layers = nn.Sequential(
            
#             # 1st fractional strided convolution layer (upsample from 4*4 -> 8*8)
#             nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),

#             # 2nd fractional strided convolution layer (upsample from 8*8 -> 16*16)
#             nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
            
#             # 3rd fractional strided convolution layer (upsample from 16*16 -> 32*32)
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             # Output fractional strided convolution layer (upsample from 32*32 -> 64*64)
#             nn.ConvTranspose2d(in_channels=128, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()
#         )
    
#     def forward(self, z):
#         z_projected = self.projection(z) # project each z from z_dim*1*1 into 1024*4*4
#         return self.conv_layers(z_projected)


# class Discriminator(nn.Module):

#     def __init__(self, img_channels=3):
#         super().__init__()

#         self.conv_layers = nn.Sequential(
            
#             # 1st fractional strided convolution layer (downsample from 64*64 -> 32*32)
#             nn.Conv2d(in_channels=img_channels, out_channels=128, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.2),

#             # 2nd fractional strided convolution layer (downsample from 32*32 -> 16*16)
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(negative_slope=0.2),
            
#             # 3rd fractional strided convolution layer (downsample from 16*16 -> 8*8)
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(negative_slope=0.2),

#             # Output fractional strided convolution layer (downsample from 8*8 -> 4*4)
#             nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(negative_slope=0.2)
#         )

#         # No fully connected layer for DCGAN, use another way (instead of nn.Flatten(), nn.Linear(in_features=1024*4*4, out_features=1))
#         # Use another convolutional layer (to ensure from 4x4 to 1x1, with stride = 2 and kernal = 4, we need padding = 0 now (for a x4 reduction))
#         self.classifier = nn.Sequential(
#             nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, padding=0),
#             nn.Sigmoid() # ensure prediction is within [0, 1]
#         )
    
#     def forward(self, x):
#         return self.classifier(self.conv_layers(x))