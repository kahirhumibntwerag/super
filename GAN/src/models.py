import torch.nn as nn
import torch
# Initialize the generator
import torch.nn as nn

class DiscriminatorSRGAN(nn.Module):
    def __init__(self, disc_channels=16, disc_num_layers=7):
        super(DiscriminatorSRGAN, self).__init__()
        
        layers = []
        in_channels = 1
        out_channels = disc_channels

        for i in range(disc_num_layers):
            # Add a convolutional layer with increasing channels and alternating stride
            stride = 2 if i % 2 != 0 else 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
            
            # Double the number of channels every two layers to mimic the original pattern
            if i % 2 != 0:
                out_channels *= 2

        self.features = nn.Sequential(*layers)


        # Calculate the flattened size with a dummy forward pass
        with torch.no_grad():
            # The dimensions 1, 3, 512, 512 correspond to batch_size, channels, height, and width
            dummy_input = torch.zeros(1, 1, 512, 512)
            dummy_features = self.features(dummy_input)
            self.flattened_size = dummy_features.view(1, -1).size(1)

        # Define the classifier part of the discriminator
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    import torch.nn as nn

import torch

import numpy as np


class ResidualBlock(nn.Module):
    """
    Define a Residual Block without Batch Normalization
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class RRDB(nn.Module):
    """
    Define the Residual in Residual Dense Block (RRDB)
    """
    def __init__(self, in_features, num_dense_layers=3):
        super(RRDB, self).__init__()
        self.residual_blocks = nn.Sequential(*[ResidualBlock(in_features) for _ in range(num_dense_layers)])

    def forward(self, x):
        return x + self.residual_blocks(x)


class Generator(nn.Module):
    """
    Define the Generator network for solar images with 1 channel
    """
    def __init__(self, in_channels=1, initial_channel=64,num_rrdb_blocks=4, upscale_factor=8):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, initial_channel, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # RRDB blocks
        self.rrdbs = nn.Sequential(*[RRDB(initial_channel) for _ in range(num_rrdb_blocks)])

        # Post-residual blocks
        self.post_rrdb = nn.Sequential(
            nn.Conv2d(initial_channel, initial_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        # Upsampling layers
        self.upsampling = nn.Sequential(
            *[nn.Conv2d(initial_channel, 4*initial_channel, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()]*int(np.log2(upscale_factor)))
        # Output layer
        self.output = nn.Conv2d(initial_channel, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        rrdbs = self.rrdbs(initial)
        post_rrdb = self.post_rrdb(rrdbs + initial)
        upsampled = self.upsampling(post_rrdb)
        return self.output(upsampled)