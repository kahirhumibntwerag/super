import torch
import torch.nn as nn

class DiscriminatorSRGAN(nn.Module):
    def __init__(self, in_channels=1, channel_list=[16, 32, 64, 128], lr=1e-6):
        super(DiscriminatorSRGAN, self).__init__()
        self.lr = lr
        
        # Build feature layers dynamically
        layers = []
        
        # Input layer
        layers.extend([
            nn.Conv2d(in_channels, channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Hidden layers
        for i in range(len(channel_list)-1):
            # First conv at current channel size
            layers.extend([
                nn.Conv2d(channel_list[i], channel_list[i], kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                
                # Second conv increasing channel size
                nn.Conv2d(channel_list[i], channel_list[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
        self.features = nn.Sequential(*layers)

        # Calculate the flattened size with a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 512, 512)
            dummy_features = self.features(dummy_input)
            self.flattened_size = dummy_features.view(1, -1).size(1)

        # Define the classifier part of the discriminator
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
