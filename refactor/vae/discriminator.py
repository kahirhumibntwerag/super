import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, channel_list=[64, 128, 256], lr=1e-6):

        super().__init__()
        self.lr = lr
        kernel_size = 4
        padding = 1
        
        layers = [
            nn.Conv2d(input_channels, channel_list[0], kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        for i in range(1, len(channel_list)):
            layers += [
                nn.Conv2d(channel_list[i - 1], channel_list[i], kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(channel_list[i]),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        layers += [
            nn.Conv2d(channel_list[-1], channel_list[-1], kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channel_list[-1]),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        
        layers += [
            nn.Conv2d(channel_list[-1], 1, kernel_size=kernel_size, stride=1, padding=padding)
        ]  
        
        self.model = nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.model(input_tensor)
    
    def flops_and_parameters(self, input_shape):
        from ptflops import get_model_complexity_info
        flops, parameters = get_model_complexity_info(self, input_shape, as_strings=True, print_per_layer_stat=False)
        return flops, parameters
