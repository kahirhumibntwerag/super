from .unet import Unet
from refactor.vae.vae import VAE
import torch
import torch.nn as nn

if __name__ == '__main__':
    x = torch.randn(1, 1, 512, 512)
    t = torch.randint(1000, size=(1,))
    
    vaee = VAE(in_channels=1)
    unet = Unet()
    z, _, _ =  vaee.encoder.encode(x)
    
    print(z.shape)
    print(unet(z, t).shape)