
import torch 
import torch.nn as nn
from refactor.unet import InConvBlock, OutConvBlock, Downsample, Upsample, ChannelChanger




class Resblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resblock = nn.Sequential(
            InConvBlock(in_channels, in_channels),
            OutConvBlock(in_channels)
        )
    def forward(self, x):
        return self.resblock(x) + x

class ResChain(nn.Module):
    def __init__(self, in_channels, num_resblocks):
        super().__init__()
        self.reschain = nn.Sequential(
            *[Resblock(in_channels) for _ in range(num_resblocks)]
            
        )
    def forward(self, x):
        return self.reschain(x)



class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=3, channels=[64, 128, 256, 512], num_resblocks=5):
        super().__init__()
        encoder = []
        
        self.proj_in = [nn.Conv2d(in_channels, channels[0], 1, padding=0)] 
        for i in range(len(channels)-1):
            encoder.append(ResChain(channels[i], num_resblocks))
            encoder.append(ChannelChanger(channels[i], channels[i+1]))
            encoder.append(Downsample(0.5))
        self.proj_out = [nn.Conv2d(channels[-1], 2*latent_channels, 1, padding=0)]
        
        encoder = self.proj_in + encoder + self.proj_out

        self.encoder = nn.Sequential(
            *encoder
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        e = torch.randn(size=mean.shape, device=mean.device)
        z = mean + torch.exp(logvar)*e
        return z, mean, logvar
    

class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_channels=3, channels=[64, 128, 256, 512], num_resblocks=5):
        super().__init__()
        
        decoder = []
        reversed_channels = list(reversed(channels))

        self.proj_in = [nn.Conv2d(latent_channels , reversed_channels[0], 1, padding=0)]
        for i in range(len(reversed_channels)-1):
            decoder.append(Upsample(2))
            decoder.append(ChannelChanger(reversed_channels[i], reversed_channels[i+1]))
            decoder.append(ResChain(reversed_channels[i+1], num_resblocks))
        self.proj_out = [nn.Conv2d(reversed_channels[-1], out_channels, 1, padding=0)]

        decoder = self.proj_in + decoder + self.proj_out

        self.decoder = nn.Sequential(
            *decoder
        )
    def decode(self, z):
        return self.decoder(z)



class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=3, channels=[64, 128, 256, 512], num_resblocks=5, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.encoder = Encoder(in_channels, latent_channels, channels, num_resblocks)
        self.decoder = Decoder(in_channels, latent_channels, channels, num_resblocks)
        self.logs = {}
    def forward(self, x):
        z, mean, logvar = self.encoder.encode(x)
        x = self.decoder.decode(z)
        return x, mean, logvar
    
    def training_step(self, data, opt_idx):
        if opt_idx == 0:
            _, hr = data
            hr = hr/hr.max()
            decoded, _, _ = self(hr)
            loss = nn.functional.mse_loss(hr, decoded)
            self.log('l2 training loss', loss)
            return loss
    
    def validation_step(self, data):
        _, hr = data
        hr = hr/hr.max()
        decoded, _, _ = self(hr)
        self.log_image(decoded)
        loss = nn.functional.mse_loss(hr, decoded)
        self.log('l2 validation loss', loss)
        return loss

    
    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.lr)]
    
    def log(self, name, to_log):
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append(to_log.detach().item())

    def log_image(self, image):
        import matplotlib.pyplot as plt
        image = image[0][0].detach().cpu().numpy()
        plt.imshow(image, cmap='afmhot')
        plt.axis('off')
        plt.savefig('image.png', bbox_inches='tight', pad_inches=0)
        plt.clf()


    
        
    def flops_and_parameters(self, input_shape):
        from ptflops import get_model_complexity_info
        flops, parameters = get_model_complexity_info(self, input_shape, as_strings=True, print_per_layer_stat=False)
        return flops, parameters


if __name__ == '__main__':
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        vae = VAE()
        z, mean, logvar = vae.encoder.encode(x)
        print(z.shape, mean.shape, logvar.shape)