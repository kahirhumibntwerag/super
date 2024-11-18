import sys
import os
from tqdm import tqdm

module_path = os.path.abspath(os.path.join('..', r'C:\Users\mhesh\OneDrive\Desktop\projee\super'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import torch.nn as nn
from vae import VAE
from discriminator import Discriminator
from lpips  import VAELOSS
from torch.utils.data import DataLoader
from data.Dataset import Dataset
from data.LoadData import load_single_aws_zarr, AWS_ZARR_ROOT, s3_connection
from torchvision import transforms

class DataModule:
    def __init__(self, batch_size, downsample_factor, transform):
        self.data = self.prepare_data()
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.transform = transform
    
    def prepare_data(self, year=2015, wavelength='171A'):
        return load_single_aws_zarr(path_to_zarr=AWS_ZARR_ROOT + str(year), wavelength=wavelength)

    
    def train_loader(self):
        train_data = self.prepare_train_data()
        train_dataset = Dataset(numpy_data=train_data, downsample_factor=self.downsample_factor, transform=self.transform)
        return DataLoader(train_dataset, batch_size=self.batch_size)
    
    def val_loader(self):
        val_data = self.prepare_val_data()
        val_dataset = Dataset(numpy_data=val_data, downsample_factor=self.downsample_factor, transform=self.transform)
        return DataLoader(val_dataset, batch_size=self.batch_size)       

    def prepare_train_data(self):
        return self.data[:100]

    def prepare_val_data(self):
        return self.data[1000:1100]
    

class VAEGAN(nn.Module):
    def __init__(self,
                  vae: VAE,
                 discriminator: Discriminator,
                 loss: VAELOSS,
                 datamodule: DataModule,
                 epochs: int,
                 device: str
                ):
        super().__init__()
        self.vae = vae.to(device)
        self.discriminator = discriminator.to(device)
        self.loss = loss
        self.vae_opt, self.disc_opt = self.configure_optimizers()
        
        self.datamodule = datamodule
        self.epochs = epochs
        self.device = device


    def vae_training_step(self, x):
        z, mean, logvar = self.vae.encoder.encode(x)
        decoded = self.vae.decoder.decode(z)
        logits_fake = self.discriminator(decoded)
        
        g_loss = self.loss.g_loss(logits_fake)
        print(g_loss.shape)
        kl_losss = self.loss.kl_loss(mean, logvar)
        print(kl_losss.shape)
        l2_loss = self.loss.l2_loss(x, decoded)
        print(l2_loss.shape)
        perceptual_loss = self.loss.perceptual_loss(x, decoded).view(-1)
        print(perceptual_loss.shape)

        perceptual_component = self.loss.perceptual_weight * perceptual_loss
        l2_component = self.loss.l2_weight * l2_loss
        adversarial_component = self.loss.adversarial_weight * g_loss
        kl_component = self.loss.kl_weight * kl_losss

        loss = perceptual_component + l2_component + adversarial_component + kl_component
        print(loss.shape)

        return loss, decoded 
        
    def discriminator_training_step(self, x, decoded):
        logits_real = self.discriminator(x.contiguous().detach())      
        logits_fake = self.discriminator(decoded.contiguous().detach())      
        d_loss = self.loss.adversarial_loss(logits_real, logits_fake)
        return d_loss
    

    def training_step(self, x):
        #vae part
        self.vae_opt.zero_grad()
        loss, decoded = self.vae_training_step(x)
        loss.backward()
        self.vae_opt.step()

        #discriminator part
        self.disc_opt.zero_grad()
        d_loss = self.discriminator_training_step(x, decoded)
        d_loss.backward()
        self.disc_opt.step()

        return loss.detach().item(), d_loss.detach().item()
    
    def training_batches(self):
        self.vae.train()
        self.discriminator.train()
        training_loss_d = 0.0
        training_loss_v = 0.0
        for _, hr in tqdm(self.datamodule.train_loader()):
            hr = hr.to(self.device)
            vae_loss, disc_loss = self.training_step(hr)
            training_loss_v += vae_loss
            training_loss_d += disc_loss
        
        training_loss_v = training_loss_v/len(self.datamodule.train_loader())
        training_loss_d = training_loss_d/len(self.datamodule.train_loader())
        #print(f'VAE training loss: {training_loss_v/len(self.datamodule.train_loader())}')
        #print(f'Discriminator training loss: {training_loss_d/len(self.datamodule.train_loader())}')

        return training_loss_v, training_loss_d
    
    def main_loop(self):
        for epoch in range(self.epochs):
            training_loss_v, training_loss_d = self.training_batches()
            print(f'Epoch {epoch} | VAE training loss: {training_loss_v}')
            print(f'Epoch {epoch} | Discriminator training loss: {training_loss_d}')    
    
    
    def configure_optimizers(self):
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator.lr, betas=(0.5, 0.9)) 
        vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.vae.lr, betas=(0.5, 0.9)) 
        return vae_opt, disc_opt



if __name__ == '__main__':
    vae = VAE(in_channels=1)
    disc = Discriminator(input_channels=1)
    transform = transforms.Compose([transforms.ToTensor()])
    datamodule = DataModule(batch_size=1, downsample_factor=1/4, transform=transform)
    loss = VAELOSS(perceptual_weight=1.0, l2_weight=0.01, adversarial_weight=0.001, kl_weight=1e-6)
    gan = VAEGAN(vae, disc, loss, datamodule, epochs=100, device='cpu')
    gan.main_loop()
