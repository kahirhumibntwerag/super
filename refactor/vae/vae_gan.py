import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from .vae import VAE
from .discriminator import Discriminator
from .lpips  import VAELOSS
from torch.utils.data import DataLoader
from data.Dataset import Dataset
from data.LoadData import load_single_aws_zarr, AWS_ZARR_ROOT, s3_connection
from torchvision import transforms
import wandb
import os
import yaml
import matplotlib.pyplot as plt


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
        return self.data[:10]

    def prepare_val_data(self):
        return self.data[1000:1010]
            

    
class Trainer:
    def __init__(self,
                 max_epochs: int,
                 device: str,
                 log_every: int,
                 checkpoint_path: str,
                 log_path: str
                 ):
        
        self.max_epochs = max_epochs
        self.device = device
        self.log_every = log_every
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.epoch = 0
    
    def fit(self, model, datamodule):
        self.model = model
        self.optimizers = model.configure_optimizers()
        self.train_loader = datamodule.train_loader()
        self.val_loader = datamodule.val_loader()
        self.load_checkpoint()  
        for self.epoch in tqdm(range(self.max_epochs)):
            self.fit_()
            self.validation_loop()
            self.print_losses()
            self.save_checkpoint()

    def fit_(self):
        self.model.train()
        for data in tqdm(self.train_loader):
            data = [t.to(self.device) for t in data ]
            for opt_idx in range(len(self.optimizers)):
                self.training_step(data, opt_idx)
    
    def training_step(self, data, opt_idx):
        loss = self.model.training_step(data, opt_idx)
        self.optimizers[opt_idx].zero_grad()
        loss.backward()
        self.optimizers[opt_idx].step()
    
    @torch.no_grad()
    def validation_loop(self):
        self.model.eval()
        for data in tqdm(self.val_loader):
            data = [t.to(self.device) for t in data ]
            self.model.validation_step(data)
        self.log_images(data)
        



    def save_checkpoint(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if self.epoch % self.log_every == 0:
            checkpoint = {
                'model': self.model,
                'optimizers': self.optimizers,
                'epoch': self.epoch
                }
            
            torch.save(checkpoint, os.path.join(self.log_path, f'checkpoint{self.epoch}.pth'))
            print(f"Checkpoint saved at {self.log_path}") 
    
    def load_checkpoint(self):
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            print(f"Invalid checkpoint path: {self.checkpoint_path}")
            return None

        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizers.load_state_dict(checkpoint['optimizers'])
        self.epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {self.checkpoint_path}, resuming at epoch {self.epoch}.")
    
    def print_losses(self):
        for loss_name, loss in self.model.logs.items():
            print(f'Epoch --> {self.epoch} | {loss_name} --> {sum(loss)/len(loss)}')
            self.model.logs[loss_name] = []
    
    @torch.no_grad()
    def log_images(self, image):
        image = image[0][0].detach().cpu().numpy()
        plt.imshow(image, cmap='afmhot')
        plt.axis('off')
        plt.savefig('image.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        



      



class VAEGAN(nn.Module):
    def __init__(self, **configs):
        super().__init__()
        
        self.vae = VAE(**configs['vae'])
        self.discriminator = Discriminator(**configs['discriminator'])
        self.loss = VAELOSS(**configs['loss'])
        self.logs = {} 

    def training_step(self, x, opt_idx):
        z, mean, logvar = self.vae.encoder.encode(x)
        decoded = self.vae.decoder.decode(z)
        
        if opt_idx == 0:
            logits_fake = self.discriminator(decoded)
            
            g_loss = self.loss.g_loss(logits_fake)
            kl_losss = self.loss.kl_loss(mean, logvar)
            l2_loss = self.loss.l2_loss(x, decoded)
            perceptual_loss = self.loss.perceptual_loss(x, decoded).view(-1)

            perceptual_component = self.loss.perceptual_weight * perceptual_loss
            l2_component = self.loss.l2_weight * l2_loss
            adversarial_component = self.loss.adversarial_weight * g_loss
            kl_component = self.loss.kl_weight * kl_losss

            loss = perceptual_component + l2_component + adversarial_component + kl_component
            return loss 
        
        if opt_idx == 1:
            logits_real = self.discriminator(x.contiguous().detach())      
            logits_fake = self.discriminator(decoded.contiguous().detach())      
            d_loss = self.loss.adversarial_loss(logits_real, logits_fake)
            return d_loss
    
    
    def configure_optimizers(self):
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator.lr, betas=(0.5, 0.9)) 
        vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.vae.lr, betas=(0.5, 0.9)) 
        return vae_opt, disc_opt
    
    def log(self, name, to_log):
        if not self.logs[name]:
            self.logs[name] = []
        self.logs[name] += [to_log.detach().item()] 





def load_config(config_path):
    """loading the config file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)




if __name__ == '__main__':
    config = load_config(r'config\configG.yml')
    transform = transforms.Compose([transforms.ToTensor()])
    datamodule = DataModule(**config['data'], transform=transform )
    vae = VAE(**config['vae_gan']['vae'])
    #gan = VAEGAN(**config['vae_gan'])
    trainer = Trainer(**config['trainer'])
    trainer.fit(vae, datamodule)
