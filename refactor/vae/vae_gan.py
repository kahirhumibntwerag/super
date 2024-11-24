import os
import torch
import torch.nn as nn
from refactor.vae.test import VAE
from refactor.vae.discriminator import Discriminator
from refactor.vae.lpips  import VAELOSS
from refactor.vae.datamodule  import DataModule

from torchvision import transforms
import wandb
import os
import yaml
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
import lightning as L



      



class VAEGAN(L.LightningModule):
    def __init__(self, **configs):
        super().__init__()
        
        self.vae = VAE(**configs['vae'])
        self.discriminator = Discriminator(**configs['discriminator'])
        self.loss = VAELOSS(**configs['loss'])

    def training_step(self, batch, optimizer_idx):
      _, hr = batch
      decoded, mean, logvar = self.vae(hr)
      
      if optimizer_idx == 0:

          logits_fake = self.discriminator(decoded)
          g_loss = self.loss.g_loss(logits_fake)
          kl_loss = self.loss.kl_loss(mean, logvar)
          l2_loss = self.loss.l2_loss(hr, decoded)
          perceptual_loss = self.loss.perceptual_loss(hr, decoded).mean()
          self.log('train_g_loss', g_loss)
          self.log('train_kl_loss', kl_loss)
          self.log('train_l2_loss', l2_loss)
          self.log('train_perceptual_loss', perceptual_loss)

          perceptual_component = self.loss.perceptual_weight * perceptual_loss
          l2_component = self.loss.l2_weight * l2_loss
          adversarial_component = self.loss.adversarial_weight * g_loss
          kl_component = self.loss.kl_weight * kl_loss

          loss = perceptual_component + l2_component + adversarial_component + kl_component
          return loss 
      
      if optimizer_idx == 1:
            logits_real = self.discriminator(hr.contiguous().detach())      
            logits_fake = self.discriminator(decoded.contiguous().detach())      
            d_loss = self.loss.adversarial_loss(logits_real, logits_fake)
            self.log('d_loss', d_loss)
            return d_loss
      
    def validation_step(self, x):
      _, hr = x
      decoded, mean, logvar = self.vae(hr)
     
      logits_fake = self.discriminator(decoded)
      
      g_loss = self.loss.g_loss(logits_fake)
      kl_loss = self.loss.kl_loss(mean, logvar)
      l2_loss = self.loss.l2_loss(hr, decoded)
      perceptual_loss = torch.mean(self.loss.perceptual_loss(hr, decoded))

      self.log('val_g_loss', g_loss)
      self.log('val_kl_loss', kl_loss)
      self.log('val_l2_loss', l2_loss)
      self.log('val_perceptual_loss', perceptual_loss)
  
    
    def configure_optimizers(self):
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator.lr, betas=(0.5, 0.9)) 
        vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.vae.lr, betas=(0.5, 0.9)) 
        return vae_opt, disc_opt


def load_config(config_path):
    """loading the config file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def rescalee(images):
    images_clipped = torch.clamp(images, min=1)
    images_log = torch.log(images_clipped)
    max_value = torch.log(torch.tensor(20000))
    max_value = torch.clamp(max_value, min=1e-9)
    images_normalized = images_log / max_value
    return images_normalized
def inverse_rescalee(images_normalized):
    max_value = torch.log(torch.tensor(20000.0))
    max_value = torch.clamp(max_value, min=1e-9)
    images_log = images_normalized * max_value
    images_clipped = torch.exp(images_log)

    return images_clipped
def rescale(images):

    rescaled_images = images / 20000
    rescaled_images = (rescaled_images*2) - 1

    return rescaled_images
if __name__ == '__main__':
    config = load_config(os.path.join('config', 'configG.yml'))
    transform = transforms.Compose([transforms.ToTensor(), rescalee])
    datamodule = DataModule(**config['data'], transform=transform )
    #vae = VAE(**config['vae_gan']['vae'])
    gan = VAEGAN(**config['vae_gan'])
    trainer = L.Trainer(max_epochs=30)
    trainer.fit(gan, datamodule)
