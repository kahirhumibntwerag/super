import os
import torch
import torch.nn as nn
from refactor.vae.test import VAE
from refactor.vae.discriminator import Discriminator
from refactor.vae.lpips  import VAELOSS
from refactor.vae.datamodule  import DataModule
from lightning.pytorch.callbacks import ModelCheckpoint

from torchvision import transforms
import os
import yaml
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
import lightning as L

import io
import matplotlib.pyplot as plt
import wandb
import numpy as np
from PIL import Image
      
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

class VAEGAN(L.LightningModule):
    def __init__(self, **configs):
        super().__init__()
        self.automatic_optimization = False
        self.vae = VAE(**configs['vae'])
        self.discriminator = Discriminator(**configs['discriminator'])
        self.loss = VAELOSS(**configs['loss'])

    def training_step(self, batch, batch_idx):
      opt_g, opt_disc = self.optimizers()
      
      _, hr = batch
      decoded, mean, logvar = self.vae(hr)
      disc_factor = self.adopt_weight(self.discriminator.disc_factor, self.global_step//2, threshold=self.discriminator.iter_start)
      ###### discriminator #######
      logits_real = self.discriminator(hr.contiguous().detach())      
      logits_fake = self.discriminator(decoded.contiguous().detach())      
      d_loss = disc_factor*self.loss.adversarial_loss(logits_real, logits_fake)
      self.log('d_loss', d_loss, prog_bar=True, logger=True)
      
      opt_disc.zero_grad()
      self.manual_backward(d_loss)
      opt_disc.step()
      ##### generator ######

      logits_fake = self.discriminator(decoded)
      g_loss = self.loss.g_loss(logits_fake)
      kl_loss = self.loss.kl_loss(mean, logvar)
      l2_loss = self.loss.l2_loss(hr, decoded)
      perceptual_loss = self.loss.perceptual_loss(hr, decoded).mean()
      
      self.log('train_g_loss', g_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_kl_loss', kl_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_l2_loss', l2_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_perceptual_loss', perceptual_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

      perceptual_component = self.loss.perceptual_weight * perceptual_loss
      l2_component = self.loss.l2_weight * l2_loss
      rec_loss = l2_component + perceptual_component
      nll_loss = rec_loss / torch.exp(self.discriminator.logvar) + self.discriminator.logvar 
      adversarial_component = self.loss.adversarial_weight * g_loss
      kl_component = self.loss.kl_weight * kl_loss
      
      last_layer = self.vae.decoder.conv_out.weight
      d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)

      loss = nll_loss + d_weight*disc_factor*adversarial_component + kl_component


      opt_g.zero_grad()
      self.manual_backward(loss)
      opt_g.step()

      if (batch_idx % 100) == 0:
            fig, ax = plt.subplots()
            ax.imshow(inverse_rescalee(decoded)[0].detach().cpu().numpy().squeeze(), cmap='afmhot')
            ax.axis('off')

            # Save the figure to a buffer in RGB format
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            # Convert buffer to a NumPy array
            image = Image.open(buf)
            image_np = np.array(image)

            # Log the image to Wandb
            wandb_image = wandb.Image(image_np, caption=f"train Image Batch {batch_idx} with afmhot colormap")
            self.logger.experiment.log({f"train_image_afmhot_batch_{batch_idx}": wandb_image})

    def validation_step(self, x, batch_idx):
        _, hr = x
        decoded, mean, logvar = self.vae(hr)
        logits_fake = self.discriminator(decoded)
        
        g_loss = self.loss.g_loss(logits_fake)
        kl_loss = self.loss.kl_loss(mean, logvar)
        l2_loss = self.loss.l2_loss(hr, decoded)
        perceptual_loss = torch.mean(self.loss.perceptual_loss(hr, decoded))
        
        self.log('val_g_loss', g_loss, prog_bar=True, sync_dist=True)
        self.log('val_kl_loss', kl_loss, prog_bar=True, sync_dist=True)
        self.log('val_l2_loss', l2_loss, prog_bar=True, sync_dist=True)
        self.log('val_perceptual_loss', perceptual_loss, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            fig, ax = plt.subplots()
            ax.imshow(inverse_rescalee(decoded)[0].cpu().numpy().squeeze(), cmap='afmhot')
            ax.axis('off')
            
            # Save the figure to a buffer in RGB format
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            # Convert buffer to a NumPy array
            image = Image.open(buf)
            image_np = np.array(image)

            # Log the image to Wandb
            wandb_image = wandb.Image(image_np, caption=f"Validation Image Batch {batch_idx} with afmhot colormap")
            self.logger.experiment.log({f"val_image_afmhot_batch_{batch_idx}": wandb_image})

    def configure_optimizers(self):
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator.lr, betas=(0.5, 0.9)) 
        vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.vae.lr, betas=(0.5, 0.9)) 
        return [vae_opt, disc_opt]
    
    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator.discriminator_weight
        return d_weight

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
    
    checkpoint_callback = ModelCheckpoint(**config['callbacks']['checkpoint'])

    logger = WandbLogger(**config['logger'], config=config)

    transform = transforms.Compose([rescalee,
    transforms.RandomRotation(degrees=(-30, 30)),  # Rotate between -15 to 15 degrees
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate within 10% of image size
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Apply Gaussian blur with random sigma
])
    datamodule = DataModule(**config['data'],
                            aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
                            aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                            transform=transform
                            )
    
    gan = VAEGAN(**config['vae_gan'])

    logger.watch(gan, log='all')

    trainer = L.Trainer(logger=logger,
                        callbacks=checkpoint_callback,
                        strategy=DDPStrategy(process_group_backend='gloo', find_unused_parameters=True),
                        **config['trainer']
                        )
    trainer.fit(gan, datamodule, ckpt_path='drive/MyDrive/epoch-epoch=349.ckpt')
    trainer.test(gan, datamodule)
