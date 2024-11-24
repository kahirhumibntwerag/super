import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from refactor.vae.test import VAE
from refactor.vae.discriminator import Discriminator
from refactor.vae.lpips  import VAELOSS
from torch.utils.data import DataLoader
from data.Dataset import Dataset
from data.LoadData import load_single_aws_zarr, AWS_ZARR_ROOT, s3_connection
from torchvision import transforms
import wandb
import os
import yaml
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

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
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
    
    def val_loader(self):
        val_data = self.prepare_val_data()
        val_dataset = Dataset(numpy_data=val_data, downsample_factor=self.downsample_factor, transform=self.transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, sampler=DistributedSampler(val_dataset))       

    def prepare_train_data(self):
        return self.data[:100]

    def prepare_val_data(self):
        return self.data[100:110]
            

    
class Trainer:
    def __init__(self,
                 max_epochs: int,
                 device: str,
                 log_every: int,
                 checkpoint_path: str,
                 log_path: str,
                 accelerator: bool
                 ):
        
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        if accelerator == 'ddp':
            self.setup_ddp()
            self.device = int(os.environ['LOCAL_RANK'])
        else:
            self.device = device
        self.log_every = log_every
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.epoch = 0
    
    def fit(self, model, datamodule):
        self.model = model.to(self.device)
        if self.accelerator == 'ddp':
            self.model = DDP(self.model, device_ids=[self.device])
        #if self.device == 0:
        #    wandb.init(project="your_project_name")
        #    wandb.watch(self.model, log='all', log_freq=5)
        self.optimizers = model.configure_optimizers()
        self.train_loader = datamodule.train_loader()
        self.val_loader = datamodule.val_loader()
        self.load_checkpoint()  
        for self.epoch in tqdm(range(self.max_epochs)):
            self.fit_()
            self.validation_loop()
            self.print_losses()
            self.save_checkpoint()
        
        if self.accelerator == 'ddp':
            destroy_process_group()
        #if self.device == 0:
        #    wandb.finish()

    def fit_(self):
        self.model.train()
        for data in self.train_loader:
            data = [t.to(self.device) for t in data ]
            for opt_idx in range(len(self.optimizers)):
                self.training_step(data, opt_idx)
    
    def training_step(self, data, opt_idx):
        loss = self.model.module.training_step(data, opt_idx)
        self.optimizers[opt_idx].zero_grad()
        loss.backward()
        self.optimizers[opt_idx].step()
    
    @torch.no_grad()
    def validation_loop(self):
        self.model.eval()
        for data in self.val_loader:
            data = [t.to(self.device) for t in data ]
            self.model.module.validation_step(data)
        self.log_images(data)
        



    def save_checkpoint(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if self.device == 0 and self.epoch % self.log_every == 0:
            checkpoint = {
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': [opt.state_dict() for opt in self.optimizers],
                'epoch': self.epoch
            }

            checkpoint_file = os.path.join(self.log_path, f'checkpoint{self.epoch}.pth')
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved at {checkpoint_file}")

    def load_checkpoint(self):
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            print(f"Invalid checkpoint path: {self.checkpoint_path}")
            return

        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        for opt, state_dict in zip(self.optimizers, checkpoint['optimizer_state_dict']):
            opt.load_state_dict(state_dict)

        self.epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {self.checkpoint_path}, resuming at epoch {self.epoch}.")

    def print_losses(self):
        for loss_name, loss in self.model.module.logs.items():
            print(f'Epoch --> {self.epoch} | {loss_name} --> {sum(loss)/len(loss)}')
            #wandb.log({loss_name:sum(loss)/len(loss)})
            self.model.module.logs[loss_name] = []
    
    @torch.no_grad()
    def log_images(self, data):
        _, image = data
        decoded, _, _ = self.model.module.vae(image)
        decoded = inverse_rescalee(decoded)[0][0].detach().cpu().numpy()
        plt.imshow(decoded, cmap='afmhot')
        plt.axis('off')
        plt.savefig(os.path.join(self.log_path, f'image{self.epoch}.png'), bbox_inches='tight', pad_inches=0)
        plt.clf()
        
        image = inverse_rescalee(image)[0][0].detach().cpu().numpy()
        plt.imshow(image, cmap='afmhot')
        plt.axis('off')
        plt.savefig(os.path.join(self.log_path, f'imageO{self.epoch}.png'), bbox_inches='tight', pad_inches=0)
        plt.clf()
    
    def setup_ddp(self):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend='nccl')
            



      



class VAEGAN(nn.Module):
    def __init__(self, **configs):
        super().__init__()
        
        self.vae = VAE()
        self.discriminator = Discriminator(**configs['discriminator'])
        self.loss = VAELOSS(**configs['loss'])
        self.logs = {} 

    def training_step(self, x, opt_idx):
      _, hr = x
      decoded, mean, logvar = self.vae(hr)
      
      if opt_idx == 0:

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
      
      if opt_idx == 1:
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
    
    def log(self, name, to_log):
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append(to_log.detach().item())





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
    trainer = Trainer(**config['trainer'])
    trainer.fit(gan, datamodule)
