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
import dask.array as da

class DataModule:
    def __init__(self, batch_size, downsample_factor, transform):
        self.data = self.prepare_data()
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.transform = transform
    
    def prepare_data(self, wavelength='171A'):
        data = [load_single_aws_zarr(path_to_zarr=AWS_ZARR_ROOT + str(year), wavelength=wavelength) for year in range(2011, 2013)]
        data = da.concatenate(data, axis=0)
        return data


    
    def train_loader(self):
        train_data = self.prepare_train_data()
        train_dataset = Dataset(numpy_data=train_data, downsample_factor=self.downsample_factor, transform=self.transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
    
    def val_loader(self):
        val_data = self.prepare_val_data()
        val_dataset = Dataset(numpy_data=val_data, downsample_factor=self.downsample_factor, transform=self.transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, sampler=DistributedSampler(val_dataset))       

    def prepare_train_data(self):
        return self.data[::80]

    def prepare_val_data(self):
        return self.data[::700]
            

    
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
        if self.device == 0:
            wandb.init(project="your_project_name")
            wandb.watch(self.model, log='all', log_freq=5)
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
        if self.device == 0:
            wandb.finish()

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
            artifact = wandb.Artifact('model-checkpoints', type='model')
            artifact.add_file(checkpoint_file)
            wandb.log_artifact(artifact)

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
            if self.device == 0:
                wandb.log({loss_name:sum(loss)/len(loss)})
            self.model.module.logs[loss_name] = []
    
    @torch.no_grad()
    def log_images(self, data):
        _, image = data

        decoded, _, _ = self.model.module.vae(image)
        decoded = inverse_rescalee(decoded)[0][0].detach().cpu().numpy()
        
        plt.figure()
        plt.imshow(decoded, cmap='afmhot')
        plt.axis('off')
        plt.savefig(os.path.join(self.log_path, f'image{self.epoch}.png'), bbox_inches='tight', pad_inches=0)
        if self.device == 0:
            wandb.log({f"decoded_image_epoch_{self.epoch}": wandb.Image(plt, caption=f"Decoded Image at Epoch {self.epoch}")})
        plt.clf()
    def setup_ddp(self):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend='nccl')
            