import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch.nn as nn
import torch
from torch.optim import AdamW
from schedules import LinearSchedule, BetaSchedule
from loss import L2Loss, Loss
from unet import Unet
from diffusion import Diffusion
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.Dataset import Dataset
from data.LoadData import load_single_aws_zarr, AWS_ZARR_ROOT, s3_connection
from unet import Unet
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
        return self.data[:1000]

    def prepare_val_data(self):
        return self.data[1000:1100]
    
class DDPM(nn.Module):
    def __init__(self,
                 datamodule: DataModule,
                 diffusion: Diffusion,
                 loss: Loss,
                 model: Unet,
                 lr: float,
                 device: str
                 
                 ):
        super().__init__()
        
        self.diffusion = diffusion
        self.model = model
        self.loss = loss
        self.lr = lr
        self.optimizer = self.configure_optimizer()
        self.datamodule = datamodule
        self.device = device
    
    def training_step(self, x, c=None):
        t = torch.randint(1000, size=(x.shape[0],))
        xt, noise = self.diffusion.add_noise(x, t)
        predicted_noise = self.model(xt, t)
        return self.loss(noise, predicted_noise)
    
    def training_step_(self, x, c=None):
        self.optimizer.zero_grad()
        loss = self.training_step(x)        
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    
    
    def training_batches(self):
        self.model.train()
        training_loss = 0.0
        for lr, _ in self.datamodule.train_loader():
            lr = lr.to(self.device)            
            training_loss += self.training_step_(lr)
        
        return training_loss/len(self.datamodule.train_loader())
    
    def training_loop(self, max_epochs):
        for epoch in range(max_epochs):
            self.model.train()
            loss = self.training_batches()
            print(f'Epoch {epoch} | training loss {loss}')
            if self.datamodule.val_loader():
                print(f'Epoch {epoch} | training loss {self.validation_batches}')


    @torch.no_grad
    def validation_batches(self):
        self.model.eval()
        validation_loss = 0.0
        for lr, hr in self.datamodule.val_loader():
            lr, _ = lr.to(self.device)
            t = torch.randint(1000, size=hr.shape[0])
            
            validation_loss += self.validation_step(lr, t)
    
        return validation_loss/len(self.datamodule.val_loader())
    
    @torch.no_grad
    def validation_step(self, x, t, c=None):
        xt, noise = self.diffusion.add_noise(x, t)
        predicted_noise = self.model(xt)
        return self.loss(noise, predicted_noise).item()

    def log(self):
        pass
    
    def configure_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.lr)


if __name__ == '__main__':
    scheduel = LinearSchedule()
    diffusion = Diffusion(scheduel)
    loss = nn.MSELoss()
    model = Unet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 1e-4
    transform = transforms.Compose([transforms.ToTensor()])
    datamodule = DataModule(batch_size=1, downsample_factor=1/4, transform=transform)
    datamodule.prepare_data()
    ddpm = DDPM(
        datamodule,
        diffusion,
        loss,
        model,
        lr,
        device
        )
    
    ddpm.training_loop(100)