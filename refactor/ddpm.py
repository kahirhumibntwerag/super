#############################################################
#################### datamodule ############################
###########################################################
import lightning as L
from torch.utils.data import DataLoader
from data.Dataset import Dataset
from torchvision import transforms
import dask.array as da
import boto3
import torch
import io
import os

def load_tensor_from_s3(bucket_name, s3_key, aws_access_key=None, aws_secret_key=None, region_name='eu-north-1', save_to_disk_path=None):
    if aws_access_key and aws_secret_key:
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=region_name)
    else:
        s3 = boto3.client('s3', region_name=region_name)
        
    download_buffer = io.BytesIO()

    try:
        s3.download_fileobj(bucket_name, s3_key, download_buffer)
        download_buffer.seek(0)
        tensor = torch.load(download_buffer)
        print("Tensor loaded successfully from S3.")

        # Save the tensor to disk if save_to_disk_path is provided
        if save_to_disk_path:
            torch.save(tensor, save_to_disk_path)
            print(f"Tensor saved to disk at {save_to_disk_path}")

        return tensor

    except Exception as e:
        print(f"Error loading tensor from S3: {e}")
        return None

class DataModule(L.LightningDataModule):
    def __init__(self, bucket_name, s3_key_train, s3_key_val, s3_key_test, aws_access_key, aws_secret_key=None, batch_size=4, downsample_factor=1/4, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.transform = transform
        self.s3_key_train = s3_key_train
        self.s3_key_val = s3_key_val
        self.s3_key_test = s3_key_test
        self.bucket_name = bucket_name
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key

    def prepare_data(self):
        print("Connecting to S3 and downloading metadata...")
        if not os.path.exists('5K.pt'):
            load_tensor_from_s3(bucket_name=self.bucket_name, s3_key=self.s3_key_train, aws_access_key=self.aws_access_key, aws_secret_key=self.aws_secret_key, save_to_disk_path='train_data.pt')
        if not os.path.exists('val_data.pt'):
            load_tensor_from_s3(bucket_name=self.bucket_name, s3_key=self.s3_key_val, aws_access_key=self.aws_access_key, aws_secret_key=self.aws_secret_key, save_to_disk_path='val_data.pt')
        if not os.path.exists('test_data.pt'):
            load_tensor_from_s3(bucket_name=self.bucket_name, s3_key=self.s3_key_test, aws_access_key=self.aws_access_key, aws_secret_key=self.aws_secret_key, save_to_disk_path='test_data.pt')
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Dataset(tensors=torch.load('5K.pt'), downsample_factor=self.downsample_factor, transform=self.transform)
            self.val_dataset = Dataset(tensors=torch.load('val_data.pt'), downsample_factor=self.downsample_factor, transform=self.transform)

        if stage == "test" or stage is None:
            self.test_dataset = Dataset(tensors=torch.load('test_data.pt'), downsample_factor=self.downsample_factor, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=190, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=190, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=190, persistent_workers=True)
    
###################################################################
############################# diffusion ###########################
###################################################################
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from inspect import isfunction
import numpy as np
from schedules import extract_into_tensor, exists

to_torch = partial(torch.tensor, dtype=torch.float32)

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class BetaSchedule(ABC):
    @abstractmethod
    def betas(self):
        pass

class LinearSchedule(BetaSchedule):
    def __init__(self, timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        self.timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end

    def betas(self):
        return (
            torch.linspace(self.linear_start ** 0.5, self.linear_end ** 0.5, self.timesteps, dtype=torch.float64) ** 2
        )


to_torch = partial(torch.tensor, dtype=torch.float32)

class Diffusion(nn.Module):
    def __init__(self,
                parameterization: str ='eps',
                v_posterior: float = 0.0
                ):

        super().__init__()
        self.parameterization = parameterization
        self.v_posterior = v_posterior
        self.schedule = LinearSchedule()
        self.register_schedule()

    def register_schedule(self):

        betas =  self.schedule.betas().numpy()
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod.shape[0] == self.schedule.timesteps

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas

        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")

        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def add_noise(self, x0, t):
      noise = torch.randn_like(x0)

      sqrt_alphas_cumprod = extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape)
      sqrt_one_minus_alphas_cumprod = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

      xt = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
      return xt, noise

#########################################################
######################### LDM ###########################
#########################################################
import os
from lightning.pytorch.callbacks import ModelCheckpoint

import torch.nn as nn
import torch
from torch.optim import AdamW
from schedules import LinearSchedule, BetaSchedule
from unet import Unet
from diffusion import Diffusion
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.Dataset import Dataset
from unet2 import Unet
from torchvision import transforms
from lightning.pytorch.loggers import WandbLogger
from refactor.vae.vae_gan import VAEGAN
import yaml
class LDM(nn.Module):
    def __init__(self, **configs):
        super().__init__()
        self.diffusion = Diffusion()
        self.vae = VAEGAN.load_from_checkpoint(configs['vae']['path'], vae=configs['vae']['generator'], discriminator=configs['vae']['discriminator'], loss=configs['vae']['loss']).vae.eval()
        self.unet = Unet(**configs['unet'])
        self.loss = nn.MSELoss()
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def training_step(self, batch):
        lr, hr = batch
        z, _, _ = self.vae.encoder(hr)

        t = torch.randint(1000, size=(z.shape[0],))
        zt, noise = self.diffusion.add_noise(z, t)
        predicted_noise = self.unet(torch.cat([zt, lr], dim=1), t)
        loss = self.loss(noise, predicted_noise)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch):
        lr, hr = batch
        z, _, _ = self.vae.encoder(hr)

        t = torch.randint(1000, size=(z.shape[0],))
        zt, noise = self.diffusion.add_noise(z, t)
        predicted_noise = self.unet(torch.cat([zt, lr], dim=1), t)
        loss = self.loss(noise, predicted_noise)
        self.log('val_loss', loss, prog_bar=True)

        return loss

        
    def configure_optimizers(self):
        return AdamW(self.unet.parameters(), lr=self.lr)

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
    config = load_config(os.path.join('config', 'ldm.yml'))
    
    checkpoint_callback = ModelCheckpoint(**config['callbacks']['checkpoint'])

    logger = WandbLogger(**config['logger'], config=config)

    transform = transforms.Compose([rescalee])
    datamodule = DataModule(**config['data'],
                            aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
                            aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                            transform=transform
                            )
    
    gan = LDM(**config['ldm'])

    logger.watch(gan, log='all')

    trainer = L.Trainer(logger=logger,
                        callbacks=checkpoint_callback,
                        **config['trainer']
                        )
    trainer.fit(gan, datamodule)
    trainer.test(gan, datamodule)
