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
            load_tensor_from_s3(bucket_name=self.bucket_name, s3_key=self.s3_key_train, aws_access_key=self.aws_access_key, aws_secret_key=self.aws_secret_key, save_to_disk_path='5K.pt')
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=12, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=12, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=12, persistent_workers=True)


#########################################################
######################### LDM ###########################
#########################################################
import os
from lightning.pytorch.callbacks import ModelCheckpoint

import torch.nn as nn
import torch
from torch.optim import AdamW
from .unet2 import Unet
from data.Dataset import Dataset
from torchvision import transforms
from lightning.pytorch.loggers import WandbLogger
from refactor.vae.vae_gan import VAEGAN
from .diffusion import Diffusion
from .sampler import Schedule, Runner
import yaml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb
class LDM(L.LightningModule):
    def __init__(self, **configs):
        super().__init__()
        self.save_hyperparameters()
        self.vae = VAEGAN.load_from_checkpoint(configs['vae']['path'], vae=configs['vae']['vae'], discriminator=configs['vae']['discriminator'], loss=configs['vae']['loss']).vae.eval()
        self.unet = Unet(**configs['unet'])
        self.loss = nn.MSELoss()
        self.diffusion = Diffusion()
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def training_step(self, batch):
        lr, hr = batch
        z, _, _ = self.vae.encoder(hr)

        t = torch.randint(1000, size=(z.shape[0],), device=z.device)
        zt, noise = self.add_noise(z, t)
        predicted_noise = self.unet(torch.cat([zt, lr], dim=1), t)
        loss = self.loss(noise, predicted_noise)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        z, _, _ = self.vae.encoder(hr)

        t = torch.randint(1000, size=(z.shape[0],), device=z.device)
        zt, noise = self.add_noise(z, t)
        predicted_noise = self.unet(torch.cat([zt, lr], dim=1), t)
        loss = self.loss(noise, predicted_noise)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        if (batch_idx % 10) == 0:
            sr = self.sample(lr, 50)
            fig, ax = plt.subplots()
            ax.imshow(inverse_rescalee(sr)[0].cpu().numpy().squeeze(), cmap='afmhot')
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
        return loss

        
    def configure_optimizers(self):
        return AdamW(self.unet.parameters(), lr=self.unet.lr)
    

    def sample(self, lrs, sample_speed, device):
        runner = Runner(Schedule(), sample_speed, device)
        skip = 1000//sample_speed
        seq = range(0, 1000, skip)
        noise = torch.randn(size=lrs.shape, device=lrs.device)
        z = runner.sample_image(lrs, noise, seq, self.unet)
        sr = self.vae.decoder(z)
        return sr
    


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
