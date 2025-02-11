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
    
#####################################################################################
############################### dicriminator ########################################
#####################################################################################

import torch.nn as nn
import torch
# Initialize the generator
import lightning as L

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, channel_list=[64, 128, 256], lr=1e-6):

        super().__init__()
        self.lr = lr
        kernel_size = 4
        padding = 1
        
        layers = [
            nn.Conv2d(in_channels, channel_list[0], kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        for i in range(1, len(channel_list)):
            layers += [
                nn.Conv2d(channel_list[i - 1], channel_list[i], kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm2d(channel_list[i]),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        layers += [
            nn.Conv2d(channel_list[-1], channel_list[-1], kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channel_list[-1]),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        
        layers += [
            nn.Conv2d(channel_list[-1], 1, kernel_size=kernel_size, stride=1, padding=padding)
        ]  
        
        self.model = nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.model(input_tensor)
    
    def flops_and_parameters(self, input_shape):
        from ptflops import get_model_complexity_info
        flops, parameters = get_model_complexity_info(self, input_shape, as_strings=True, print_per_layer_stat=False)
        return flops, parameters

import numpy as np

############################################################################
############################## Generator ###################################
############################################################################
class ResidualBlock(nn.Module):
    """
    Define a Residual Block without Batch Normalization
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class RRDB(nn.Module):
    """
    Define the Residual in Residual Dense Block (RRDB)
    """
    def __init__(self, in_features, num_dense_layers=3):
        super(RRDB, self).__init__()
        self.residual_blocks = nn.Sequential(*[ResidualBlock(in_features) for _ in range(num_dense_layers)])

    def forward(self, x):
        return x + self.residual_blocks(x)


class Generator(nn.Module):
    """
    Define the Generator network for solar images with 1 channel
    """
    def __init__(self, in_channels=1, initial_channel=64, num_rrdb_blocks=4, upscale_factor=4, lr=1e-4, **kwargs):
        super(Generator, self).__init__()

        self.lr = lr

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, initial_channel, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # RRDB blocks
        self.rrdbs = nn.Sequential(*[RRDB(initial_channel) for _ in range(num_rrdb_blocks)])

        # Post-residual blocks
        self.post_rrdb = nn.Sequential(
            nn.Conv2d(initial_channel, initial_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        # Upsampling layers
        self.upsampling = nn.Sequential(
            *[nn.Conv2d(initial_channel, 4*initial_channel, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()]*int(np.log2(upscale_factor)))
        # Output layer
        self.output = nn.Conv2d(initial_channel, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        rrdbs = self.rrdbs(initial)
        post_rrdb = self.post_rrdb(rrdbs + initial)
        upsampled = self.upsampling(post_rrdb)
        return self.output(upsampled)
#############################################################################
########################### Perceptual loss and other losses ################
#############################################################################


"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
import torch.nn.functional as F
from abc import abstractmethod
import os, hashlib
import requests
from tqdm import tqdm
from torchvision.transforms import Normalize

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)




class VAELOSS(nn.Module):
    def __init__(self, perceptual_weight=1.0, l2_weight=0.01, adversarial_weight=0.001, kl_weight=0.000001):
        super().__init__()
        self.lpips = LPIPS()
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.adversarial_weight = adversarial_weight
        self.l2_weight = l2_weight
    
    def kl_loss(self, mean, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return kl_div
    
    def l2_loss(self, input, reconstructed):
        return F.mse_loss(input, reconstructed)
    
    def perceptual_loss(self, hr, reconstructed):
        return self.lpips(hr, reconstructed)

    def adversarial_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss
    
    def g_loss(self, fake_logits):
        return -torch.mean(fake_logits)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load pre-trained VGG19 model
        self.vgg = models.vgg19(pretrained=True).features
        self.vgg = self.vgg.to('cuda' if torch.cuda.is_available() else 'cpu')
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Define layers to use for feature extraction
        self.layers = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '17': 'relu3_3',
            '26': 'relu4_3'
        }

        # Normalization for VGG19
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_features(self, image):
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

    def forward(self, target_image, output_image):
        # Preprocess the images
        #target_image = target_image/target_image.max()
        target_image = target_image.repeat(1,3,1,1)

        #output_image = output_image/output_image.max()
        output_image = output_image.repeat(1,3,1,1)

        target_image = self.normalize(target_image)
        output_image = self.normalize(output_image)

        # Extract features
        target_features = self.get_features(target_image)
        output_features = self.get_features(output_image)

        # Calculate Perceptual Loss
        loss = 0.0
        for layer in self.layers.values():
            loss += torch.nn.functional.l1_loss(target_features[layer], output_features[layer])

        return 
    
######################################################################
######################## final Gan ###################################
######################################################################
import os
import torch
import torch.nn as nn
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

class GAN(L.LightningModule):
    def __init__(self, **configs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.generator = Generator(**configs['generator'])
        self.discriminator = Discriminator(**configs['discriminator'])
        self.loss = VAELOSS(**configs['loss'])


    def training_step(self, batch, batch_idx):
      opt_g, opt_disc = self.optimizers()
      
      lr, hr = batch
      sr = self.generator(lr)
      ###### discriminator #######
      logits_real = self.discriminator(hr.contiguous().detach())      
      logits_fake = self.discriminator(sr.contiguous().detach())      
      d_loss = self.loss.adversarial_loss(logits_real, logits_fake)
      self.log('d_loss', d_loss, prog_bar=True, logger=True)
      ##### generator ######
      opt_disc.zero_grad()
      self.manual_backward(d_loss)
      opt_disc.step()

      logits_fake = self.discriminator(sr)
      g_loss = self.loss.g_loss(logits_fake)
      l2_loss = self.loss.l2_loss(hr, sr)
      perceptual_loss = self.loss.perceptual_loss(hr, sr).mean()
      self.log('train_g_loss', g_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_l2_loss', l2_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      self.log('train_perceptual_loss', perceptual_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

      perceptual_component = self.loss.perceptual_weight * perceptual_loss
      l2_component = self.loss.l2_weight * l2_loss
      adversarial_component = self.loss.adversarial_weight * g_loss

      loss = perceptual_component + l2_component + adversarial_component 


      opt_g.zero_grad()
      self.manual_backward(loss)
      opt_g.step()

      if (batch_idx % 100) == 0:
            fig, ax = plt.subplots()
            ax.imshow(inverse_rescalee(sr)[0].detach().cpu().numpy().squeeze(), cmap='afmhot')
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
        lr, hr = x
        sr = self.generator(lr)
        logits_fake = self.discriminator(sr)
        
        g_loss = self.loss.g_loss(logits_fake)
        l2_loss = self.loss.l2_loss(hr, sr)
        perceptual_loss = torch.mean(self.loss.perceptual_loss(hr, sr))
        
        self.log('val_g_loss', g_loss, prog_bar=True, sync_dist=True)
        self.log('val_l2_loss', l2_loss, prog_bar=True, sync_dist=True)
        self.log('val_perceptual_loss', perceptual_loss, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
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

    def configure_optimizers(self):
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator.lr, betas=(0.5, 0.9)) 
        vae_opt = torch.optim.Adam(self.generator.parameters(), lr=self.generator.lr, betas=(0.5, 0.9)) 
        return [vae_opt, disc_opt]



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

    return 

def resize(image):
    lr = F.interpolate(image, size=(int(128), int(128)), mode='bilinear', align_corners=False)
    return lr

if __name__ == '__main__':
    config = load_config(os.path.join('config', 'configGan.yml'))
    
    checkpoint_callback = ModelCheckpoint(**config['callbacks']['checkpoint'])

    logger = WandbLogger(**config['logger'], config=config)

    transform = transforms.Compose([rescalee,
    transforms.RandomHorizontalFlip(p=0.5)])  # 50% chance to flip horizontally


    datamodule = DataModule(**config['data'],
                            aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
                            aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                            transform=transform
                            )
    
    gan = GAN(**config)

    logger.watch(gan, log='all')

    trainer = L.Trainer(logger=logger,
                        callbacks=checkpoint_callback,
                        **config['trainer']
                        )
    trainer.fit(gan, datamodule)
    trainer.test(gan, datamodule)