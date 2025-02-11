import torch
import torch.nn as nn
import numpy as np
import lightning as L
import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image
from .discriminator import DiscriminatorSRGAN
from .loss import CombinedLoss

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


class LightningGAN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        
        # Initialize networks using unpacked config sections
        self.generator = Generator(**config['generator'])
        self.discriminator = DiscriminatorSRGAN(**config['discriminator'])
        
        # Initialize loss
        self.combined_loss = CombinedLoss(self.discriminator, **config['loss'])

    def training_step(self, batch, batch_idx):
        # Get optimizers
        opt_g, opt_d = self.optimizers()
        
        # Get data
        lr_imgs, hr_imgs = batch
        
        # Train Discriminator
        opt_d.zero_grad()
        
        # Generate fake images
        fake_imgs = self.generator(lr_imgs)
        
        # Real images
        real_preds = self.discriminator(hr_imgs)
        real_labels = torch.ones_like(real_preds, device=self.device)
        d_real_loss = nn.BCEWithLogitsLoss()(real_preds, real_labels)
        
        # Fake images
        fake_preds = self.discriminator(fake_imgs.detach())
        fake_labels = torch.zeros_like(fake_preds, device=self.device)
        d_fake_loss = nn.BCEWithLogitsLoss()(fake_preds, fake_labels)
        
        # Combined discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Train Generator
        opt_g.zero_grad()
        
        # Calculate generator losses
        g_loss, mse_loss, adv_loss, percep_loss = self.combined_loss(fake_imgs, hr_imgs)
        
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Logging
        self.log_dict({
            'g_loss': g_loss,
            'd_loss': d_loss,
            'mse_loss': mse_loss,
            'adv_loss': adv_loss,
            'percep_loss': percep_loss
        }, prog_bar=True, sync_dist=True)
        
        # Visualize results periodically
        if batch_idx % 100 == 0:
            self._log_images(fake_imgs, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        fake_imgs = self.generator(lr_imgs)
        
        # Calculate validation losses
        g_loss, mse_loss, adv_loss, percep_loss = self.combined_loss(fake_imgs, hr_imgs)
        
        # Logging
        self.log_dict({
            'val_g_loss': g_loss,
            'val_mse_loss': mse_loss,
            'val_adv_loss': adv_loss,
            'val_percep_loss': percep_loss
        }, prog_bar=True, sync_dist=True)
        
        # Visualize first batch
        if batch_idx == 0:
            self._log_images(fake_imgs, batch_idx, 'val')

    def _log_images(self, images, batch_idx, prefix='train'):
        """Helper method to log images to wandb"""
        fig, ax = plt.subplots()
        # Assuming images need to be inverse scaled - adjust this based on your preprocessing
        img_to_plot = self._inverse_transform(images[0]).cpu().numpy().squeeze()
        ax.imshow(img_to_plot, cmap='afmhot')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        image = Image.open(buf)
        image_np = np.array(image)
        
        wandb_image = wandb.Image(image_np, caption=f"{prefix} Image Batch {batch_idx}")
        self.logger.experiment.log({f"{prefix}_image_batch_{batch_idx}": wandb_image})

    def _inverse_transform(self, x):
        """Inverse transform for your specific scaling - adjust as needed"""
        return x  # Implement your inverse transform here

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.generator.lr,
            betas=(0.9, 0.999)
        )
        d_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator.lr,
            betas=(0.9, 0.999)
        )
        return [g_opt, d_opt]
    



