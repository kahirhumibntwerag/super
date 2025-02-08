import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import lightning as L
import wandb

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


class LightningGenerator(L.LightningModule):
    def __init__(self, config):
        super(LightningGenerator, self).__init__()
        self.save_hyperparameters()
        self.generator = Generator(**config)
    
    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr)
        self.log('val_loss', loss)
        # Store predictions for epoch_end logging
        if batch_idx == 0:  # Only store first batch to save memory
            self.validation_step_outputs = {'lr': lr, 'hr': hr, 'sr': sr}
        return loss
    
    def test_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr)
        self.log('test_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        # Get the stored predictions
        lr = self.validation_step_outputs['lr']
        hr = self.validation_step_outputs['hr']
        sr = self.validation_step_outputs['sr']
        
        # Log only first few images
        num_images = min(4, lr.size(0))
        
        def tensor_to_image(tensor):
            # Move to CPU, convert to numpy, ensure proper shape
            img = tensor.cpu().float().clamp(0, 1).squeeze().numpy()
            return img
        
        # Create and log images
        for i in range(num_images):
            self.logger.experiment.log({
                f"validation/image_{i}/lr": wandb.Image(
                    tensor_to_image(lr[i]), 
                    caption="Low Resolution",
                    cmap='afmhot'
                ),
                f"validation/image_{i}/hr": wandb.Image(
                    tensor_to_image(hr[i]), 
                    caption="High Resolution",
                    cmap='afmhot'
                ),
                f"validation/image_{i}/sr": wandb.Image(
                    tensor_to_image(sr[i]), 
                    caption="Super Resolution",
                    cmap='afmhot'
                )
            })
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.generator.lr)
        return optimizer
    