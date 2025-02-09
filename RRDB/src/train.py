import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
import yaml
from pathlib import Path
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

from src.RRDB import LightningGenerator
from src.datamodule import DataModule

def power_transform(images, lambda_param=0.1):
    """
    Apply power transform (Box-Cox like transform) to the images.
    Args:
        images: Input tensor
        lambda_param: Power transform parameter (default=0.5 for square root)
    Returns:
        Transformed tensor
    """
    # Ensure positive values
    eps = 1e-8
    images = torch.clamp(images, min=eps)
    
    # Apply power transform
    if lambda_param == 0:
        transformed = torch.log(images)
    else:
        transformed = (images ** lambda_param - 1) / lambda_param
    
    # Normalize to [0, 1]
    min_val = transformed.min()
    max_val = transformed.max()
    normalized = (transformed - min_val) / (max_val - min_val)
    
    return normalized

def rescale(images):
    rescaled_images = images / 20000
    rescaled_images = (rescaled_images*2) - 1
    return rescaled_images

def train():
    # Load config
    with open('src/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['trainer']['checkpoint_dir'], exist_ok=True)

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config['project_name'],
        name=config['run_name'],
        config=config
    )

    # Define transforms with power transform instead of rescalee
    transform = transforms.Compose([
        power_transform,  # Replace rescalee with power_transform
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    # Initialize DataModule and Model with transforms
    datamodule = DataModule(**config['data'], transform=transform)
    model = LightningGenerator(config)
    wandb_logger.watch(model, log='all')
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(**config['callbacks']['checkpoint']),
    ]

    # Initialize Trainer
    trainer = L.Trainer(
        **config['trainer'],
        logger=wandb_logger,
        callbacks=callbacks
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train()
