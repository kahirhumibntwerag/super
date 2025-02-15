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
def train():
    # Load config
    with open('src/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        **config['logger'],
        config=config
    )

    # Define transforms with power transform instead of rescalee
    transform = None
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
