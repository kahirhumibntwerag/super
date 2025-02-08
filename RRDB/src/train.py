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

from src.RRDB import Generator
from src.datamodule import DataModule

def rescalee(images):
    images_clipped = torch.clamp(images, min=1)
    images_log = torch.log(images_clipped)
    max_value = torch.log(torch.tensor(20000))
    max_value = torch.clamp(max_value, min=1e-9)
    images_normalized = images_log / max_value
    return images_normalized

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

    # Define transforms
    transform = transforms.Compose([
        rescalee,
        transforms.RandomHorizontalFlip(p=0.5)  # 50% chance to flip horizontally
    ])

    # Initialize DataModule and Model with transforms
    datamodule = DataModule(config['data'], transform=transform)
    model = Generator(config['generator'])
    wandb_logger.watch(model, log='all')
    # Callbacks
    callbacks = [
        ModelCheckpoint(**config['callbacks']['checkpoint']),
    ]

    # Initialize Trainer
    trainer = L.Trainer(
        **config['trainer'],
        logger=wandb_logger,
        callbacks=callbacks    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    train()
