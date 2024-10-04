from data.LoadData import load_single_aws_zarr, s3_connection, AWS_ZARR_ROOT, build_dataloader
from data.Dataset import Dataset
from tqdm import tqdm
import dask.array as da
import yaml
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from RRDB.src.RRDB import Generator, build_generator
import matplotlib.pyplot as plt
import argparse
import wandb  
from omegaconf import OmegaConf
import os
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def load_config(config_path):
    """loading the config file"""
    abs_path = os.path.abspath(config_path)
    config_path = Path(abs_path).resolve()
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train(model, dataloader, criterion, optimizer, device):
    """a function that train the model for only one step and returns 
       the average loss after the step 
    """
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, criterion, device):
    """a function that validate the model for only one step and returns 
       the average loss after the step 
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def save_checkpoint(model, optimizer, epoch, loss, file_path='checkpoint.pth'):
    """
    Save a model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The current loss.
        file_path (str): The path where the checkpoint will be saved.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at {file_path}")



def load_checkpoint(model, optimizer, custom_checkpoint_path):
    """
    Load a checkpoint from a specified path.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        custom_checkpoint_path (str): Path to the specific checkpoint file.

    Returns:
        int: The epoch to resume from, or 0 if no checkpoint is found.
        float: The loss value at the checkpoint, or None if no checkpoint is found.
    """
    import os

    # Check if the custom checkpoint path is valid
    if not custom_checkpoint_path or not os.path.isfile(custom_checkpoint_path):
        print(f"Invalid checkpoint path: {custom_checkpoint_path}")
        return 0, None  # Return 0 epoch if the path is invalid
    
    abs_path = os.path.abspath(custom_checkpoint_path)
    config_path = Path(abs_path).resolve()
    # Load the checkpoint
    checkpoint = torch.load(config_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from {custom_checkpoint_path}, resuming at epoch {epoch}.")
    
    return epoch, loss


def parse_and_merge_config(config_path):
    # Create a parser to accept arguments from the command line
    parser = argparse.ArgumentParser(description="Train the RRDB model with optional config overrides.")
    
    # Parse known arguments and handle unknown arguments separately
    parser.add_argument('--opt', nargs='+', default=None)
    args = parser.parse_args()
    
    # Load the configuration from the YAML file
    cfg = load_config(config_path)
    
    # Convert the loaded YAML config into an OmegaConf object
    cfg = OmegaConf.create(cfg)
    
    # Merge the loaded configuration with any command-line options
    config = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opt))
    
    # Return the final configuration
    return config


def main(rank, world_size, config_path):
    setup_ddp(rank, world_size)
    # Parse and merge configuration
    cfg = parse_and_merge_config(config_path)

    # Initialize the model
    generator = DDP(build_generator(cfg), device_ids=rank)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=cfg.RRDB.training.lr)

    # Load model checkpoint
    resume_epoch, loss = load_checkpoint(generator, optimizer, cfg.RRDB.training.checkpoint)

    # Build dataloaders for training and validation
    train_loader, val_loader = build_dataloader(cfg)

    # Training and validation loop
    train_losses = []
    validation_losses = []
    for epoch in range(resume_epoch, cfg.RRDB.training.epochs):
        print(f"Epoch {epoch + 1}/{cfg.RRDB.training.epochs}")

        # Training step
        train_loss = train(generator, train_loader, criterion, optimizer, rank)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation step
        val_loss = validate(generator, val_loader, criterion, rank)
        print(f"Validation Loss: {val_loss:.4f}")

        # Store losses for future use
        train_losses.append(train_loss)
        validation_losses.append(val_loss)

        # Save checkpoint at defined intervals
        if rank == 0 and epoch % cfg.RRDB.training.save_frequency == 0:
            checkpoint_path = Path(os.path.abspath(cfg.RRDB.training.save_checkpoint_in + f'/checkpoint_{cfg.data.year}_{cfg.data.wavelength}_{epoch}.pth')).resolve()
            save_checkpoint(generator, optimizer, epoch, train_loss, checkpoint_path)
    destroy_process_group()
    print("Training complete.")

if __name__ == "__main__":
    # Call the main function with the path to your config file
    config_path = 'config/config.yml'
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config_path), nprocs=world_size)



