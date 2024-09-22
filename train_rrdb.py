from data.LoadData import load_single_aws_zarr, s3_connection, AWS_ZARR_ROOT
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
from RRDB.src.RRDB import Generator
import matplotlib.pyplot as plt
import argparse
import wandb  



def load_config(config_path):
    """loading the config file"""
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
        'model_state_dict': model.state_dict(),
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

    # Load the checkpoint
    checkpoint = torch.load(custom_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from {custom_checkpoint_path}, resuming at epoch {epoch}.")
    
    return epoch, loss





if __name__ == "__main__":
    # Training the RRDB model from a config file
    parser = argparse.ArgumentParser(description="Train the RRDB model with optional config overrides.")

    # Add arguments to override config values
    parser.add_argument("--batch_size", type=int, help="Override batch size from config")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--lr", type=float, help="Override learning rate from config")
    parser.add_argument("--checkpoint", type=str, help="Override checkpoint path from config")
    parser.add_argument("--save_checkpoint_in", type=str, help="Override checkpoint save directory from config")
    parser.add_argument("--save_frequency", type=int, help="Override save frequency from config")
    parser.add_argument("--year", type=int, help="Override year from config")
    parser.add_argument("--wavelength", type=int, help="Override wavelength from config")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    # Load the configuration from the YAML file
    config = load_config(r'SR\config\config.yml')

    # Loading and initializing variables

    # Model variables initialization
    in_channels = config['RRDB']['model']['in_channels']
    initial_channel = config['RRDB']['model']['initial_channel']
    num_rrdb_blocks = config['RRDB']['model']['num_rrdb_blocks']
    upscale_factor = config['RRDB']['model']['upscale_factor']
    
    # Override training-related variables if command-line arguments are provided
    batch_size = args.batch_size if args.batch_size is not None else config['RRDB']['training']['batch_size']
    epochs = args.epochs if args.epochs is not None else config['RRDB']['training']['epochs']
    lr = args.lr if args.lr is not None else config['RRDB']['training']['lr']
    checkpoint = args.checkpoint if args.checkpoint is not None else config['RRDB']['training']['checkpoint']
    save_checkpoint_in = args.save_checkpoint_in if args.save_checkpoint_in is not None else config['RRDB']['training']['save_checkpoint_in']
    save_frequency = args.save_frequency if args.save_frequency is not None else config['RRDB']['training']['save_frequency']

    # Override data-related variables if command-line arguments are provided
    year = args.year if args.year is not None else config['data']['year']
    wavelength = args.wavelength if args.wavelength is not None else config['data']['wavelength']
    
    # Defining the device 
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    # Initializing the model
    generator = Generator(
        in_channels=in_channels,
        initial_channel=initial_channel,
        num_rrdb_blocks=num_rrdb_blocks,
        upscale_factor=upscale_factor
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    # Defining the optimizer
    optimizer = optim.Adam(generator.parameters(), lr=lr)

    # Loading the model from a checkpoint
    resume_epoch, loss = load_checkpoint(generator, optimizer, checkpoint)

    # Load the data from aws s3 
    data = load_single_aws_zarr(
        path_to_zarr=AWS_ZARR_ROOT + str(year),
        wavelength=wavelength
    )

    # Splitting the data into training and validation sets
    train_data = data[960:1000]
    val_data = data[:10]
    
    # Composing the transformation to be applied to the data
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Initializing the validation and training dataset and dataloader modules from PyTorch 
    downsample_factor = 1 / upscale_factor
    train_dataset = Dataset(numpy_data=train_data, downsample_factor=downsample_factor, transform=transform)
    val_dataset = Dataset(numpy_data=val_data, downsample_factor=downsample_factor, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize wandb
    wandb.init(project="your_project_name", config={
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "checkpoint": checkpoint,
        "save_checkpoint_in": save_checkpoint_in,
        "save_frequency": save_frequency,
        "year": year,
        "wavelength": wavelength,
        "in_channels": in_channels,
        "initial_channel": initial_channel,
        "num_rrdb_blocks": num_rrdb_blocks,
        "upscale_factor": upscale_factor,
    })
    # Optionally, if you want to track gradients and model parameters
    wandb.watch(generator, log='all', log_freq=5)

    # Training and validation loop
    train_losses = []
    validation_losses = []
    for epoch in range(resume_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training step
        train_loss = train(generator, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation step
        val_loss = validate(generator, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Appending validation and training loss to a list to plot them later
        train_losses.append(train_loss)
        validation_losses.append(val_loss)

        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch+1
        })

        # Saving the checkpoint 
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = save_checkpoint_in + f'/checkpoint_{year}_{wavelength}_{epoch}.pth'
            save_checkpoint(generator, optimizer, epoch, train_loss, checkpoint_path)
            
        # Log the checkpoint as an artifact
            artifact = wandb.Artifact(
            name=f'checkpoint_{year}_{wavelength}_{epoch}', 
            type='model'
        )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    # Finish wandb run
    wandb.finish()
