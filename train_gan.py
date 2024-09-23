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
from GAN.src.models import Generator, DiscriminatorSRGAN
from GAN.src.loss import CombinedLoss

import matplotlib.pyplot as plt
import argparse
import wandb  



def load_config(config_path):
    """loading the config file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train(generator, discriminator, dataloader, combined_criterion, disc_criterion, gen_optimizer, disc_optimizer, device):
    """
    A function that trains the SRGAN for one step and returns the average loss for both generator and discriminator.
    """
    generator.train()
    discriminator.train()

    running_gen_loss = 0.0
    running_disc_loss = 0.0

    for low_res, high_res in tqdm(dataloader, desc="Training"):
        low_res, high_res = low_res.to(device), high_res.to(device)

        # ===========================
        # Train the Discriminator
        # ===========================

        # Generate high-resolution images from low-resolution inputs
        fake_hr = generator(low_res)

        # Train discriminator on real high-resolution images
        disc_optimizer.zero_grad()
        real_labels = torch.ones(high_res.size(0), 1, device=device)
        fake_labels = torch.zeros(high_res.size(0), 1, device=device)

        # Discriminator loss on real images
        real_outputs = discriminator(high_res)
        real_loss = disc_criterion(real_outputs, real_labels)

        # Discriminator loss on fake (super-resolved) images
        fake_outputs = discriminator(fake_hr.detach())  # Detach to prevent backpropagation into the generator
        fake_loss = disc_criterion(fake_outputs, fake_labels)

        # Total discriminator loss and update
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        disc_optimizer.step()

        running_disc_loss += disc_loss.item()

        # ===========================
        # Train the Generator
        # ===========================

        # Train generator to fool the discriminator and improve perceptual quality
        gen_optimizer.zero_grad()

        # Calculate the combined loss using the CombinedLoss class
        gen_loss, mse_loss_value, adversarial_loss_value, perceptual_loss_value = combined_criterion(fake_hr, high_res)

        # Backward pass and update generator
        gen_loss.backward()
        gen_optimizer.step()

        running_gen_loss += gen_loss.item()

        # Optional: Log each loss component if needed
        # print(f"MSE Loss: {mse_loss_value.item()}, Adversarial Loss: {adversarial_loss_value.item()}, Perceptual Loss: {perceptual_loss_value.item()}")

    avg_gen_loss = running_gen_loss / len(dataloader)
    avg_disc_loss = running_disc_loss / len(dataloader)

    return avg_gen_loss, avg_disc_loss


def validate(generator, discriminator, dataloader, combined_criterion, disc_criterion, device):
    """
    A function that validates the SRGAN for one step and returns the average loss for both generator and discriminator.
    """
    generator.eval()
    discriminator.eval()

    running_gen_loss = 0.0
    running_disc_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for validation
        for low_res, high_res in tqdm(dataloader, desc="Validation"):
            low_res, high_res = low_res.to(device), high_res.to(device)

            # ===========================
            # Validate the Discriminator
            # ===========================

            # Generate high-resolution images from low-resolution inputs
            fake_hr = generator(low_res)

            # Discriminator loss on real high-resolution images
            real_labels = torch.ones(high_res.size(0), 1, device=device)
            fake_labels = torch.zeros(high_res.size(0), 1, device=device)

            # Discriminator loss on real images
            real_outputs = discriminator(high_res)
            real_loss = disc_criterion(real_outputs, real_labels)

            # Discriminator loss on fake (super-resolved) images
            fake_outputs = discriminator(fake_hr)
            fake_loss = disc_criterion(fake_outputs, fake_labels)

            # Total discriminator loss
            disc_loss = real_loss + fake_loss
            running_disc_loss += disc_loss.item()

            # ===========================
            # Validate the Generator
            # ===========================

            # Calculate the combined loss using the CombinedLoss class
            gen_loss, mse_loss_value, adversarial_loss_value, perceptual_loss_value = combined_criterion(fake_hr, high_res)

            running_gen_loss += gen_loss.item()

    avg_gen_loss = running_gen_loss / len(dataloader)
    avg_disc_loss = running_disc_loss / len(dataloader)

    return avg_gen_loss, avg_disc_loss


def save_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer,epoch, loss, file_path='checkpoint.pth'):
    """
    Save a model checkpoint.

    Args:
        generator (torch.nn.Module): The generator to save.
        discriminator (torch.nn.Nodule): the discriminator to save
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The current loss.
        file_path (str): The path where the checkpoint will be saved.
    """
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict':discriminator.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),

        'loss': loss
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at {file_path}")



def load_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer,custom_checkpoint_path):
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
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
    disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
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
    parser.add_argument("--gen_lr", type=float, help="Override learning rate from config")
    parser.add_argument("--disc_lr", type=float, help="Override learning rate from config")
    parser.add_argument("--checkpoint", type=str, help="Override checkpoint path from config")
    parser.add_argument("--save_checkpoint_in", type=str, help="Override checkpoint save directory from config")
    parser.add_argument("--save_frequency", type=int, help="Override save frequency from config")
    parser.add_argument("--year", type=int, help="Override year from config")
    parser.add_argument("--wavelength", type=int, help="Override wavelength from config")
    parser.add_argument("--disc_num_layers", type=int, help="Override wavelength from config")
    parser.add_argument("--disc_channels", type=int, help="Override wavelength from config")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    # Load the configuration from the YAML file
    config = load_config(r'super\config\config.yml')

    # Loading and initializing variables

    # generator initialization arguments
    in_channels = config['gan']['generator']['in_channels']
    initial_channel = config['gan']['generator']['initial_channel']
    num_rrdb_blocks = config['gan']['generator']['num_rrdb_blocks']
    upscale_factor = config['gan']['generator']['upscale_factor']
    #Discriminator initialization arguments
    disc_num_layers = config['gan']['discriminator']['num_layers']
    disc_channels = config['gan']['discriminator']['channels']


    
    # Override training-related variables if command-line arguments are provided
    batch_size = args.batch_size if args.batch_size is not None else config['gan']['training']['batch_size']
    epochs = args.epochs if args.epochs is not None else config['gan']['training']['epochs']
    gen_lr = args.gen_lr if args.gen_lr is not None else config['gan']['generator']['lr']
    disc_lr = args.disc_lr if args.disc_lr is not None else config['gan']['discriminator']['lr']
    checkpoint = args.checkpoint if args.checkpoint is not None else config['gan']['training']['checkpoint']
    save_checkpoint_in = args.save_checkpoint_in if args.save_checkpoint_in is not None else config['gan']['training']['save_checkpoint_in']
    save_frequency = args.save_frequency if args.save_frequency is not None else config['gan']['training']['save_frequency']

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

    discriminator = DiscriminatorSRGAN(
        disc_channels=disc_channels,
        disc_num_layers=disc_num_layers
    ).to(device)

    # Define loss function and optimizer
    combined_criterion = CombinedLoss(discriminator)

    # Defining the generator optimizer
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr)

    # Loading the model from a checkpoint
    resume_epoch, loss = load_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, checkpoint)

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
        "gen_lr": gen_lr,
        "disc_lr": disc_lr,
        "checkpoint": checkpoint,
        "save_checkpoint_in": save_checkpoint_in,
        "save_frequency": save_frequency,
        "year": year,
        "wavelength": wavelength,
        "in_channels": in_channels,
        "initial_channel": initial_channel,
        "num_rrdb_blocks": num_rrdb_blocks,
        "upscale_factor": upscale_factor,
        "disc_channels": disc_channels,
        "disc_num_layers": disc_num_layers

    })
    # Optionally, if you want to track gradients and model parameters
    wandb.watch(generator, log='all', log_freq=5)
    wandb.watch(discriminator, log='all', log_freq=5)


train_gen_losses = []
train_disc_losses = []
val_gen_losses = []
val_disc_losses = []

for epoch in range(resume_epoch, epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Training step
    avg_gen_loss, avg_disc_loss = train(
        generator, 
        discriminator, 
        train_loader, 
        combined_criterion, 
        nn.BCEWithLogitsLoss(), 
        gen_optimizer, 
        disc_optimizer, 
        device
    )
    print(f"Training Generator Loss: {avg_gen_loss:.4f}, Training Discriminator Loss: {avg_disc_loss:.4f}")

    # Validation step
    avg_gen_valloss, avg_disc_valloss = validate(
        generator, 
        discriminator, 
        val_loader, 
        combined_criterion, 
        nn.BCEWithLogitsLoss(), 
        device
    )
    print(f"Validation Generator Loss: {avg_gen_valloss:.4f}, Validation Discriminator Loss: {avg_disc_valloss:.4f}")

    # Appending validation and training losses to lists
    train_gen_losses.append(avg_gen_loss)
    train_disc_losses.append(avg_disc_loss)
    val_gen_losses.append(avg_gen_valloss)
    val_disc_losses.append(avg_disc_valloss)

    # Log metrics to wandb
    wandb.log({
        "train_gen_loss": avg_gen_loss,
        "train_disc_loss": avg_disc_loss,
        "val_gen_loss": avg_gen_valloss,
        "val_disc_loss": avg_disc_valloss,
        "epoch": epoch + 1
    })

    # Save checkpoint if it's time according to the save frequency
    if (epoch + 1) % save_frequency == 0:
        checkpoint_path = save_checkpoint_in + f'/checkpoint_{year}_{wavelength}_{epoch}.pth'
        save_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, epoch, avg_gen_loss, checkpoint_path)

        # Log the checkpoint as an artifact in wandb
        artifact = wandb.Artifact(
            name=f'checkpoint_{year}_{wavelength}_{epoch}',
            type='model'
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

# Finish the wandb run
wandb.finish()
