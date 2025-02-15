import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from src.metrics import kld_loss
from src.RRDB import LightningGAN
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def rescalee(images):
    """Rescale tensor using log normalization"""
    images_clipped = torch.clamp(images, min=1)
    images_log = torch.log(images_clipped)
    max_value = torch.log(torch.tensor(20000))
    max_value = torch.clamp(max_value, min=1e-9)
    images_normalized = images_log / max_value
    return images_normalized

def inverse_rescalee(images_normalized):
    """Inverse rescale from normalized to original range"""
    max_value = torch.log(torch.tensor(20000.0))
    max_value = torch.clamp(max_value, min=1e-9)
    images_log = images_normalized * max_value
    images_clipped = torch.exp(images_log)
    return images_clipped

def load_model(config_path, checkpoint_path):
    """Load only the generator from the trained model checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = LightningGAN(config)
    
    # Load checkpoint with weights_only=True
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    # Filter state dict to only include generator weights
    state_dict = checkpoint['state_dict']
    generator_state_dict = {
        k: v for k, v in state_dict.items() 
        if k.startswith('generator.')
    }
    
    # Remove 'generator.' prefix from keys
    generator_state_dict = {
        k.replace('generator.', ''): v 
        for k, v in generator_state_dict.items()
    }
    
    # Load only generator weights
    model.generator.load_state_dict(generator_state_dict)
    model.eval()
    
    # Return only the generator
    return model.generator

def calculate_metrics(pred, target):
    """Calculate PSNR, SSIM and KLD metrics."""
    # Ensure tensors have same dimensions
    if target.dim() == 5:
        target = target.squeeze(2)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        
    try:
        psnr_val = psnr(pred, target)
        ssim_val = ssim(pred, target)
        kld_val = kld_loss(target, pred)
        return psnr_val.item(), ssim_val.item(), kld_val
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return 0.0, 0.0, 0.0

def process_batch(model, batch, device, batch_idx, start_idx):
    """Process a single batch of images."""
    lr, hr = batch['lr'], batch['hr']
    
    # Move to device and ensure correct dimensions
    lr = lr.to(device)
    hr = hr.to(device)
    
    # Add channel dimension if missing
    if lr.dim() == 3:
        lr = lr.unsqueeze(1)  # [B, H, W] -> [B, C, H, W]
    if hr.dim() == 3:
        hr = hr.unsqueeze(1)
    
    # Remove extra dimension if present
    if hr.dim() == 5:
        hr = hr.squeeze(2)
    
    # Ensure single channel
    if lr.size(1) > 1:
        lr = lr[:, 0:1, :, :]  # Keep only first channel
    
    with torch.no_grad():
        sr = model(lr)
        
        # Calculate metrics including KLD
        psnr_val, ssim_val, kld_val = calculate_metrics(sr, hr)
        
        # Save sample images periodically
        if batch_idx % 10 == 0:
            save_sample_images(lr, sr, hr, batch_idx, start_idx)
            
    return psnr_val, ssim_val, kld_val

def process_dataset(model, data_path, device, batch_size=4):
    """Process entire dataset."""
    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
    lr_data, hr_data = data['lr'], data['hr']
    
    dataloader = DataLoader(
        TensorDataset(lr_data, hr_data),
        batch_size=batch_size,
        shuffle=False
    )
    
    total_psnr = 0
    total_ssim = 0
    total_kld = 0
    num_batches = 0
    
    print("Processing images...")
    for i, batch in enumerate(tqdm(dataloader)):
        batch = {
            'lr': batch[0],
            'hr': batch[1]
        }
        
        # Process batch
        batch_psnr, batch_ssim, batch_kld = process_batch(model, batch, device, num_batches, i)
        
        print(f"Batch metrics: PSNR={batch_psnr:.3f}, SSIM={batch_ssim:.3f}, KLD={batch_kld:.3f}")
        
        total_psnr += batch_psnr
        total_ssim += batch_ssim
        total_kld += batch_kld
        num_batches += 1
    
    # Calculate final metrics
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_kld = total_kld / num_batches
    
    print("\nFinal Average Metrics:")
    print(f"PSNR: {avg_psnr:.3f}")
    print(f"SSIM: {avg_ssim:.3f}")
    print(f"KLD: {avg_kld:.3f}")
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'kld': avg_kld
    }

def save_sample_images(lr, sr, hr, batch_idx, start_idx):
    """Save sample images for comparison."""
    output_dir = Path('results/samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to numpy arrays
    lr_img = lr[0, 0].cpu().numpy()
    sr_img = sr[0, 0].cpu().numpy()
    hr_img = hr[0, 0].cpu().numpy()
    
    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr_img, cmap='gray')
    axes[0].set_title('Low Resolution')
    axes[1].imshow(sr_img, cmap='gray')
    axes[1].set_title('Super Resolution')
    axes[2].imshow(hr_img, cmap='gray')
    axes[2].set_title('High Resolution')
    
    plt.savefig(output_dir / f'sample_{start_idx + batch_idx}.png')
    plt.close()

def main():
    # Paths
    config_path = 'src/config.yaml'
    checkpoint_path = 'checkpoints/last.ckpt'
    data_path = 'src/data.pt'
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(config_path, checkpoint_path)
    model = model.to(device)
    model.eval()
    
    # Process dataset
    metrics = process_dataset(model, data_path, device, batch_size=4)
    
    # Save metrics
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f'{metric_name}: {value:.4f}\n')
    
    # Plot distributions
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for ax, (metric_name, value) in zip(axes, metrics.items()):
        sns.histplot(value, ax=ax, kde=True)
        ax.set_title(f'Distribution of {metric_name}')
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_distribution.png')
    plt.close(fig)

if __name__ == '__main__':
    main() 