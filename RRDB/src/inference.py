import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from src.metrics import kld_loss
from src.RRDB import LightningGenerator
import numpy as np
import torch.nn.functional as F

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
    """Load the trained model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = LightningGenerator(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model

def calculate_metrics(sr_batch, hr_batch):
    """
    Calculate metrics for batches of images.
    
    Args:
        sr_batch: Super-resolved images tensor [B, H, W]
        hr_batch: High-resolution ground truth tensor [B, H, W]
    
    Returns:
        dict: Dictionary containing lists of metrics for each image
    """
    metrics_dict = {
        'PSNR': [],
        'SSIM': [],
        'KLD': []
    }
    
    # Process each image in the batch
    for sr, hr in zip(sr_batch, hr_batch):
        # Add channel dimension for all metrics (BxCxHxW format)
        sr_formatted = sr.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        hr_formatted = hr.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Calculate metrics
        psnr = peak_signal_noise_ratio(sr_formatted, hr_formatted)
        ssim = structural_similarity_index_measure(sr_formatted, hr_formatted)
        
        # For KLD, ensure tensors are in [C, H, W] format
        sr_kld = sr_formatted.squeeze(0)  # [1, H, W]
        hr_kld = hr_formatted.squeeze(0)  # [1, H, W]
        kld = kld_loss(hr_kld, sr_kld)
        
        # Store metrics
        metrics_dict['PSNR'].append(psnr.item())
        metrics_dict['SSIM'].append(ssim.item())
        metrics_dict['KLD'].append(kld)
    
    return metrics_dict

def process_dataset(model, data_path, device='cuda', batch_size=4):
    """
    Process dataset and collect metrics.
    
    Args:
        model: Trained model
        data_path: Path to file containing HR-LR pairs
        device: Device to use for processing
        batch_size: Batch size for processing
    """
    # Load data with weights_only=True to address warning
    print(f"Loading data from {data_path}")
    data = torch.load(data_path, weights_only=True)
    lr_images = data['lr']  # [1000, 128, 128]
    hr_images = data['hr']  # [1000, 512, 512]
    
    num_images = lr_images.shape[0]
    metrics_dict = {
        'PSNR': [],
        'SSIM': [],
        'KLD': []
    }
    
    # Process in batches
    for i in range(0, num_images, batch_size):
        batch_end = min(i + batch_size, num_images)
        print(f"Processing images {i} to {batch_end-1}")
        
        try:
            # Get batch
            lr_batch = lr_images[i:batch_end].to(device)
            hr_batch = hr_images[i:batch_end].to(device)
            
            # Add channel dimension and apply rescalee for model input
            lr_batch = lr_batch.unsqueeze(1)  # [B, 1, 128, 128]
            lr_batch = rescalee(lr_batch)
            
            # Generate SR images
            with torch.no_grad():
                sr_batch = model(lr_batch)
            
            # Inverse rescale SR images back to original range
            sr_batch = inverse_rescalee(sr_batch)
            
            # Ensure values are in valid range
            sr_batch = torch.clamp(sr_batch, min=0.0)
            
            # Move to CPU for metric calculation
            sr_batch = sr_batch.cpu()
            hr_batch = hr_batch.cpu()
            
            # Calculate metrics
            batch_metrics = calculate_metrics(sr_batch.squeeze(1), hr_batch)
            
            # Check for invalid metrics
            for metric_name, values in batch_metrics.items():
                valid_values = [v for v in values if not torch.isnan(torch.tensor(v)) and not torch.isinf(torch.tensor(v))]
                if valid_values:
                    metrics_dict[metric_name].extend(valid_values)
            
            # Print progress for this batch
            batch_means = {k: np.mean(metrics_dict[k][-len(valid_values):]) 
                         for k in metrics_dict if metrics_dict[k]}
            print(f"Batch metrics: ", end='')
            for metric_name, value in batch_means.items():
                print(f"{metric_name}={value:.3f}", end=', ')
            print()
            
        except Exception as e:
            print(f"Error processing batch {i} to {batch_end-1}: {str(e)}")
            continue
    
    # Calculate and print final averages (excluding NaN and inf values)
    print("\nFinal Average Metrics:")
    for metric_name in metrics_dict:
        values = metrics_dict[metric_name]
        valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
        if valid_values:
            mean = np.mean(valid_values)
            std = np.std(valid_values)
            print(f"{metric_name}: {mean:.3f} ± {std:.3f}")
        else:
            print(f"{metric_name}: No valid values")
    
    return metrics_dict

def main():
    # Paths
    config_path = 'src/config.yml'
    checkpoint_path = 'checkpoints/last.ckpt'
    data_path = 'src/data.pt'  # File containing HR-LR pairs
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(config_path, checkpoint_path)
    model = model.to(device)
    
    # Process dataset
    metrics = process_dataset(model, data_path, device, batch_size=4)
    
    # Save metrics
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        for metric_name, values in metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            f.write(f'{metric_name}: {mean:.4f} ± {std:.4f}\n')
    
    # Plot distributions
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        sns.histplot(values, ax=ax, kde=True)
        ax.set_title(f'Distribution of {metric_name}')
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_distribution.png')
    plt.close(fig)

if __name__ == '__main__':
    main() 