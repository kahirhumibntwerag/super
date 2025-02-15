import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from src.metrics import kld_loss
import numpy as np

def calculate_metrics(sr_batch, hr_batch):
    """
    Calculate metrics for a batch of images.
    
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

def process_dataset(model_path, data_path, device='cuda', batch_size=4):
    """
    Process dataset and collect metrics.
    
    Args:
        model_path: Path to saved model checkpoint
        data_path: Path to file containing HR-LR pairs
        device: Device to use for processing
        batch_size: Batch size for processing
    """
    # Load generator only from checkpoint
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, weights_only=True)  # Add weights_only=True
    model = checkpoint['state_dict']
    
    # Extract only generator weights
    generator_weights = {k.replace('generator.', ''): v for k, v in model.items() 
                       if k.startswith('generator.')}
    
    # Initialize and load generator
    from src.RRDB import Generator
    model = Generator().to(device)
    model.load_state_dict(generator_weights)
    model.eval()
    
    # Load data
    print(f"Loading data from {data_path}")
    data = torch.load(data_path)
    lr_images = data['lr'].to(device)
    hr_images = data['hr'].to(device)
    
    num_images = len(lr_images)
    metrics_dict = {'PSNR': [], 'SSIM': [], 'KLD': []}
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_end = min(i + batch_size, num_images)
            print(f"Processing images {i} to {batch_end-1}")
            
            # Get batch
            lr_batch = lr_images[i:batch_end]
            hr_batch = hr_images[i:batch_end]
            
            # Generate SR images
            sr_batch = model(lr_batch.unsqueeze(1))  # Add channel dim for model input
            
            # Ensure correct dimensions for metrics calculation
            sr_batch = sr_batch.squeeze(1)  # Remove channel dim if present
            hr_batch = hr_batch.squeeze(1)  # Remove channel dim if present
            
            # Calculate metrics for each image in batch
            for sr, hr in zip(sr_batch, hr_batch):
                # Add necessary dimensions for metrics
                sr = sr.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                hr = hr.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                psnr = peak_signal_noise_ratio(sr, hr)
                ssim = structural_similarity_index_measure(sr, hr)
                kld = kld_loss(hr, sr)
                
                metrics_dict['PSNR'].append(psnr.item())
                metrics_dict['SSIM'].append(ssim.item())
                metrics_dict['KLD'].append(kld)
            
            # Print batch metrics
            batch_psnr = np.mean(metrics_dict['PSNR'][-batch_size:])
            batch_ssim = np.mean(metrics_dict['SSIM'][-batch_size:])
            batch_kld = np.mean(metrics_dict['KLD'][-batch_size:])
            print(f"Batch metrics: PSNR={batch_psnr:.3f}, SSIM={batch_ssim:.3f}, KLD={batch_kld:.3f}")
    
    # Calculate and print final metrics
    final_metrics = {
        name: (np.mean(values), np.std(values)) 
        for name, values in metrics_dict.items()
    }
    
    print("\nFinal Metrics:")
    for name, (mean, std) in final_metrics.items():
        print(f"{name}: {mean:.3f} ± {std:.3f}")
    
    return metrics_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/gan-epoch=68-val_loss=0.70.ckpt')
    parser.add_argument('--data', type=str, default='src/data.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    process_dataset(args.model, args.data, args.device, args.batch_size)