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

def power_transform(images, lambda_param=0.5):
    """
    Apply power transform to images, ensuring positive output.
    
    Args:
        images: Input tensor
        lambda_param: Power transform parameter (default=0.5 for square root)
    Returns:
        Normalized tensor in range [0, 1]
    """
    # Ensure positive values and avoid division by zero
    eps = 1e-8
    images = torch.clamp(images, min=eps)
    
    # Apply power transform
    if lambda_param == 0:
        transformed = torch.log(images + eps)
    else:
        transformed = ((images + eps) ** lambda_param - 1) / lambda_param
    
    # Normalize to [0, 1] range
    min_val = transformed.min()
    max_val = transformed.max()
    
    # Handle the case where all values are the same
    if max_val == min_val:
        return torch.ones_like(transformed)
    
    normalized = (transformed - min_val) / (max_val - min_val)
    return normalized

def load_model(config_path, checkpoint_path):
    """Same as before"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = LightningGenerator(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model

def calculate_metrics(sr_image, hr_image):
    """
    Calculate various image quality metrics.
    
    Args:
        sr_image: Super-resolved image tensor (single channel)
        hr_image: High-resolution ground truth tensor (single channel)
    
    Returns:
        dict: Dictionary containing metrics
    """
    # Ensure images are in the right format
    sr = sr_image.squeeze()
    hr = hr_image.squeeze()
    
    # Normalize images to [0, 1] range if needed
    if sr.max() > 1 or sr.min() < 0:
        sr = (sr - sr.min()) / (sr.max() - sr.min())
    if hr.max() > 1 or hr.min() < 0:
        hr = (hr - hr.min()) / (hr.max() - hr.min())
    
    # Calculate metrics
    psnr = peak_signal_noise_ratio(sr, hr)
    ssim = structural_similarity_index_measure(sr.unsqueeze(0), hr.unsqueeze(0))
    kld = kld_loss(hr, sr)
    
    return {
        'PSNR': psnr.item(),
        'SSIM': ssim.item(),
        'KLD': kld
    }

def plot_metrics(*metrics_dicts, **labels):
    """Plot metrics distributions."""
    if len(metrics_dicts) == 0:
        raise ValueError("At least one metrics dictionary is required.")

    if len(labels) != len(metrics_dicts):
        raise ValueError("The number of labels must match the number of metrics dictionaries.")

    model_labels = list(labels.values())
    metric_keys = set(metrics_dicts[0].keys())
    
    if not all(set(d.keys()) == metric_keys for d in metrics_dicts):
        raise ValueError("All metrics dictionaries must have the same keys.")

    num_metrics = len(metric_keys)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    if num_metrics == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metric_keys):
        for i, metrics_dict in enumerate(metrics_dicts):
            metric_values = metrics_dict[metric_name]
            sns.kdeplot(metric_values, ax=ax, label=model_labels[i], fill=True)

        ax.set_title(f'Distribution of {metric_name}')
        ax.legend()

    plt.tight_layout()
    return fig

def process_dataset(model, lr_dir, hr_dir, device='cuda'):
    """Process multiple images and collect metrics."""
    metrics_dict = {
        'PSNR': [],    # Higher is better (typical range: 20-50 dB)
        'SSIM': [],    # Higher is better (range: 0-1)
        'KLD': []      # Lower is better
    }
    
    lr_paths = sorted(Path(lr_dir).glob('*.pt'))
    hr_paths = sorted(Path(hr_dir).glob('*.pt'))
    
    for lr_path, hr_path in zip(lr_paths, hr_paths):
        try:
            # Load images
            lr = torch.load(lr_path)
            hr = torch.load(hr_path)
            
            # Generate SR image
            lr_transformed = power_transform(lr.unsqueeze(0))
            lr_transformed = lr_transformed.to(device)
            
            with torch.no_grad():
                sr = model(lr_transformed)
            
            sr = sr.cpu()
            
            # Calculate metrics
            metrics = calculate_metrics(sr, hr)
            
            # Collect metrics
            for metric_name, value in metrics.items():
                metrics_dict[metric_name].append(value)
            
            # Print progress
            print(f"Processed {lr_path.name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}={value:.3f}", end=', ')
            print()
            
        except Exception as e:
            print(f"Error processing {lr_path.name}: {e}")
            continue
    
    # Calculate and print average metrics
    print("\nAverage Metrics:")
    for metric_name in metrics_dict:
        values = metrics_dict[metric_name]
        if values:  # Only calculate if we have valid values
            mean = np.mean(values)
            std = np.std(values)
            print(f"{metric_name}: {mean:.3f} ± {std:.3f}")
    
    return metrics_dict

def main():
    # Paths
    config_path = 'src/config.yml'
    checkpoint_paths = [
        'checkpoints/model1.ckpt',
        'checkpoints/model2.ckpt',
        # Add more checkpoint paths as needed
    ]
    lr_dir = 'path/to/lr_images'
    hr_dir = 'path/to/hr_images'
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Process each model
    all_metrics = []
    model_names = []
    
    for i, ckpt_path in enumerate(checkpoint_paths):
        # Load model
        model = load_model(config_path, ckpt_path)
        model = model.to(device)
        
        # Process dataset and collect metrics
        metrics_dict = process_dataset(model, lr_dir, hr_dir, device)
        all_metrics.append(metrics_dict)
        model_names.append(f'Model {i+1}')
    
    # Plot metrics
    labels = {f'model{i+1}': name for i, name in enumerate(model_names)}
    fig = plot_metrics(*all_metrics, **labels)
    
    # Save plot
    fig.savefig(output_dir / 'metrics_comparison.png')
    plt.close(fig)
    
    # Save metrics to file
    for i, metrics in enumerate(all_metrics):
        metrics_path = output_dir / f'metrics_model{i+1}.txt'
        with open(metrics_path, 'w') as f:
            for metric_name, values in metrics.items():
                mean = np.mean(values)
                std = np.std(values)
                f.write(f'{metric_name}: {mean:.4f} ± {std:.4f}\n')

if __name__ == '__main__':
    main() 