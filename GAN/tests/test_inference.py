import sys
import os
import torch
import numpy as np
from pathlib import Path
import pytest

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Changed to include parent of GAN

from metrics import kld_loss  # Import from the root metrics.py
from inference import rescalee, inverse_rescalee, calculate_metrics

def test_rescalee():
    """Test the rescalee normalization function"""
    # Test with known values
    test_tensor = torch.tensor([1.0, 100.0, 1000.0, 20000.0])
    
    # Apply normalization
    normalized = rescalee(test_tensor)
    
    print("\n=== Rescalee Test ===")
    print(f"Original values: {test_tensor}")
    print(f"Normalized values: {normalized}")
    
    # Test basic properties
    assert torch.all(normalized >= 0), "Normalized values should be non-negative"
    assert torch.all(normalized <= 1), "Normalized values should be <= 1"
    assert torch.isfinite(normalized).all(), "All values should be finite"
    
    # Test inverse transform
    reconstructed = inverse_rescalee(normalized)
    print(f"Reconstructed values: {reconstructed}")
    
    # Check reconstruction (with tolerance for floating point)
    assert torch.allclose(test_tensor, reconstructed, rtol=1e-3), \
        "Inverse transform should reconstruct original values"

def test_calculate_metrics():
    """Test the metric calculation function"""
    # Create synthetic test images
    size = 512  # Changed to match expected size in metrics.py
    batch_size = 2
    
    # Create identical images first (perfect reconstruction case)
    perfect_sr = torch.ones(batch_size, size, size)
    perfect_hr = torch.ones(batch_size, size, size)
    
    # Calculate metrics for perfect case
    perfect_metrics = calculate_metrics(perfect_sr, perfect_hr)
    
    print("\n=== Perfect Reconstruction Metrics ===")
    for metric_name, values in perfect_metrics.items():
        print(f"{metric_name}: {np.mean(values):.4f}")
    
    # Create different images (imperfect reconstruction case)
    imperfect_sr = torch.rand(batch_size, size, size)
    imperfect_hr = torch.rand(batch_size, size, size)
    
    # Calculate metrics for imperfect case
    imperfect_metrics = calculate_metrics(imperfect_sr, imperfect_hr)
    
    print("\n=== Random Images Metrics ===")
    for metric_name, values in imperfect_metrics.items():
        print(f"{metric_name}: {np.mean(values):.4f}")
    
    # Basic assertions
    assert all(len(values) == batch_size for values in perfect_metrics.values()), \
        "Should have metrics for each image in batch"
    
    # Perfect reconstruction should have better metrics
    assert np.mean(perfect_metrics['PSNR']) > np.mean(imperfect_metrics['PSNR']), \
        "PSNR should be higher for perfect reconstruction"
    assert np.mean(perfect_metrics['SSIM']) > np.mean(imperfect_metrics['SSIM']), \
        "SSIM should be higher for perfect reconstruction"

def test_metric_ranges():
    """Test that metrics stay within expected ranges"""
    size = 512  # Changed to match expected size in metrics.py
    batch_size = 2
    
    # Test with different types of inputs
    test_cases = [
        ("Identical", torch.ones(batch_size, size, size), torch.ones(batch_size, size, size)),
        ("Different", torch.zeros(batch_size, size, size), torch.ones(batch_size, size, size)),
        ("Random", torch.rand(batch_size, size, size), torch.rand(batch_size, size, size))
    ]
    
    print("\n=== Metric Range Tests ===")
    for name, sr, hr in test_cases:
        metrics = calculate_metrics(sr, hr)
        print(f"\n{name} images:")
        for metric_name, values in metrics.items():
            mean_value = np.mean(values)
            print(f"{metric_name}: {mean_value:.4f}")
            
            # Check ranges
            if metric_name == 'SSIM':
                assert all(0 <= v <= 1 for v in values), f"SSIM should be in [0,1], got {values}"
            elif metric_name == 'PSNR':
                assert all(v >= 0 for v in values), f"PSNR should be non-negative, got {values}"

if __name__ == "__main__":
    print("Running Inference Tests...")
    
    test_rescalee()
    test_calculate_metrics()
    test_metric_ranges() 