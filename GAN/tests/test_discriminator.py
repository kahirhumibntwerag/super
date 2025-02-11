import sys
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from thop import profile as thop_profile
import pytest
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.discriminator import DiscriminatorSRGAN

def test_discriminator_parameters():
    """Test the number of parameters and structure of the discriminator"""
    model = DiscriminatorSRGAN()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nDiscriminator Architecture:")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Basic assertions
    assert total_params > 0, "Model should have parameters"
    assert total_params == trainable_params, "All parameters should be trainable"

def test_discriminator_memory():
    """Test the memory usage of the discriminator"""
    model = DiscriminatorSRGAN()
    
    # Get model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / 1024**2
    
    print(f"\nModel Memory Usage:")
    print(f"Parameters size (MB): {size_mb:.2f}")
    
    # Test with sample input
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 512, 512)
    
    # Only profile CUDA if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True, record_shapes=True) as prof:
            with record_function("model_inference"):
                _ = model(input_tensor)
        print("\nProfiler Results (CUDA):")
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    else:
        with profile(activities=[ProfilerActivity.CPU],
                    profile_memory=True, record_shapes=True) as prof:
            with record_function("model_inference"):
                _ = model(input_tensor)
        print("\nProfiler Results (CPU only):")
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

def test_discriminator_flops():
    """Test the computational complexity (FLOPs) of the discriminator"""
    model = DiscriminatorSRGAN()
    input_tensor = torch.randn(1, 1, 512, 512)
    
    # Calculate FLOPs
    macs, params = thop_profile(model, inputs=(input_tensor,))
    
    print(f"\nComputational Complexity:")
    print(f"MACs: {macs:,}")
    print(f"FLOPs: {macs*2:,}")  # FLOPs is typically 2 * MACs
    print(f"Parameters: {params:,}")

def test_discriminator_output():
    """Test the output shape and range of the discriminator"""
    model = DiscriminatorSRGAN()
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 1, 512, 512)
        output = model(input_tensor)
        
        # Check output shape
        assert output.shape[0] == batch_size, f"Wrong batch size in output for input batch {batch_size}"
        assert output.shape[1] == 1, "Output should have 1 channel"
        
        # Check output range (should be unbounded as it's before sigmoid)
        print(f"\nOutput statistics for batch size {batch_size}:")
        print(f"Min value: {output.min().item():.4f}")
        print(f"Max value: {output.max().item():.4f}")
        print(f"Mean value: {output.mean().item():.4f}")

if __name__ == "__main__":
    print("Running Discriminator Tests...")
    
    print("\n=== Testing Parameters ===")
    test_discriminator_parameters()
    
    print("\n=== Testing Memory Usage ===")
    test_discriminator_memory()
    
    print("\n=== Testing FLOPs ===")
    test_discriminator_flops()
    
    print("\n=== Testing Output ===")
    test_discriminator_output() 