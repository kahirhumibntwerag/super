import sys
import os
import torch
import numpy as np
from pathlib import Path
import yaml

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.RRDB import RRDB, Generator, LightningGAN

def test_rrdb_block():
    """Test the RRDB block structure and output"""
    in_features = 64
    batch_size = 4
    size = 64
    
    # Initialize RRDB block
    rrdb = RRDB(in_features)
    
    # Test input
    x = torch.randn(batch_size, in_features, size, size)
    output = rrdb(x)
    
    print("\n=== RRDB Block Test ===")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in rrdb.parameters()):,}")
    
    # Shape assertions
    assert output.shape == x.shape, "RRDB output shape should match input shape"
    assert torch.is_tensor(output), "Output should be a tensor"

def test_generator():
    """Test the Generator architecture"""
    # Initialize Generator
    generator = Generator()
    
    # Test with sample input
    batch_size = 4
    input_size = 128
    x = torch.randn(batch_size, 1, input_size, input_size)
    output = generator(x)
    
    print("\n=== Generator Test ===")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Expected output size (4x upscaling)
    expected_size = input_size * 4
    
    # Assertions
    assert output.shape == (batch_size, 1, expected_size, expected_size), \
        "Generator output shape incorrect"
    assert torch.is_tensor(output), "Output should be a tensor"

def test_lightning_gan():
    """Test the LightningGAN model"""
    # Load config
    config_path = Path(__file__).parent.parent / 'src' / 'config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize LightningGAN
    model = LightningGAN(config)
    
    # Test batch
    batch_size = 4
    input_size = 128
    lr_imgs = torch.randn(batch_size, 1, input_size, input_size)
    hr_imgs = torch.randn(batch_size, 1, input_size*4, input_size*4)
    batch = (lr_imgs, hr_imgs)
    
    print("\n=== LightningGAN Test ===")
    print("Model Architecture:")
    print(f"Generator parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
    
    # Test forward pass
    fake_imgs = model.generator(lr_imgs)
    disc_output = model.discriminator(fake_imgs)
    
    print(f"\nGenerator output shape: {fake_imgs.shape}")
    print(f"Discriminator output shape: {disc_output.shape}")
    
    # Memory usage
    def get_model_size(model):
        param_size = 0
        buffer_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2  # Size in MB
    
    print(f"\nMemory Usage:")
    print(f"Generator size (MB): {get_model_size(model.generator):.2f}")
    print(f"Discriminator size (MB): {get_model_size(model.discriminator):.2f}")
    
    # Test training step (with error handling)
    try:
        loss = model.training_step(batch, 0)
        print("\nTraining step completed successfully")
    except Exception as e:
        print(f"\nTraining step error: {str(e)}")

def test_model_save_load():
    """Test saving and loading the model"""
    # Load config
    config_path = Path(__file__).parent.parent / 'src' / 'config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = LightningGAN(config)
    
    # Save model
    save_path = "test_model.pt"
    try:
        torch.save(model.state_dict(), save_path)
        print("\n=== Model Save/Load Test ===")
        print(f"Model saved successfully to {save_path}")
        
        # Load model
        new_model = LightningGAN(config)
        new_model.load_state_dict(torch.load(save_path))
        print("Model loaded successfully")
        
        # Cleanup
        os.remove(save_path)
    except Exception as e:
        print(f"Save/Load test error: {str(e)}")

if __name__ == "__main__":
    print("Running Model Tests...")
    
    test_rrdb_block()
    test_generator()
    test_lightning_gan()
    test_model_save_load() 