import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from src.inference import (
    power_transform,
    load_model,
    calculate_metrics,
    plot_metrics,
    process_dataset
)

@pytest.fixture
def sample_image():
    return torch.rand(1, 1, 64, 64)

@pytest.fixture
def sample_config():
    return {
        'project_name': 'test-project',
        'run_name': 'test-run',
        'generator': {
            'in_channels': 1,
            'out_channels': 1,
            'num_features': 64
        }
    }

@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock(return_value=None)
    model.to = Mock(return_value=model)
    
    # Create a proper tensor output for the forward pass
    def forward_mock(x):
        # Return a tensor with the same batch size but 2x dimensions
        batch_size = x.shape[0]
        h, w = x.shape[-2:]
        return torch.rand(batch_size, 1, h*2, w*2)
    
    model.forward = forward_mock
    model.__call__ = forward_mock  # Important for Mock objects
    return model

class TestPowerTransform:
    def test_power_transform_basic(self):
        x = torch.ones(1, 1, 64, 64)
        result = power_transform(x)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)
        assert not torch.isnan(result).any()
    
    def test_power_transform_zero_lambda(self):
        x = torch.ones(1, 1, 64, 64)
        result = power_transform(x, lambda_param=0)
        assert not torch.isnan(result).any()
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)
    
    def test_power_transform_negative_values(self):
        x = torch.randn(1, 1, 64, 64)  # Contains negative values
        result = power_transform(x)
        assert not torch.isnan(result).any()
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)
    
    def test_power_transform_constant_input(self):
        x = torch.zeros(1, 1, 64, 64)
        result = power_transform(x)
        assert not torch.isnan(result).any()
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

    def test_power_transform_small_values(self):
        x = torch.ones(1, 1, 64, 64) * 1e-10
        result = power_transform(x)
        assert not torch.isnan(result).any()
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

class TestLoadModel:
    def test_load_model(self, sample_config, tmp_path):
        # Create temporary config file
        config_path = tmp_path / "config.yml"
        with open(config_path, "w") as f:
            import yaml
            yaml.dump(sample_config, f)
        
        # Create mock checkpoint
        checkpoint = {'state_dict': {'key': 'value'}}
        checkpoint_path = tmp_path / "model.ckpt"
        torch.save(checkpoint, checkpoint_path)
        
        with patch('src.inference.LightningGenerator') as mock_generator:
            mock_model = Mock()
            mock_generator.return_value = mock_model
            
            model = load_model(config_path, checkpoint_path)
            
            mock_generator.assert_called_once()
            mock_model.load_state_dict.assert_called_once_with(checkpoint['state_dict'])
            mock_model.eval.assert_called_once()

class TestCalculateMetrics:
    def test_calculate_metrics(self, sample_image):
        sr = sample_image
        hr = sample_image  # Perfect reconstruction for testing
        
        with patch('src.inference.peak_signal_noise_ratio') as mock_psnr, \
             patch('src.inference.structural_similarity_index_measure') as mock_ssim, \
             patch('src.inference.kld_loss') as mock_kld:
            
            mock_psnr.return_value = torch.tensor(30.0)
            mock_ssim.return_value = torch.tensor(0.9)
            mock_kld.return_value = 0.1
            
            metrics = calculate_metrics(sr, hr)
            
            assert isinstance(metrics, dict)
            assert all(k in metrics for k in ['PSNR', 'SSIM', 'KLD'])
            assert all(isinstance(v, float) for v in metrics.values())

class TestPlotMetrics:
    def test_plot_metrics_basic(self):
        metrics_dict = {
            'PSNR': [30.0, 31.0, 32.0],
            'SSIM': [0.9, 0.91, 0.92]
        }
        
        labels = {'model1': 'Test Model'}
        
        fig = plot_metrics(metrics_dict, **labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_metrics_validation(self):
        with pytest.raises(ValueError):
            plot_metrics()  # No metrics provided
        
        with pytest.raises(ValueError):
            plot_metrics({'PSNR': [30.0]})  # No labels provided
        
        with pytest.raises(ValueError):
            # Mismatched number of metrics and labels
            plot_metrics({'PSNR': [30.0]}, {'model1': 'Model 1', 'model2': 'Model 2'})

class TestProcessDataset:
    def test_process_dataset(self, mock_model, tmp_path):
        # Create test data
        lr_dir = tmp_path / "lr"
        hr_dir = tmp_path / "hr"
        lr_dir.mkdir()
        hr_dir.mkdir()
        
        # Save test tensors with proper dimensions
        for i in range(3):
            # Create tensors with proper dimensions
            lr_tensor = torch.rand(1, 64, 64)  # LR image
            hr_tensor = torch.rand(1, 128, 128)  # HR image (2x size)
            
            # Save tensors without weights_only parameter
            torch.save(lr_tensor, lr_dir / f"img_{i}.pt")
            torch.save(hr_tensor, hr_dir / f"img_{i}.pt")
        
        metrics = process_dataset(mock_model, lr_dir, hr_dir, device='cpu')
        
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['PSNR', 'SSIM', 'KLD'])
        assert all(isinstance(v, list) for v in metrics.values())
        assert all(len(v) == 3 for v in metrics.values())

if __name__ == '__main__':
    pytest.main([__file__]) 