import pytest
import torch
import yaml
import os
from unittest.mock import MagicMock, patch
from super.RRDB.train import train

@pytest.fixture
def sample_config():
    return {
        'project_name': 'test-project',
        'run_name': 'test-run',
        'trainer': {
            'max_epochs': 1,
            'checkpoint_dir': 'test_checkpoints',
            'accelerator': 'cpu',
            'devices': 1
        },
        'generator': {
            'in_channels': 1,
            'initial_channel': 64,
            'num_rrdb_blocks': 4,
            'upscale_factor': 4,
            'lr': 1e-4
        },
        'data': {
            'train_path': 'dummy_train.pt',
            'val_path': 'dummy_val.pt',
            'test_path': 'dummy_test.pt',
            'batch_size': 2,
            'num_workers': 0
        },
        'callbacks': {
            'checkpoint': {
                'dirpath': 'test_checkpoints',
                'filename': 'test-{epoch:02d}',
                'save_top_k': 1,
                'monitor': 'val_loss'
            }
        }
    }

@pytest.fixture
def mock_wandb():
    with patch('wandb.init'), patch('wandb.finish'):
        yield

class TestTrain:
    def test_config_loading(self, tmp_path, sample_config):
        # Create temporary config file
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
            
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = open(config_path)
            with patch('src.train.yaml.safe_load') as mock_load:
                mock_load.return_value = sample_config
                with patch('src.train.WandbLogger'):
                    with patch('src.train.DataModule'):
                        with patch('src.train.Generator'):
                            with patch('lightning.Trainer'):
                                train()
                                mock_load.assert_called_once()

    @pytest.mark.integration
    def test_train_integration(self, tmp_path, sample_config, mock_wandb):
        # Create temporary config file
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = open(config_path)
            with patch('src.train.yaml.safe_load') as mock_load:
                mock_load.return_value = sample_config
                with patch('src.train.DataModule') as mock_datamodule:
                    with patch('src.train.Generator') as mock_generator:
                        with patch('src.train.L.Trainer') as mock_trainer:
                            # Run training
                            train()
                            
                            # Verify components were initialized correctly
                            mock_datamodule.assert_called_once()
                            mock_generator.assert_called_once()
                            mock_trainer.return_value.fit.assert_called_once()

    def test_checkpoint_directory_creation(self, tmp_path, sample_config):
        checkpoint_dir = tmp_path / "checkpoints"
        sample_config['trainer']['checkpoint_dir'] = str(checkpoint_dir)
        
        with patch('builtins.open', create=True):
            with patch('src.train.yaml.safe_load') as mock_load:
                mock_load.return_value = sample_config
                with patch('src.train.WandbLogger'):
                    with patch('src.train.DataModule'):
                        with patch('src.train.Generator'):
                            with patch('lightning.Trainer'):
                                train()
                                assert os.path.exists(checkpoint_dir)

if __name__ == '__main__':
    pytest.main([__file__]) 