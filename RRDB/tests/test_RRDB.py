import pytest
import torch
import torch.nn as nn
from src.RRDB import ResidualBlock, RRDB, Generator, LightningGenerator

@pytest.fixture
def sample_input():
    return torch.randn(1, 1, 32, 32)  # Batch size 1, 1 channel, 32x32 image

@pytest.fixture
def generator_config():
    return {
        'in_channels': 1,
        'initial_channel': 64,
        'num_rrdb_blocks': 4,
        'upscale_factor': 4,
        'lr': 1e-4
    }

class TestResidualBlock:
    def test_initialization(self):
        block = ResidualBlock(64)
        assert isinstance(block, nn.Module)
        assert len(list(block.parameters())) > 0

    def test_forward_shape(self, sample_input):
        block = ResidualBlock(1)
        output = block(sample_input)
        assert output.shape == sample_input.shape

    def test_residual_connection(self, sample_input):
        block = ResidualBlock(1)
        output = block(sample_input)
        # Check that output is different from input (has been modified)
        assert not torch.allclose(output, sample_input)

class TestRRDB:
    def test_initialization(self):
        rrdb = RRDB(64)
        assert isinstance(rrdb, nn.Module)
        assert len(list(rrdb.parameters())) > 0

    def test_forward_shape(self, sample_input):
        rrdb = RRDB(1)
        output = rrdb(sample_input)
        assert output.shape == sample_input.shape

    def test_num_dense_layers(self):
        num_dense_layers = 5
        rrdb = RRDB(64, num_dense_layers=num_dense_layers)
        assert len(rrdb.residual_blocks) == num_dense_layers

class TestGenerator:
    def test_initialization(self, generator_config):
        gen = Generator(**generator_config)
        assert isinstance(gen, nn.Module)
        assert len(list(gen.parameters())) > 0

    def test_forward_shape(self, sample_input, generator_config):
        gen = Generator(**generator_config)
        output = gen(sample_input)
        expected_size = sample_input.size(-1) * generator_config['upscale_factor']
        assert output.shape == (1, 1, expected_size, expected_size)

    def test_upscale_factor(self, sample_input):
        # Test different upscale factors
        for upscale_factor in [2, 4, 8]:
            gen = Generator(upscale_factor=upscale_factor)
            output = gen(sample_input)
            expected_size = sample_input.size(-1) * upscale_factor
            assert output.shape == (1, 1, expected_size, expected_size)

class TestLightningGenerator:
    def test_initialization(self, generator_config):
        model = LightningGenerator({'generator': generator_config})  # Wrap config in dict
        assert isinstance(model, LightningGenerator)
        assert hasattr(model, 'generator')

    def test_forward(self, sample_input, generator_config):
        model = LightningGenerator({'generator': generator_config})
        output = model(sample_input)
        expected_size = sample_input.size(-1) * generator_config['upscale_factor']
        assert output.shape == (1, 1, expected_size, expected_size)

    def test_training_step(self, generator_config):
        model = LightningGenerator({'generator': generator_config})
        batch = (
            torch.randn(1, 1, 32, 32),  # LR
            torch.randn(1, 1, 128, 128)  # HR
        )
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_configure_optimizers(self, generator_config):
        model = LightningGenerator({'generator': generator_config})
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)

if __name__ == '__main__':
    pytest.main([__file__]) 