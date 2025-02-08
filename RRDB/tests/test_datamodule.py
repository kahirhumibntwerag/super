import pytest
import torch
import os
from src.datamodule import Dataset, DataModule
import torch.nn.functional as F

# Define Transform class at module level
class Transform:
    def __call__(self, x):
        return x / x.max()

@pytest.fixture
def sample_tensors():
    # Create 10 sample tensors of size 512x512
    return [torch.rand(512, 512) for _ in range(10)]

@pytest.fixture
def temp_data_files(sample_tensors, tmp_path):
    # Create temporary tensor files for testing
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    test_path = tmp_path / "test.pt"
    
    torch.save(sample_tensors[:6], train_path)  # 6 for training
    torch.save(sample_tensors[6:8], val_path)   # 2 for validation
    torch.save(sample_tensors[8:], test_path)   # 2 for testing
    
    return {
        'train_path': str(train_path),
        'val_path': str(val_path),
        'test_path': str(test_path)
    }

@pytest.fixture
def sample_transform():
    return Transform()

class TestDataset:
    def test_initialization(self, sample_tensors):
        dataset = Dataset(sample_tensors)
        assert len(dataset) == len(sample_tensors)
        
    def test_getitem_shape(self, sample_tensors):
        dataset = Dataset(sample_tensors)
        lr, hr = dataset[0]
        
        # Check shapes
        assert hr.shape == (1, 512, 512)  # Added channel dimension
        assert lr.shape == (1, 128, 128)  # Downsampled by factor 1/4
        
    def test_downsample_factor(self, sample_tensors):
        # Test different downsample factors
        factors = [1/2, 1/4, 1/8]
        for factor in factors:
            dataset = Dataset(sample_tensors, downsample_factor=factor)
            lr, hr = dataset[0]
            expected_size = int(512 * factor)
            assert lr.shape == (1, expected_size, expected_size)
            
    def test_transform(self, sample_tensors, sample_transform):
        dataset = Dataset(sample_tensors, transform=sample_transform)
        lr, hr = dataset[0]
        
        # Check if values are normalized
        assert torch.all(hr <= 1.0)
        assert torch.all(lr <= 1.0)

class TestDataModule:
    def test_initialization(self, temp_data_files):
        datamodule = DataModule(**temp_data_files, batch_size=2)
        assert datamodule.batch_size == 2
        assert datamodule.train_path == temp_data_files['train_path']
        
    def test_setup_fit(self, temp_data_files):
        datamodule = DataModule(**temp_data_files)
        datamodule.setup(stage='fit')
        
        assert hasattr(datamodule, 'train_dataset')
        assert hasattr(datamodule, 'val_dataset')
        assert len(datamodule.train_dataset) == 6
        assert len(datamodule.val_dataset) == 2
        
    def test_setup_test(self, temp_data_files):
        datamodule = DataModule(**temp_data_files)
        datamodule.setup(stage='test')
        
        assert hasattr(datamodule, 'test_dataset')
        assert len(datamodule.test_dataset) == 2
        
    def test_train_dataloader(self, temp_data_files):
        datamodule = DataModule(**temp_data_files, batch_size=2)
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        
        # Check if DataLoader is configured correctly
        assert train_loader.batch_size == 2
        assert train_loader.num_workers == 4
        assert train_loader.pin_memory == True
        
        # Check batch shapes
        batch = next(iter(train_loader))
        lr, hr = batch
        assert lr.shape == (2, 1, 128, 128)  # Batch size 2
        assert hr.shape == (2, 1, 512, 512)
        
    def test_val_dataloader(self, temp_data_files):
        datamodule = DataModule(**temp_data_files, batch_size=2)
        datamodule.setup(stage='fit')
        val_loader = datamodule.val_dataloader()
        
        batch = next(iter(val_loader))
        lr, hr = batch
        assert lr.shape == (2, 1, 128, 128)
        assert hr.shape == (2, 1, 512, 512)
        
    def test_test_dataloader(self, temp_data_files):
        datamodule = DataModule(**temp_data_files, batch_size=2)
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        
        batch = next(iter(test_loader))
        lr, hr = batch
        assert lr.shape == (2, 1, 128, 128)
        assert hr.shape == (2, 1, 512, 512)
        
    def test_with_transform(self, temp_data_files, sample_transform):
        datamodule = DataModule(**temp_data_files, transform=sample_transform)
        datamodule.setup(stage='fit')
        
        # Check if transform is applied
        batch = next(iter(datamodule.train_dataloader()))
        lr, hr = batch
        assert torch.all(hr <= 1.0)
        assert torch.all(lr <= 1.0)

if __name__ == '__main__':
    pytest.main([__file__]) 