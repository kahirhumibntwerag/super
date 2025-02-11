import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class Dataset(Dataset):
    def __init__(self, tensors, downsample_factor=1/4, transform=None):
        self.tensors = tensors
        self.downsample_factor = downsample_factor
        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):        
        hr = self.tensors[idx]
        if len(hr.shape) == 2:
            hr = hr.unsqueeze(0)
        
        if self.transform:
            hr = self.transform(hr)
        
        hr = hr.float().view(-1, 1, 512, 512)
        lr = F.interpolate(
            hr, 
            size=(int(512*self.downsample_factor), int(512*self.downsample_factor)), 
            mode='bilinear', 
            align_corners=False
        )
            
        return lr.squeeze(0), hr.squeeze(0)

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        downsample_factor: float = 1/4,
        transform = None
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.downsample_factor = downsample_factor
        self.transform = transform
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_tensors = torch.load(self.train_path)
            val_tensors = torch.load(self.val_path)
            
            self.train_dataset = Dataset(
                tensors=train_tensors,
                downsample_factor=self.downsample_factor,
                transform=self.transform
            )
            
            self.val_dataset = Dataset(
                tensors=val_tensors,
                downsample_factor=self.downsample_factor,
                transform=self.transform
            )
            
        if stage == 'test' and self.test_path:
            test_tensors = torch.load(self.test_path)
            self.test_dataset = Dataset(
                tensors=test_tensors,
                downsample_factor=self.downsample_factor,
                transform=self.transform
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        if self.test_path:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
