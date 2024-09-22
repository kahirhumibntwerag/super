import numpy as np
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, numpy_data, downsample_factor=1/4, transform=None):
        """
        Args:
            numpy_data (numpy.array): Numpy array of shape (num_images, height, width).
            window_size (int): Number of images in each window.
            split (str): 'train', 'val', or 'test' to specify the dataset split.
            downsample_factor (float): Factor by which to downsample images.
            transform (callable, optional): Transformation function to apply to images.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the test split to include in the validation set.
            random_state (int): Random state for reproducibility.
        """
        self.numpy_data = numpy_data
        self.downsample_factor = downsample_factor
        self.transform = transform

    def __len__(self):
        return len(self.numpy_data)

    def __getitem__(self, idx):        
        
        hr = self.numpy_data[idx].compute()
        hight, width = hr.shape

        
        if self.transform:
            hr = self.transform(hr).float().view(1, 1, 512, 512)
            lr = F.interpolate(hr, size=(int(hight*self.downsample_factor), int(width*self.downsample_factor)), mode='bilinear', align_corners=False)
            
        return lr.squeeze(0), hr.squeeze(0)
