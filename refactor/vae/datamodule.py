import lightning as L
from torch.utils.data import DataLoader
from data.Dataset import Dataset
from data.LoadData import load_single_aws_zarr, AWS_ZARR_ROOT, s3_connection
from torchvision import transforms
import dask.array as da
import boto3
import torch
import io
import os

def load_tensor_from_s3(bucket_name, s3_key, aws_access_key=None, aws_secret_key=None, region_name='eu-north-1', save_to_disk_path=None):
    if aws_access_key and aws_secret_key:
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=region_name)
    else:
        s3 = boto3.client('s3', region_name=region_name)
        
    download_buffer = io.BytesIO()

    try:
        s3.download_fileobj(bucket_name, s3_key, download_buffer)
        download_buffer.seek(0)
        tensor = torch.load(download_buffer)
        print("Tensor loaded successfully from S3.")

        # Save the tensor to disk if save_to_disk_path is provided
        if save_to_disk_path:
            torch.save(tensor, save_to_disk_path)
            print(f"Tensor saved to disk at {save_to_disk_path}")

        return tensor

    except Exception as e:
        print(f"Error loading tensor from S3: {e}")
        return None

class DataModule(L.LightningDataModule):
    def __init__(self, bucket_name, s3_key_train, s3_key_val, s3_key_test, aws_access_key, aws_secret_key=None, batch_size=4, downsample_factor=1/4, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.transform = transform
        self.s3_key_train = s3_key_train
        self.s3_key_val = s3_key_val
        self.s3_key_test = s3_key_test
        self.bucket_name = bucket_name
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key

    def prepare_data(self):
        print("Connecting to S3 and downloading metadata...")
        if not os.path.exists('train_data.pt'):
            load_tensor_from_s3(bucket_name=self.bucket_name, s3_key=self.s3_key_train, aws_access_key=self.aws_access_key, aws_secret_key=self.aws_secret_key, save_to_disk_path='train_data.pt')
        if not os.path.exists('val_data.pt'):
            load_tensor_from_s3(bucket_name=self.bucket_name, s3_key=self.s3_key_val, aws_access_key=self.aws_access_key, aws_secret_key=self.aws_secret_key, save_to_disk_path='val_data.pt')
        if not os.path.exists('test_data.pt'):
            load_tensor_from_s3(bucket_name=self.bucket_name, s3_key=self.s3_key_test, aws_access_key=self.aws_access_key, aws_secret_key=self.aws_secret_key, save_to_disk_path='test_data.pt')
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Dataset(tensors=torch.load('train_data.pt'), downsample_factor=self.downsample_factor, transform=self.transform)
            self.val_dataset = Dataset(tensors=torch.load('val_data.pt'), downsample_factor=self.downsample_factor, transform=self.transform)

        if stage == "test" or stage is None:
            self.test_dataset = Dataset(tensors=torch.load('test_data.pt'), downsample_factor=self.downsample_factor, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

if __name__ == '__main__':
    datamodule = DataModule(
        bucket_name='your-bucket-name',
        s3_key_train='train-key',
        s3_key_val='val-key',
        s3_key_test='test-key',
        aws_access_key='your-access-key',
        aws_secret_key='your-secret-key'
    )
