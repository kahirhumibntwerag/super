import os
import s3fs
import zarr
from typing import Union
import dask.array as da
from torchvision import transforms
from .Dataset import Dataset
from torch.utils.data import DataLoader

AWS_ZARR_ROOT = (
    "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2.zarr/"
)


def s3_connection(path_to_zarr: os.path) -> s3fs.S3Map:
    """
    Instantiate connection to aws for a given path `path_to_zarr`
    """
    return s3fs.S3Map(
        root=path_to_zarr,
        s3=s3fs.S3FileSystem(anon=True),
        # anonymous access requires no credentials
        check=False,
    )


def load_single_aws_zarr(
    path_to_zarr: os.path,
    cache_max_single_size: int = None,
    wavelength='171A',
) -> Union[zarr.Array, zarr.Group]:
    """
    load zarr from s3 using LRU cache
    """
    root = zarr.open(
            zarr.LRUStoreCache(
                store=s3_connection(path_to_zarr),
                max_size=cache_max_single_size,
            ),
            mode="r",
         )
    data = root[wavelength]
    data = da.from_array(data)

    return data

from torch.utils.data.distributed import DistributedSampler

def build_dataloader(config):
        # Load the data from aws s3 
    data = load_single_aws_zarr(
        path_to_zarr=AWS_ZARR_ROOT + str(config.data.year),
        wavelength=config.data.wavelength
    )

    # Splitting the data into training and validation sets
    train_data = data[960:1000]
    val_data = data[:10]
    
    # Composing the transformation to be applied to the data
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Initializing the validation and training dataset and dataloader modules from PyTorch 
    downsample_factor = 1 / config.RRDB.model.upscale_factor
    train_dataset = Dataset(numpy_data=train_data, downsample_factor=downsample_factor, transform=transform)
    val_dataset = Dataset(numpy_data=val_data, downsample_factor=downsample_factor, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.RRDB.training.batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=config.RRDB.training.batch_size, shuffle=False, sampler=DistributedSampler(val_dataset))
    return train_loader, val_loader


if __name__ == '__main__':
    data = load_single_aws_zarr(path_to_zarr=AWS_ZARR_ROOT + str(2015), wavelength='171A')
    print(data[0].compute().shape)
