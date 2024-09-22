from data.LoadData import load_single_aws_zarr, AWS_ZARR_ROOT, s3_connection
from data.Dataset import Dataset
import torch.nn.functional as F
from RRDB.src.RRDB import Generator
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import dask.array as da
import matplotlib.pyplot as plt
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    

root = load_single_aws_zarr(
    path_to_zarr=AWS_ZARR_ROOT+str(2015),
    )

data = root['171A']
data = da.from_array(data)
hr = torch.from_numpy(data[46543].compute()).unsqueeze(0).unsqueeze(0)
lr = F.interpolate(hr, size=(128, 128), mode='bilinear', align_corners=False)

config = load_config(r'C:\Users\mhesh\OneDrive\Desktop\pro\SR\config\config.yml')
generator = Generator(
        in_channels=config['RRDB']['in_channels'],
        initial_channel=config['RRDB']['initial_channel'],
        num_rrdb_blocks=config['RRDB']['num_rrdb_blocks'],
        upscale_factor=config['RRDB']['upscale_factor']
    )

state_dict = torch.load(r'C:\Users\mhesh\OneDrive\Desktop\pro\SR\generator.pth')
# Load the state dictionary into the model
generator.load_state_dict(state_dict)

with torch.no_grad():
    generator.to('cuda')
    lr = lr.to('cuda')
    sr = generator(lr)
    sr = sr.squeeze(0).squeeze(0).cpu().numpy()
    plt.imshow(sr, cmap='afmhot')
    plt.show()

