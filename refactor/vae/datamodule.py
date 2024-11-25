import lightning as L
from torch.utils.data import DataLoader
from data.Dataset import Dataset
from data.LoadData import load_single_aws_zarr, AWS_ZARR_ROOT, s3_connection
from torchvision import transforms
 

class DataModule(L.LightningDataModule):
    def __init__(self, year=2015, wavelength='171A', batch_size=4, downsample_factor=1/4, transform=None):
        super().__init__()
        self.year = year
        self.wavelength = wavelength
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.transform = transform



    def prepare_data(self):
        print("Connecting to S3 and downloading metadata...")
        self.data = load_single_aws_zarr(
            path_to_zarr=AWS_ZARR_ROOT+str(self.year),
            wavelength=self.wavelength
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Dataset(numpy_data=self.data[::800], downsample_factor=self.downsample_factor, transform=self.transform)
            self.val_dataset = Dataset(numpy_data=self.data[::700], downsample_factor=self.downsample_factor, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = Dataset(numpy_data=self.data[::750], downsample_factor=self.downsample_factor, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)  

    def val_dataloader(self):
        return DataLoader(self.val_dataset , batch_size=self.batch_size, shuffle=False)  

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False) 



if __name__ == '__main__':
    datamodule = DataModule()
     