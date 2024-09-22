import os
import s3fs
import zarr
from typing import Union
import dask.array as da


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


