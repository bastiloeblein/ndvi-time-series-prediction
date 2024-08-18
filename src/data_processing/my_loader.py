"""
This file contains functions and a dataset class for loading, processing, and analyzing NDVI values and other geospatial data from minicubes stored in an S3 bucket. It includes functionality to compute NDVI values from Sentinel2 Bands, apply cloud masks, and process various data types.
"""

import torch
import pandas as pd
import numpy as np
import zarr
import re
from torch.utils.data import Dataset
import s3fs
import constants
import matplotlib.pyplot as plt
from constants import (
    VARIABLE_KEYS,
    S2_BANDS,
    MIN_PERCENTILE,
    MAX_PERCENTILE,
    DEM_MIN,
    DEM_MAX,
    LANDCOVER_CLASSES,
)


class DeepCubeTSDatasetBasti(Dataset):
    """
    A PyTorch Dataset class for loading and processing data cubes stored in an S3 bucket using Zarr format.
    This dataset is tailored for handling multi-dimensional spatial data often used in environmental and geospatial studies.

    Attributes:
        s3_filesystem (s3fs.S3FileSystem): An instance of the S3FileSystem for accessing the S3 bucket.
        s3_bucket (str): The name of the S3 bucket where data cubes are stored.
        registry (pd.DataFrame): A DataFrame containing metadata for each cube, including paths or identifiers.
        quantiles (dict): Processed quantile data for normalizing dataset features.
        cubeshape (tuple): The expected spatial dimensions of each data cube.
        empty (tuple): Placeholder tensors for cases where data cannot be loaded properly.
        ndvi (dict): Computed NDVI values for all cubes.
        valid_cubes (list): List of valid cube indices to be loaded.

    Methods:
        __len__: Returns the total number of data points in the dataset.
        __repr__: Provides a human-readable representation of the dataset.
        idx2tsidxs: Maps a linear index to specific cube and spatial index within that cube.
        cubeidx2path: Constructs the S3 path to a data cube.
        __getitem__: Loads and returns a single item from the dataset.
        compute_and_store_ndvi: Computes and returns NDVI for a specific cube.
        compute_all_ndvi: Computes NDVI for all cubes.
        get_cube_class: Retrieves the class of a specific cube.
        get_classes: Retrieves the classes for all cubes.
        get_cube_dates: Retrieves the dates for a specific cube.
    """

    def __init__(
        self,
        s3_filesystem: s3fs.S3FileSystem,
        s3_bucket: str,
        registry: pd.DataFrame,
        quantile_data: dict,
        valid_cubes: list,
    ):
        """
        Initializes the dataset with an S3FileSystem instance, the S3 bucket name, registry, and quantile data.

        Args:
            s3_filesystem (s3fs.S3FileSystem): An instance of S3FileSystem for S3 bucket access.
            s3_bucket (str): The name of the S3 bucket where the data cubes are stored.
            registry (pd.DataFrame): Registry containing metadata of the cubes.
            quantile_data (dict): Dictionary with quantile information for data normalization.
            valid_cubes (list): List of valid cube indices to be loaded.
            number_of_cubes (int): The number of cubes that are loaded.
        """
        super().__init__()
        self.s3_filesystem = s3_filesystem
        self.s3_bucket = s3_bucket
        self.registry = registry
        self.valid_cubes = valid_cubes
        self.number_of_cubes = len(valid_cubes)
        self.cubeshape = (495, 128, 128)  # 495 time periods, 128 x 128 pixels
        self.quantiles = quantile_data2quantiles(quantile_data)
        out_shape = (constants.TIMESERIES_LENGTH, constants.NUM_VARIABLES)
        self.empty = (torch.zeros(out_shape), torch.ones(out_shape[0]).bool())

    def __len__(self):
        """Returns the total number of data points in the dataset."""
        return len(self.valid_cubes) * self.cubeshape[1] * self.cubeshape[2]

    def __repr__(self):
        """Provides a human-readable representation of the dataset."""
        return f"<(128 * 128 data points per cube): 16.384 * {len(self.valid_cubes)} (number of cubes) = {len(self)} (Total number of data points)>"

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Loads and returns a single item from the dataset based on the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple[torch.Tensor, torch.Tensor, float, float, str]:
            A tuple containing the NDVI tensor, cloud mask tensor, latitude, and longitude and the cube class.
        """
        # cube_id, (row_id, col_id) = self.idx2tsidxs(idx)
        cube_path = self.cubeidx2path(idx)

        # Get lat and long
        lat, lon = extract_lat_lon(cube_path)        
        print(cube_path)
        print(lat, lon)
        store = s3fs.S3Map(root=cube_path, s3=self.s3_filesystem)
        try:
            zarr_group = zarr.open(store, mode="r")
            # print("Access successful")
        except Exception as e:
            print("Error accessing the store:", e)
            return self.empty
        
        # Get class
        cube_class = self.get_cube_class(idx)

        # Load cloud mask
        if "cloudmask_en" not in zarr_group.keys():
            print(f"Missing cloudmask in cube {cube_path}")
            return self.empty
        cloudmask = load_cloudmask_cube(zarr_group)

        # Load s2 data
        nir_band = "B8A"  # Near-Infrared band
        red_band = "B04"  # Red band
        if nir_band not in zarr_group.keys() or red_band not in zarr_group.keys():
            print(f"Missing bands in cube {cube_path}")
            return self.empty

        nir = load_s2_cube(zarr_group, nir_band).squeeze()
        red = load_s2_cube(zarr_group, red_band).squeeze()

        # Apply cloud mask
        try:
            nir[~cloudmask] = torch.nan
            red[~cloudmask] = torch.nan
        except IndexError as e:
            print(f"Error applying cloud mask: {e}")
            print(
                f"NIR shape: {nir.shape}, Red shape: {red.shape}, Cloudmask shape: {cloudmask.shape}"
            )
            return self.empty

        # Compute NDVI
        ndvi = (nir - red) / (nir + red + 1e-10)
        # print(ndvi)

        # Ensure NDVI is in float32
        ndvi = ndvi.to(torch.float32)

        # Set NDVI values that are less than -1 to 0
        ndvi = torch.where(ndvi < -1, torch.tensor(0, dtype=torch.float32), ndvi)

        # Mask values that are nan or zero
        ndvi = torch.where(
            (ndvi == 0) | torch.isnan(ndvi),
            torch.tensor(-9999.0, dtype=torch.float32),
            ndvi,
        )

        return ndvi, cloudmask, lat, lon, cube_class


    def idx2tsidxs(self, idx: int) -> tuple[int, tuple[int, int]]:
        """
        Converts a linear index into a specific cube identifier and its corresponding spatial indices.

        Args:
            idx (int): Index of the data point in the dataset.

        Returns:
            tuple[int, tuple[int, int]]: Tuple containing the cube ID and the (row, column) index.
        """
        cube_id, ts_id = divmod(idx, self.cubeshape[1] * self.cubeshape[2])
        ts_xy = divmod(ts_id, self.cubeshape[2])
        return cube_id, ts_xy

    def cubeidx2path(self, cube_id: int) -> str:
        """
        Constructs the full S3 path to a data cube using its index in the registry.

        Args:
            cube_id (int): Index of the cube in the registry.

        Returns:
            str: Full S3 path to the cube.
        """
        cube_rel_path = self.registry["mc_id"].iloc[cube_id]
        return f"{self.s3_bucket}/{cube_rel_path}"

    def get_cube_class(self, cube_id):
        """
        Retrieve the class information for a specific data cube.

        Args:
            cube_id (int): The index of the cube in the dataset.

        Returns:
            str: The class information of the cube, or None if an error occurs.
        """
        # Construct the S3 path to the data cube using its index
        cube_path = self.cubeidx2path(cube_id)

        # Create a mapping for the Zarr store in the S3 bucket
        store = s3fs.S3Map(root=cube_path, s3=self.s3_filesystem)

        try:
            # Open the Zarr group for the data cube
            zarr_group = zarr.open(store, mode="r")

            # Access the class information from the metadata attribute
            class_info = zarr_group.attrs["metadata"]["class"]

            return class_info
        except Exception as e:
            # Print an error message if accessing metadata fails
            print(f"Error accessing metadata for cube ID {cube_id}: {e}")

            return None

    def get_classes(self, num_cubes=None):
        """
        Retrieve the class information for the specified number of data cubes.

        Args:
            num_cubes (int, optional): The number of cubes to retrieve classes for. Defaults to all cubes.

        Returns:
            dict: A dictionary where each key is a cube ID and each value is the class information of the cube.
        """
        # Initialize a dictionary to store the class information for each cube
        classes = {}

        # Iterate over the specified number of cube IDs in the dataset
        for cube_id in valid_cubes:
            # Retrieve and store the class information for each cube
            classes[cube_id] = self.get_cube_class(cube_id)

        return classes

    def get_cube_dates(self, cube_id):
        cube_path = self.cubeidx2path(cube_id)
        store = s3fs.S3Map(root=cube_path, s3=self.s3_filesystem)
        try:
            zarr_group = zarr.open(store, mode="r")
            # Assuming there's a time dataset or attribute in the Zarr group
            time_data = zarr_group["time"][:]
            readable_times = pd.to_datetime(
                time_data, unit="s", origin=pd.Timestamp("1970-01-01T00:00:00")
            )
            return readable_times
        except KeyError:
            print(f"Time data not found in cube {cube_id}")
        except Exception as e:
            print(f"Error accessing the store: {e}")
        return None


def quantile_data2quantiles(quantile_data):
    """
    Processes raw quantile data into a dictionary mapping variables to their minimum and maximum quantiles.

    Args:
        quantile_data (dict): Raw quantile data.

    Returns:
        dict: A dictionary with keys being variables and values being another dict with 'min' and 'max' keys.
    """
    res = {}
    for key in quantile_data.keys():
        res[key] = {
            "min": quantile_data[key]["percentiles"][MIN_PERCENTILE],
            "max": quantile_data[key]["percentiles"][MAX_PERCENTILE],
        }
    return res


def load_s2_cube(zarr_group: zarr.hierarchy.Group, band: str) -> torch.Tensor:
    """
    Load an entire Sentinel-2 band from the minicube.

    Args:
        zarr_group (zarr.hierarchy.Group): The Zarr group containing Sentinel-2 bands.
        band (str): The specific Sentinel-2 band to load (e.g., 'B02', 'B03').

    Returns:
        torch.Tensor: A tensor of shape [ntime, nrow, ncol] containing the normalized data for the specified band.
    """
    # Extract the full band data across all time points and spatial dimensions
    x = zarr_group[band][:]

    # Convert data to float32 for precision and normalization compatibility
    x = x.astype(np.float32)

    # Clip the data to handle out-of-range values, ensuring no values exceed 10000 or fall below -9999
    x = np.clip(x, -9999, 10000)

    # Normalize non-negative values to the range [0, 1]
    xltzero = x >= 0
    x[xltzero] = x[xltzero] / 10000

    # Convert the numpy array to a PyTorch tensor
    x = torch.from_numpy(x)

    # Ensure the tensor is of type float32
    x = x.type(torch.float32)

    return x


def load_cloudmask_cube(zarr_group: zarr.hierarchy.Group) -> torch.Tensor:
    """load the cloudmask from the minicube
    data has classes [0, 1, 2, 3, 4] where 0 is clear sky

    returns a bool tensor of shape [ntime, nrow, ncol]
    """
    cloudmask = zarr_group["cloudmask_en"][:] == 0
    cloudmask = torch.from_numpy(cloudmask)
    cloudmask = cloudmask.type(torch.bool)

    return cloudmask


def load_cloudmask_cube(zarr_group: zarr.hierarchy.Group) -> torch.Tensor:
    """
    Load the cloud mask data from a Zarr group and convert it to a boolean tensor indicating clear sky.

    Args:
        zarr_group (zarr.hierarchy.Group): The Zarr group containing cloud mask data.

    Returns:
        torch.Tensor: A boolean tensor of shape [ntime, nrow, ncol] where True indicates clear sky.
    """
    # Extract the cloud mask data across the full time and spatial extent
    cloudmask = zarr_group["cloudmask_en"][:]

    # Identify clear sky pixels (class 0) and create a boolean mask
    cloudmask = cloudmask == 0

    # Convert the boolean numpy array to a PyTorch tensor
    cloudmask = torch.from_numpy(cloudmask)

    # Ensure the tensor is of type bool
    cloudmask = cloudmask.type(torch.bool)

    return cloudmask


def extract_lat_lon(cube_path):
    """
    Extract latitude and longitude from a cube path string.

    Args:
        cube_path (str): The cube path string.

    Returns:
        tuple: A tuple containing latitude and longitude as floats.
    """
    # Use regular expression to find the latitude and longitude
    pattern = r"mc_([\-0-9\.]+)_([\-0-9\.]+)_"
    match = re.search(pattern, cube_path)
    
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return lat, lon
    else:
        raise ValueError("The input string does not match the expected format.")