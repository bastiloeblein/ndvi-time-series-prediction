"""
Import minicubes
"""

import pandas as pd
import s3fs
import sys
import math

sys.path.insert(0, "../src/")
import constants
from constants import BAD_CUBES


def read_registry(path: str, minicubefs: s3fs.S3FileSystem) -> pd.DataFrame:
    """
    Read a registry file from an S3 bucket using s3fs.

    Parameters:
    path (str): The S3 path to the file (including the bucket name).
    minicubefs (s3fs.S3FileSystem): An initialized s3fs file system object.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    # List all files in the given path
    full_paths = minicubefs.ls(path)

    # Extract just the file name from each full path
    file_ids = [full_path.split("/")[-1] for full_path in full_paths]

    return pd.DataFrame({"mc_id": file_ids})


def remove_cubes(
    cube_registry: pd.DataFrame, empty_cube_list: set = BAD_CUBES
) -> pd.DataFrame:
    """
    Remove specified cubes from the registry based on a set of cube IDs.

    Parameters:
    cube_registry (pd.DataFrame): A pandas DataFrame containing the cube registry,
                                  expected to include a column 'mc_id' that contains cube IDs.

    Returns:
    pd.DataFrame: The updated registry after removing the specified cubes.
    """
    # Filter out the cubes that are in the empty_cube_list
    filtered_registry = cube_registry[~cube_registry["mc_id"].isin(empty_cube_list)]
    return filtered_registry


def read_split_table(split_table_path: str) -> pd.DataFrame:
    """read table containing the CV splits from disk"""
    return pd.read_csv(split_table_path, sep=";")


def preprocess_mc_id(mc_id: str) -> str:
    """Preprocess mc_id to remove the .zarr suffix"""
    return mc_id.replace(".zarr", "")


def postprocess_mc_id(mc_id: str) -> str:
    """Postprocess mc_id to add the .zarr suffix"""
    return mc_id + ".zarr"


def split_datasets(
    cube_registry: pd.DataFrame,
    split_table_path: str,
    test_group: int = 5,
    train_split: float = 0.8,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    load the dataset from disk and split it into training, validation and
    test sets.

    the data set has been pre-split into cross-validation groups. For the
    defautl machine learning application we use group 5 for testing.
    """

    # split table is a csv file with columns [mc_id, lon, lat, group] where
    # group" is the cv group.
    split_table = read_split_table(split_table_path)

    # Preprocess mc_id to remove .zarr suffix for merge
    cube_registry["mc_id"] = cube_registry["mc_id"].apply(preprocess_mc_id)

    # join split table and cube_registy on mc_id there are less cubes in
    # the split table than in the registry... default is inner join which
    # removes all cubes that do no have a cv group
    full_cube_registry = cube_registry.merge(split_table, on="mc_id")

    full_cube_registry["mc_id"] = full_cube_registry["mc_id"].apply(postprocess_mc_id)
    n_cubes = full_cube_registry.shape[0]
    n_train_cubes = math.ceil(n_cubes * train_split)

    # assign cubes to training/validation/test data
    train_test_split = full_cube_registry["group"] != test_group

    allcubes = full_cube_registry[train_test_split]
    # TODO: shuffle cubes, make this reproducible(?)
    allcubes = allcubes.sample(frac=1, random_state=random_seed)

    # results
    traincubes = allcubes[:n_train_cubes]
    valcubes = allcubes[n_train_cubes:]
    testcubes = full_cube_registry[~train_test_split]

    return traincubes, valcubes, testcubes
