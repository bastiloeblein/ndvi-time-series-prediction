"""
Functions to perform sanity checks on processed ndvi data and cubes
"""

import os
import xarray as xr
import numpy as np

def check_cube_integrity(ndvi_data, expected_shape):
    """Ensure that the shape and dimensions of each cube are consistent and correct."""
    assert ndvi_data.shape == expected_shape, f"NDVI cube shape mismatch. Expected {expected_shape}, got {ndvi_data.shape}"

def check_ndvi_distribution(ndvi_data):
    """
    Analyze the distribution of NDVI values to ensure they fall within the expected range [-1, 1].
    Values of -9.0 are not considered in the analysis.
    """
    # Ensure no NaN values in the data for distribution analysis
    valid_ndvi_data = ndvi_data[~np.isnan(ndvi_data)]
       
    # Check the min and max values within the valid NDVI data
    min_ndvi = np.min(valid_ndvi_data)
    max_ndvi = np.max(valid_ndvi_data)
    
    # Use np.isclose to handle floating-point precision issues
    assert min_ndvi >= -1.0 or np.isclose(min_ndvi, -1.0, atol=0.01), f"NDVI values below expected range. Found minimum NDVI: {min_ndvi}"
    assert max_ndvi <= 1.0 or np.isclose(max_ndvi, 1.0, atol=0.01), f"NDVI values above expected range. Found maximum NDVI: {max_ndvi}"
    
    
    print(f"NDVI min: {np.min(valid_ndvi_data)}, NDVI max: {np.max(valid_ndvi_data)}")

def check_if_contains_low_values(ndvi_data):
    """
    Check that there are no pixels with an average NDVI value below 0.2 over the entire time period.
    Also calculates the overall average NDVI and counts the number of masked values.

    Args:
        ndvi_data (np.ndarray): A 3D array of NDVI values with dimensions [time, height, width].

    Returns:
        tuple: Overall average NDVI and number of masked values
    """
    # Count the number of masked values
    number_of_masked_values = np.isnan(ndvi_data).sum()
    print(f"Number of masked values: {number_of_masked_values}")
    
    # Calculate the average NDVI for each pixel over time (first dimension)
    average_ndvi = np.nanmean(ndvi_data, axis=0)

    # Check which average NDVI values are below 0.2
    low_value_mask = average_ndvi < 0.2

    # Count the number of pixels with average NDVI below 0.2
    low_value_count = np.sum(low_value_mask)
    total_pixels = low_value_mask.size

    print(f"Number of pixels with average NDVI < 0.2: {low_value_count}")
    print(f"Total number of pixels: {total_pixels}")

    # Assert to ensure no pixels have an average NDVI below 0.2
    assert low_value_count == 0, "Found pixels with average NDVI below 0.2"

    # Calculate the overall average NDVI
    overall_average_ndvi = np.nanmean(ndvi_data)

    return overall_average_ndvi, number_of_masked_values
