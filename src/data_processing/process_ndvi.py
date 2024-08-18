"""
This file contains functions for processing NDVI values of minicubes for model training
"""

import torch

def mask_low_ndvi_values(average_ndvi, ndvi_cube, placeholder=-9999.0):
    """
    Mask pixels with average NDVI < 0.2 in all time periods with a placeholder value.

    Args:
        average_ndvi (torch.Tensor): 2D tensor of average NDVI values for each pixel.
        ndvi_cube (torch.Tensor): 3D tensor of NDVI values with dimensions [time, height, width].
        placeholder (float): Value to use for masking low NDVI values.

    Returns:
        torch.Tensor: NDVI cube with low NDVI values masked with the placeholder value.
    """
    # Identify pixels with average NDVI < 0.2
    low_value_mask = average_ndvi < 0.2

    # Apply this mask across all time periods for the NDVI cube
    for t in range(ndvi_cube.shape[0]):  # first dimension is time
        ndvi_cube[t][low_value_mask] = placeholder

    # Convert all NaN values to the placeholder value in the NDVI cube
    ndvi_cube[torch.isnan(ndvi_cube)] = placeholder

    return ndvi_cube


def calculate_average_ndvi_for_each_pixel(ndvi_cube, placeholder=-9999.0):
    """
    Calculate the average NDVI value for each pixel, excluding placeholder values.

    Args:
        ndvi_cube (torch.Tensor): 3D tensor of NDVI values with dimensions [time, height, width].

    Returns:
        torch.Tensor: 2D tensor of average NDVI values for each pixel.
    """
    # Convert the NDVI cube to float32 to handle NaN values properly
    ndvi_cube = ndvi_cube.to(torch.float32)
    
    # Replace placeholder values with NaN to exclude them from mean calculation
    ndvi_cube[ndvi_cube == placeholder] = torch.nan

    # Calculate the average NDVI for each pixel along the time dimension, ignoring NaNs
    average_ndvi = torch.nanmean(ndvi_cube, dim=0)
    
    return average_ndvi

