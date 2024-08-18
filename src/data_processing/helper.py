"""
Helper functions.
"""

from typing import Optional
import os
import pickle


def get_var_quantiles(file: Optional[str] = None) -> dict:
    """get the quantiles for the climate variables, the pickle is distributed
    with this package in the "data" directory."""
    if file is None:
        pklpath = os.path.join(os.path.dirname(__file__), "climate_vars_dict.pkl")
    else:
        pklpath = file
    with open(pklpath, "rb") as f:
        quantiles = pickle.load(f)
    return quantiles
