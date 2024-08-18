""" Function to apply STL interpolation on a single NDVI time series
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

def stl_interpolate(ndvi_pixel):
    df = ndvi_pixel.to_dataframe(name='NDVI').reset_index()
    df.index = pd.date_range(start='2017-07-04', periods=len(df), freq='5D')
    
    imputed_indices = df[df.isnull()].index
    df['NDVI'] = df['NDVI'].astype('float32')
    
    if not df['NDVI'].isnull().all():
        stl = STL(df['NDVI'].interpolate(), seasonal=73) # seasonal period of 73 (approximately one year with 5-day intervals)
        res = stl.fit()
        
        seasonal_component = res.seasonal
        df_deseasonalised = df['NDVI'] - seasonal_component
        
        df_deseasonalised_imputed = df_deseasonalised.interpolate(method="linear")
        df_deseasonalised_imputed = df_deseasonalised_imputed.astype('float32')
        
        df_imputed = df_deseasonalised_imputed + seasonal_component
        df.loc[imputed_indices, 'NDVI'] = df_imputed[imputed_indices].astype('float32')
    
    return df['NDVI']