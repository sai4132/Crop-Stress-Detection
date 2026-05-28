import matplotlib.pyplot as plt
import math
import numpy as np


NIR_BAND = 8
RED_BAND = 4

def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray):
    e = 1e-10
    return (nir_band - red_band) / (nir_band + red_band + e)

def visualize_ndvi(band_data: np.ndarray, index: int):
    # if validate_raster(path) and check_band_indices(path, expected_bands=bands):
    nir_data = band_data[NIR_BAND-1]
    red_data = band_data[RED_BAND-1]
    ndvi_data = calculate_ndvi(nir_data, red_data)

    fig, axes = plt.subplots(figsize=(10, 10))
    axes.imshow(ndvi_data, cmap='magma')
    axes.set_title(f"NDVI Image of index_{index}")  
    
    return fig, axes