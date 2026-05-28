import numpy as np

NIR_BAND = 8
RED_BAND = 4

def calculate_ndvi(band_data: np.ndarray):
    e = 1e-10
    nir_band = band_data[NIR_BAND-1]
    red_band = band_data[RED_BAND-1]
    return (nir_band.astype(float) - red_band.astype(float)) / (nir_band.astype(float) + red_band.astype(float) + e)