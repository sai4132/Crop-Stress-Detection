import rasterio as rio
from pathlib import Path

def get_raster_metadata(raster: rio.DatasetReader):
    return {
        "crs": raster.crs,
        "transform": raster.transform,
        "shape": raster.shape,
        "dtype": raster.dtypes[0],
        "band_count": raster.count
        }

def load_raster(raster = rio.DatasetReader):
    return raster.read()

def load_bands(raster = rio.DatasetReader, bands: list[int] = None):
    return raster.read(bands)