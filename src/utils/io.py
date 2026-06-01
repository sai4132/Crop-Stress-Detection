import rasterio as rio
from dataclasses import dataclass
import numpy as np

@dataclass
class RasterMetadata:
    crs: rio.crs.CRS
    transform: rio.transform.Affine
    shape: tuple[int, int]
    dtype: np.dtype
    band_count: int

def get_raster_metadata(raster: rio.DatasetReader):
    return RasterMetadata(
        crs=raster.crs,
        transform=raster.transform,
        shape=raster.shape,
        dtype=raster.dtypes[0],
        band_count=raster.count
    )

def load_raster(raster = rio.DatasetReader):
    return raster.read()

def load_bands(raster = rio.DatasetReader, bands: list[int] = None):
    return raster.read(bands)