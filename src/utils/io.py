import rasterio as rio
from pathlib import Path

def get_metadata(path: Path):
    with rio.open(path) as dataset:
        return {
            "crs": dataset.crs,
            "transform": dataset.transform,
            "shape": dataset.shape,
            "dtype": dataset.dtypes[0],
            "band_count": dataset.count
        }

def load_raster(path: Path):
    with rio.open(path) as dataset:
        return dataset.read()
    
def load_bands(path: Path, bands: list[int]):
    with rio.open(path) as dataset:
        return dataset.read(bands)

def validate_raster(path: Path, expected_bands: int = 13):
    # Standardize to a Path object
    path = Path(path)
    
    # 1. Check if path is readable/exists
    if not path.is_file():
        raise FileNotFoundError(f"Path is not a readable file or does not exist: {path}")

    with rio.open(path) as src:
        # 2. Check for empty raster (dimensions)
        if src.width == 0 or src.height == 0:
            raise ValueError(f"Empty raster dimensions: {src.width}x{src.height}")
        
        # 3. Band Count Logic
        
        if src.count != expected_bands:
            raise ValueError(
                f"Band count mismatch for '{path.name}'. "
                f"Expected {expected_bands}, found {src.count}"
            )

        # 4. Dtype Check
        if not src.dtypes[0]:
            raise ValueError("Raster has no valid data type (dtype is undefined).")

    return True

def check_band_indices(path: Path, expected_bands: list[int] = [1,2,3,4,5,6,7,8,9,10,11,12,13]):
    with rio.open(path) as dataset:
        actual_indices = set(dataset.indexes)
        missing_indices = set(expected_bands) - actual_indices
        if missing_indices:
            raise ValueError(f"Missing expected band indices: {missing_indices}")
    return True