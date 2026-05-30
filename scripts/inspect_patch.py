from pathlib import Path
import argparse
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import warnings
import rasterio as rio
from src.utils import io, metadata, paths
from src.visualization import rgb, bands, histograms, ndvi, lc, sar
from src.indices import vegetation

file_paths = []
if paths.MULTI_SPECTRAL_DIR.exists():
    file_paths = sorted(list(paths.MULTI_SPECTRAL_DIR.rglob("*.tif")))

if not file_paths:
    raise FileNotFoundError(f"No .tif files found in {paths.MULTI_SPECTRAL_DIR}. Please check the directory and ensure it contains the expected data.")

sample_data = dict()
for idx, path in enumerate(file_paths):
    sample_data[idx] = {"s2": path,
                        "s1": metadata.get_s1_path_from_s2_path(path, paths.SAR_DIR),
                        "lc": metadata.get_lc_path_from_s2_path(path, paths.LAND_COVER_DIR)}

def validate_raster(path: Path, raster_metadata: dict = None, sample_metadata: dict = None):

    # 1. Check for empty raster (dimensions)
    if 0 in raster_metadata["shape"]:
        raise ValueError(f"Empty raster dimensions: {raster_metadata['shape']} for file: {path}")
    
    # 2. Sensor Type Logic
    if sample_metadata["sensor"] == "s2":
        expected_bands = 13
    elif sample_metadata["sensor"] == "s1":
        expected_bands = 2
    elif sample_metadata["sensor"] == "lc":
        expected_bands = 4
    else:
        raise ValueError(f"Unknown sensor '{sample_metadata['sensor']}'. Expected 's2', 's1', or 'lc'.")
    
    # 3. Band Count Logic
    if raster_metadata["band_count"] != expected_bands:
        raise ValueError(
            f"Band count mismatch for '{path.name}'. "
            f"Expected {expected_bands}, found {raster_metadata['band_count']}"
        )

    # 4. Dtype Check
    if not raster_metadata["dtype"]:
        raise ValueError("Raster has no valid data type (dtype is undefined).")

    return True

def save_figure(fig: plt.Figure, save_path: Path, index: int, sensor: str, folder: str, visualization_type: str, time: str):
    save_path = paths.INSPECTION_OUTPUT_DIR/f"{sensor}/{folder}/INDEX_{index}_{visualization_type}_{time}.png" if save_path is None \
        else save_path/f"{sensor}_INDEX_{index}_{visualization_type}_{time}.png"
    fig.savefig(save_path)
    print(f"{visualization_type} visualization saved to {save_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Inspect a specific patch by index.")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--index", type=int, help='Index of patch to inspect')
    group.add_argument("--random", action = 'store_true', help='Inspect a random patch')
    parser.add_argument("--sensor", type = str, required = True, nargs = '+', choices=['s1', 's2', 'lc', 'all'], help='Sensor type(s) to inspect')
    parser.add_argument("--validate-only", action = 'store_true', help='Only validate the patch without visualizations and prints diagnostics')
    parser.add_argument("--bands", type = int, nargs='*', required=False, help='Band indices to visualize (default: [4, 3, 2] i.e, RGB)')
    parser.add_argument("--histogram-bands", type = int, nargs='*', required=False, help='Visualize histogram of a band of patch (default: [4, 3, 2] i.e, RGB)')
    parser.add_argument("--save", type = str, required = False, help = 'Relative directory path from project root to save the outputs')
    parser.add_argument("--no_rgb", action = 'store_true', help = 'Do not visualize RGB image')
    parser.add_argument("--ndvi", action = 'store_true', help = 'Visualize NDVI image')

    args = parser.parse_args()

    index = args.index if args.index else random.randint(0, len(sample_data)-1)
    if index < 0 or index >= len(sample_data):
        raise ValueError(f"Index out of range. Must be between 0 and {len(sample_data)-1}")
    
    sensors = ['s1', 's2', 'lc'] if 'all' in args.sensor else sorted(list(set(args.sensor)))

    for sensor in sensors:

        patch_path = sample_data[index][sensor]
        print(f"\nInspecting {sensor} patch at index {index} located at {patch_path}")
        if not patch_path.is_file():
            raise FileNotFoundError(f"File not found at path: {patch_path}")

        with rio.open(patch_path) as raster:
                
            raster_metadata = io.get_raster_metadata(raster)
            sample_metadata = metadata.parse_patch_path(patch_path)

            if validate_raster(patch_path, raster_metadata=raster_metadata, sample_metadata=sample_metadata):
                print(f"The {sensor} patch at index {index} is valid and can be processed")
            
            band_data = io.load_raster(raster)
            print(f"{sensor} Patch metadata and diagnostics:\n")
            print(f"\nPatch CRS: {raster_metadata['crs']} \nPatch shape: {raster_metadata['shape']} \nPatch dtype: {raster_metadata['dtype']} \nBand count: {raster_metadata['band_count']}")
            print(f"\nPatch diagnostics are: \nmin_value: {band_data.min()}, max_value: {band_data.max()} \n \
                mean_value: {band_data.mean()}, std_value: {band_data.std()}\n \
                NaN values: {np.isnan(band_data).sum()} \nInf values: {(band_data == np.inf).sum()}\n")
            
            if args.validate_only:
                continue

            time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if not args.no_rgb and sensor == 's2':
                rgb_fig, _ = rgb.visualize_rgb(band_data, index)
                save_figure(rgb_fig, args.save, index, sensor, "rgb", "RGB", time_stamp)


            if args.bands is not None and sensor == 's2':
                if args.bands==[]:
                    args.bands = [4, 3, 2]
                bands_fig, _ = bands.visualize_bands(band_data, args.bands, index)
                save_figure(bands_fig, args.save, index, sensor, "bands", f"BANDS_{args.bands}", time_stamp)


            if args.histogram_bands is not None and sensor == 's2': 
                if args.histogram_bands == []:
                    args.histogram_bands = [4, 3, 2]
                histogram_fig, _ = histograms.visualize_histogram(band_data, index, bands=args.histogram_bands)
                save_figure(histogram_fig, args.save, index, sensor, "histograms", f"HISTOGRAM_BAND_{args.histogram_bands}", time_stamp)

            if args.ndvi and sensor == 's2':
                ndvi_data = vegetation.calculate_ndvi(band_data)

                ndvi_diagnostics = {
                    "min": ndvi_data.min(),
                    "max": ndvi_data.max(),
                    "mean": ndvi_data.mean(),
                    "std": ndvi_data.std()
                }

                if ndvi_diagnostics["min"] < -1 or ndvi_diagnostics["max"] > 1:
                    warnings.warn(f"NDVI values are out of expected range [-1, 1]")
                
                print(f"\nNDVI diagnostics are: \nmin_value: {ndvi_diagnostics['min']}, max_value: {ndvi_diagnostics['max']} \n \
                    mean_value: {ndvi_diagnostics['mean']}, std_value: {ndvi_diagnostics['std']}\n \
                    NaN values: {np.isnan(ndvi_data).sum()} \nInf values: {(ndvi_data == np.inf).sum()}\n")
                
                ndvi_fig, _ = ndvi.visualize_ndvi(ndvi_data, index)
                save_figure(ndvi_fig, args.save, index, sensor, "ndvi", "NDVI", time_stamp)

            if sensor == 'lc':
                lc_fig, _ = lc.visualize_land_cover(band_data, index=index)
                save_figure(lc_fig, args.save, index, sensor, "land_cover", "LAND_COVER", time_stamp)

                lc_histogram_fig, _ = lc.visualize_land_cover_histogram(band_data, index=index)
                save_figure(lc_histogram_fig, args.save, index, sensor, "histograms", "LAND_COVER_HISTOGRAM", time_stamp)

            if sensor == 's1':
                sar_fig, _ = sar.visualize_sar(band_data, index=index)
                save_figure(sar_fig, args.save, index, sensor, "sar", "SAR", time_stamp)

                sar_histogram_fig, _ = sar.visualize_sar_histogram(band_data, index=index)
                save_figure(sar_histogram_fig, args.save, index, sensor, "histograms", "SAR_HISTOGRAM", time_stamp)

if __name__ == "__main__":
    main()