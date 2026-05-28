from src.utils.io import validate_raster, check_band_indices, get_metadata, load_raster
from src.visualization.rgb import visualize_rgb
from src.visualization.bands import visualize_bands
from src.visualization.histograms import visualize_histogram
from src.visualization.ndvi import visualize_ndvi
from pathlib import Path
import argparse
from src.utils.paths import PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR, CONFIG_DIR, MULTI_SPECTRAL_DIR, SAR_DIR, LAND_COVER_DIR, INSPECTION_OUTPUT_DIR
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from src.indices.vegetation import calculate_ndvi
import warnings

paths = []
if MULTI_SPECTRAL_DIR.exists():
    paths = sorted(list(MULTI_SPECTRAL_DIR.rglob("*.tif")))

if not paths:
    raise FileNotFoundError(f"No .tif files found in {MULTI_SPECTRAL_DIR}. Please check the directory and ensure it contains the expected data.")

def main():
    parser = argparse.ArgumentParser(description="Inspect a specific patch by index.")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--index", type=int, help='Index of patch to inspect')
    group.add_argument("--random", action = 'store_true', help='Inspect a random patch')
    parser.add_argument("--bands", type = int, nargs='*', required=False, help='Band indices to visualize (default: [4, 3, 2] i.e, RGB)')
    parser.add_argument("--histogram-bands", type = int, nargs='*', required=False, help='Visualize histogram of a band of patch (default: [4, 3, 2] i.e, RGB)')
    parser.add_argument("--save", type = str, required = False, help = 'Relative directory path from project root to save the outputs')
    parser.add_argument("--no_rgb", action = 'store_true', help = 'Do not visualize RGB image')
    parser.add_argument("--ndvi", action = 'store_true', help = 'Visualize NDVI image')

    args = parser.parse_args()

    index = args.index if args.index else random.randint(0, len(paths)-1)
    if index < 0 or index >= len(paths):
        raise ValueError(f"Index out of range. Must be between 0 and {len(paths)-1}")
    
    patch_path = paths[index]

    if validate_raster(patch_path):
        print(f"The patch at index {index} is valid and can be processed")
        
    metadata = get_metadata(patch_path)
    band_data = load_raster(patch_path)
    print(f"\nPatch CRS: {metadata['crs']} \nPatch shape: {metadata['shape']} \nPatch dtype: {metadata['dtype']} \nBand count: {metadata['band_count']}")
    print(f"\nPatch diagnostics are: \nmin_value: {band_data.min()}, max_value: {band_data.max()} \n \
          mean_value: {band_data.mean()}, std_value: {band_data.std()}\n \
          NaN values: {np.isnan(band_data).sum()} \nInf values: {(band_data == np.inf).sum()}\n")

    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not args.no_rgb:
        RGB_BANDS = [4, 3, 2]
        if check_band_indices(patch_path, expected_bands=RGB_BANDS):
            rgb_fig, _ = visualize_rgb(band_data, index)
            save_path = INSPECTION_OUTPUT_DIR/f"rgb/INDEX_{index}_RGB_{time_stamp}.png" if args.save is None \
                else PROJECT_ROOT/args.save/f"INDEX_{index}_RGB_{time_stamp}.png"
            rgb_fig.savefig(save_path)
            print(f"RGB visualization saved to {save_path}")
            plt.close(rgb_fig)


    if args.bands is not None:
        if args.bands==[]:
            args.bands = [4, 3, 2]
        if check_band_indices(patch_path, expected_bands=args.bands):
            bands_fig, _ = visualize_bands(band_data, args.bands, index)
            save_path = INSPECTION_OUTPUT_DIR/f"bands/INDEX_{index}_BANDS_{args.bands}_{time_stamp}.png" if args.save is None \
                else PROJECT_ROOT/args.save/f"INDEX_{index}_BANDS_{args.bands}_{time_stamp}.png"
            bands_fig.savefig(save_path)
            print(f"Bands visualization saved to {save_path}")
            plt.close(bands_fig)


    if args.histogram_bands is not None: 
        if args.histogram_bands == []:
            args.histogram_bands = [4, 3, 2]
        if check_band_indices(patch_path, expected_bands=args.histogram_bands):
            histogram_fig, _ = visualize_histogram(band_data, index, bands=args.histogram_bands)
            save_path = INSPECTION_OUTPUT_DIR/f"histograms/INDEX_{index}_HISTOGRAM_BAND_{args.histogram_bands}_{time_stamp}.png" if args.save is None \
                else PROJECT_ROOT/args.save/f"INDEX_{index}_HISTOGRAM_BAND_{args.histogram_bands}_{time_stamp}.png"
            histogram_fig.savefig(save_path)
            print(f"Histogram visualization saved to {save_path}")
            plt.close(histogram_fig)
    
    if args.ndvi:
        if check_band_indices(patch_path, expected_bands=[4, 8]):
            ndvi_data = calculate_ndvi(band_data)
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
            ndvi_fig, _ = visualize_ndvi(ndvi_data, index)
            save_path = INSPECTION_OUTPUT_DIR/f"ndvi/INDEX_{index}_NDVI_{time_stamp}.png" if args.save is None \
                else PROJECT_ROOT/args.save/f"INDEX_{index}_NDVI_{time_stamp}.png"
            ndvi_fig.savefig(save_path)
            print(f"NDVI visualization saved to {save_path}")
            plt.close(ndvi_fig)

if __name__ == "__main__":
    main()