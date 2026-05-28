import matplotlib.pyplot as plt
import numpy as np
# from configs.visualization import RGB_BANDS
RGB_BANDS = [4, 3, 2]

def normalize(band_data: np.ndarray):
    min_val, max_val = np.min(band_data), np.max(band_data) # Global normalization so that all bands are on the same scale for RGB visualization.
    if max_val - min_val == 0:
        return np.zeros_like(band_data)
    return (band_data - min_val) / (max_val - min_val)

def visualize_rgb(band_data: np.ndarray, index: int):
    # if validate_raster(path) and check_band_indices(path, expected_bands=RGB_BANDS):
    rgb_idx = list(map(lambda x: x-1, RGB_BANDS))
    band_data = band_data[rgb_idx]
    band_data = band_data.transpose(1, 2, 0)  # Transpose to (height, width, channels)
    band_data = normalize(band_data)

    fig, axes = plt.subplots(figsize=(15, 15))

    img = axes.imshow(band_data)
    axes.set_title(f"RGB Image of index_{index}")

    return fig, axes