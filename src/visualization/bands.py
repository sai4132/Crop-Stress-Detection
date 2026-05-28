import matplotlib.pyplot as plt
import math
import numpy as np

def visualize_bands(band_data: np.ndarray, bands: list[int], index: int):
    # if validate_raster(path) and check_band_indices(path, expected_bands=bands):
    band_idx = list(map(lambda x: x-1, bands))
    band_data = band_data[band_idx]
    num_bands = len(bands)
    cols = math.ceil(math.sqrt(num_bands))
    rows = math.ceil(num_bands / cols)
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5*cols, 5*rows), squeeze=False)
    axes_flat = axes.flatten()

    for i in range(num_bands):
        band = bands[i]
        band_img = band_data[i]
        ax = axes_flat[i]
        img = ax.imshow(band_img, cmap='gray')
        ax.set_title(f"Band {band} of index_{index}")
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    
    for j in range(num_bands, rows*cols):
        axes_flat[j].axis('off')
    plt.tight_layout()
    
    return fig, axes