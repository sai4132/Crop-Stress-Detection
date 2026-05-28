import matplotlib.pyplot as plt
import math
import numpy as np
# from configs.visualization import RGB_BANDS
RGB_BANDS = [4, 3, 2]

def visualize_histogram(band_data: np.ndarray, index: int, bands: list[int] = RGB_BANDS):
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
        hist = ax.hist(band_img.flatten(), bins=50, color='blue', alpha=0.7)
        ax.set_title(f"Histogram of Band {band} of index_{index}")
        # plt.xlabel("Pixel Value")
        # plt.ylabel("Frequency")

    for j in range(num_bands, rows*cols):
        axes_flat[j].axis('off')
    plt.tight_layout()

    return fig, axes