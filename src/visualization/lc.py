import matplotlib.pyplot as plt
import math
import numpy as np

def visualize_land_cover(band_data: np.ndarray, index: int, bands: list[int] = [1,2,3,4]):
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
        unique_classes = np.unique(band_img)
        ax = axes_flat[i]
        img = ax.imshow(band_img, cmap='tab10')
        ax.set_title(f"Land Cover Classification in band {band} of index_{index}")
        plt.colorbar(img, ax=ax, ticks=unique_classes, fraction=0.046, pad=0.04)
    
    for j in range(num_bands, rows*cols):
        axes_flat[j].axis('off')
    plt.tight_layout()
    
    return fig, axes

def visualize_land_cover_histogram(band_data: np.ndarray, index: int, bands: list[int] = [1,2,3,4]):
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
        unique_classes, class_counts = np.unique(band_img, return_counts=True)
        ax = axes_flat[i]
        img = ax.bar(unique_classes, class_counts, color='blue', alpha=0.7)
        ax.set_title(f"Land Cover Classification in band {band} of index_{index}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Frequency")
    for j in range(num_bands, rows*cols):
        axes_flat[j].axis('off')
    plt.tight_layout()
    
    return fig, axes