import matplotlib.pyplot as plt
import math
import numpy as np

def log_transform(band_data: np.ndarray):
    return np.log(np.abs(band_data) + 1)

def visualize_sar(band_data: np.ndarray, index: int, bands: list[int] = [1,2]):
    band_idx = list(map(lambda x: x-1, bands))
    band_data = band_data[band_idx]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), squeeze=False)
    axes_flat = axes.flatten()

    for i in range(2):
        band = bands[i]
        band_img = band_data[i]
        ax = axes_flat[i]
        img = ax.imshow(band_img, cmap='gray')
        ax.set_title(f"{'VV' if band == 1 else 'VH'} polarisation of index_{index}")
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        band_img = log_transform(band_img)
        ax = axes_flat[i+2]
        img = ax.imshow(band_img, cmap='gray')
        ax.set_title(f"Log {'VV' if band == 1 else 'VH'} polarisation of index_{index}")
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    return fig, axes

def visualize_sar_histogram(band_data: np.ndarray, index: int, bands: list[int] = [1,2]):
    band_idx = list(map(lambda x: x-1, bands))
    band_data = band_data[band_idx]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), squeeze=False)
    axes_flat = axes.flatten()

    for i in range(2):
        band = bands[i]
        band_img = band_data[i]
        ax = axes_flat[i]
        img = ax.hist(band_img.flatten(), bins=50, color='blue', alpha=0.7)
        ax.set_title(f"{'VV' if band == 1 else 'VH'} polarisation of index_{index}")

        band_img = log_transform(band_img)
        ax = axes_flat[i+2]
        img = ax.hist(band_img.flatten(), bins=50, color='blue', alpha=0.7)
        ax.set_title(f"Log {'VV' if band == 1 else 'VH'} polarisation of index_{index}")

    plt.tight_layout()
    
    return fig, axes