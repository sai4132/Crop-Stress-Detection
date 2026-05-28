import matplotlib.pyplot as plt
from src.indices.vegetation import calculate_ndvi
import numpy as np

def visualize_ndvi(ndvi_data: np.ndarray, index: int):
    fig, axes = plt.subplots(figsize=(10, 10))
    img = axes.imshow(ndvi_data, cmap='RdYlGn')
    plt.colorbar(img, ax=axes, fraction=0.046, pad=0.04)
    axes.set_title(f"NDVI Image of index_{index}")  
    
    return fig, axes