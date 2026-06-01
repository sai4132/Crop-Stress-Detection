import torch

def normalize_z(tensor):
    for i in range(tensor.shape[0]):
        band = tensor[i]
        mean = band.mean()
        std = band.std()
        if std > 0:
            tensor[i] = (band - mean) / std
        else:
            tensor[i] *= 0
    return tensor

def normalize_minmax(tensor):
    for i in range(tensor.shape[0]):
        band = tensor[i]
        min_val = band.min()
        max_val = band.max()
        if max_val > min_val:
            tensor[i] = (band - min_val) / (max_val - min_val)
        else:
            tensor[i] *= 0
    return tensor