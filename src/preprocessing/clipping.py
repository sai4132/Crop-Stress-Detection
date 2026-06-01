import torch

def percentile_clipping(tensor, lower_percentile=2, upper_percentile=98):
    for i in range(tensor.shape[0]):
        band = tensor[i]
        lower_value = torch.quantile(band, lower_percentile / 100.0)
        upper_value = torch.quantile(band, upper_percentile / 100.0)
        tensor[i] = torch.clamp(band, min=lower_value, max=upper_value)
    return tensor