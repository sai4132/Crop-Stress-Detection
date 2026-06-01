import torch
from typing import Dict
import numpy as np

def to_tensor(sample: Dict) -> Dict:
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            sample[key] = torch.from_numpy(value).dtype(torch.float32) if key in ["s2", "s1", "ndvi"] else torch.from_numpy(value).dtype(torch.uint8)
    return sample