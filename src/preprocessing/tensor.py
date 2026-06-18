import torch
from typing import Dict
import numpy as np

def to_tensor(sample: Dict) -> Dict:
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            sample[key] = torch.from_numpy(value).to(torch.float32) if key in ["s2", "s1", "ndvi"] else torch.from_numpy(value).to(torch.uint8)
        elif isinstance(value, int):
            sample[key] = torch.tensor(value, dtype=torch.long)
    return sample