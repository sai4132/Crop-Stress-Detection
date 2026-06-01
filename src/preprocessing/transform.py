from src.preprocessing import clipping, normalize, tensor
from typing import Dict

def transform(sample: Dict) -> Dict:
    sample = tensor.to_tensor(sample)
    if sample["s2"] is not None:
        sample["s2"] = normalize.normalize_z(sample["s2"])
    if sample["s1"] is not None:
        sample["s1"] = clipping.percentile_clipping(sample["s1"])
    return sample