from src.preprocessing import clipping, normalize, tensor
from typing import Dict

def transform(sample: Dict) -> Dict:
    sample = tensor.to_tensor(sample)
    if "s2" in sample and sample["s2"] is not None:
        sample["s2"] = normalize.normalize_z(sample["s2"])
    if "s1" in sample and sample["s1"] is not None:
        sample["s1"] = clipping.percentile_clipping(sample["s1"])
        sample["s1"] = normalize.normalize_z(sample["s1"])
    return sample