from torch.utils.data import Dataset
import torch
from src.datasets.sen12ms import SEN12MSDataset
from src.preprocessing.transform import transform
from src.utils import paths

def validate_dataset(dataset: Dataset):
    expected_keys = [sensor for sensor in dataset.sensors.keys()]+[f"{sensor}_metadata" for sensor in dataset.sensors.keys()]+(["ndvi"] if dataset.compute_ndvi else [])
    print(f"Validating dataset with {len(dataset)} samples...")
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is None:
            raise ValueError(f"Sample at index {i} is None. Please check the dataset for missing or corrupted data.")
        if not all(key in sample for key in expected_keys):
            raise ValueError(f"Sample at index {i} is missing required keys. Please check the dataset for incomplete data.")
        if sample["s2"] is not None:
            if not isinstance(sample["s2"], torch.Tensor):
                raise TypeError(f"Sample at index {i} is not a Torch tensor. Please check the dataset for incorrect data types.")
            if sample["s2"].dtype != torch.float32:
                raise TypeError(f"Sample at index {i} has incorrect data type for s2. Expected torch.float32, got {sample['s2'].dtype}. Please check the dataset for incorrect data types.")
            if sample["s2"].shape[0] != len(dataset.sensors["s2"]):
                raise ValueError(f"Sample at index {i} has incorrect number of channels for s2. Expected {len(dataset.sensors['s2'])}, got {sample['s2'].shape[0]}. Please check the dataset for incorrect data shapes.")
            if sample["s2"].shape[1] != sample["s2"].shape[2]:
                raise ValueError(f"Sample at index {i} has non-square spatial dimensions for s2. Please check the dataset for incorrect data shapes.")
            if torch.isnan(sample["s2"]).any():
                raise ValueError(f"Sample at index {i} contains NaN values in s2 data. Please check the dataset for missing or corrupted data.")
            if torch.isinf(sample["s2"]).any():
                raise ValueError(f"Sample at index {i} contains Inf values in s2 data. Please check the dataset for missing or corrupted data.")
            # if (sample["s2"].mean()!=0.0) or (sample["s2"].std() != 1.0):
            #     raise ValueError(f"Sample at index {i} contains out-of-range values in s2 data. Expected values with mean 0 and std 1, got mean {sample['s2'].mean()} and std {sample['s2'].std()}. Please check the dataset for incorrect data values.")
        else:
            if dataset.sensors["s2"] is not None:
                raise ValueError(f"Sample at index {i} is missing s2 data. Please check the dataset for missing or corrupted data.")
        if sample["s1"] is not None:
            if not isinstance(sample["s1"], torch.Tensor):
                raise TypeError(f"Sample at index {i} is not a Torch tensor. Please check the dataset for incorrect data types.")
            if sample["s1"].dtype != torch.float32:
                raise TypeError(f"Sample at index {i} has incorrect data type for s1. Expected torch.float32, got {sample['s1'].dtype}. Please check the dataset for incorrect data types.")
            if sample["s1"].shape[0] != len(dataset.sensors["s1"]):
                raise ValueError(f"Sample at index {i} has incorrect number of channels for s1. Expected {len(dataset.sensors['s1'])}, got {sample['s1'].shape[0]}. Please check the dataset for incorrect data shapes.")
            if sample["s1"].shape[1] != sample["s1"].shape[2]:
                raise ValueError(f"Sample at index {i} has non-square spatial dimensions for s1. Please check the dataset for incorrect data shapes.")
            if torch.isnan(sample["s1"]).any():
                raise ValueError(f"Sample at index {i} contains NaN values in s1 data. Please check the dataset for missing or corrupted data.")
            if torch.isinf(sample["s1"]).any():
                raise ValueError(f"Sample at index {i} contains Inf values in s1 data. Please check the dataset for missing or corrupted data.")
            # if (sample["s1"].mean() != 0.0) or (sample["s1"].std() != 1.0):
            #     raise ValueError(f"Sample at index {i} contains out-of-range values in s1 data. Expected values with mean 0 and std 1, got mean {sample['s1'].mean()} and std {sample['s1'].std()}. Please check the dataset for incorrect data values.")
        else:
            if dataset.sensors["s1"] is not None:
                raise ValueError(f"Sample at index {i} is missing s1 data. Please check the dataset for missing or corrupted data.")
        if sample["lc"] is not None:
            if not isinstance(sample["lc"], torch.Tensor):
                raise TypeError(f"Sample at index {i} is not a Torch tensor. Please check the dataset for incorrect data types.")
            if sample["lc"].dtype != torch.uint8:
                raise TypeError(f"Sample at index {i} has incorrect data type for lc. Expected torch.uint8, got {sample['lc'].dtype}. Please check the dataset for incorrect data types.")
            if ((sample["lc"].shape[0] != 1) and dataset.filter_agricultural) or ((sample["lc"].shape[0] != len(dataset.sensors["lc"])) and not dataset.filter_agricultural):
                raise ValueError(f"Sample at index {i} has incorrect number of channels for lc. Expected {1 if dataset.filter_agricultural else len(dataset.sensors['lc'])}, got {sample['lc'].shape[0]}. Please check the dataset for incorrect data shapes.")
            if sample["lc"].shape[1] != sample["lc"].shape[2]:
                raise ValueError(f"Sample at index {i} has non-square spatial dimensions for lc. Please check the dataset for incorrect data shapes.")
            if torch.isnan(sample["lc"]).any():
                raise ValueError(f"Sample at index {i} contains NaN values in lc data. Please check the dataset for missing or corrupted data.")
            if torch.isinf(sample["lc"]).any():
                raise ValueError(f"Sample at index {i} contains Inf values in lc data. Please check the dataset for missing or corrupted data.")
            if dataset.filter_agricultural and not torch.isin(sample["lc"], torch.tensor([0, 1], dtype=torch.uint8)).all():
                raise ValueError(f"Sample at index {i} contains invalid values in lc data. Expected binary values (0 and 1) for agricultural filtering, got values outside this range. Please check the dataset for incorrect data values.")
        else:
            if dataset.sensors["lc"] is not None:
                raise ValueError(f"Sample at index {i} is missing lc data. Please check the dataset for missing or corrupted data.")
        if sample["ndvi"] is not None:
            if not isinstance(sample["ndvi"], torch.Tensor):
                raise TypeError(f"Sample at index {i} is not a Torch tensor. Please check the dataset for incorrect data types.")
            if sample["ndvi"].dtype != torch.float32:
                raise TypeError(f"Sample at index {i} has incorrect data type for ndvi. Expected torch.float32, got {sample['ndvi'].dtype}. Please check the dataset for incorrect data types.")
            if sample["ndvi"].shape[0] != 1:
                raise ValueError(f"Sample at index {i} has incorrect number of channels for ndvi. Expected 1, got {sample['ndvi'].shape[0]}. Please check the dataset for incorrect data shapes.")
            if sample["ndvi"].shape[1] != sample["ndvi"].shape[2]:
                raise ValueError(f"Sample at index {i} has non-square spatial dimensions for ndvi. Please check the dataset for incorrect data shapes.")
            if torch.isnan(sample["ndvi"]).any():
                raise ValueError(f"Sample at index {i} contains NaN values in ndvi data. Please check the dataset for missing or corrupted data.")
            if torch.isinf(sample["ndvi"]).any():
                raise ValueError(f"Sample at index {i} contains Inf values in ndvi data. Please check the dataset for missing or corrupted data.")
            if (sample["ndvi"].max() > 1.0) or (sample["ndvi"].min() < -1.0):
                raise ValueError(f"Sample at index {i} contains out-of-range values in ndvi data. Expected values in [-1, 1], got min {sample['ndvi'].min()} and max {sample['ndvi'].max()}. Please check the dataset for incorrect data values.")
        else:
            if dataset.compute_ndvi:
                raise ValueError(f"Sample at index {i} is missing ndvi data. Please check the dataset for missing or corrupted data.")
    print("Dataset validation passed successfully!")

if __name__ == "__main__":
    dataset = SEN12MSDataset(
        manifest_path=paths.ALL_LABELS_PATH,
        transform=transform,  # Use the defined transform for testing
        sensors={"s2": [2, 3, 4, 8], "s1": [1, 2], "lc": [1]}  # Test with a subset of bands
    )
    validate_dataset(dataset)