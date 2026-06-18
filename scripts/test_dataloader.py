import torch
from torch.utils.data import DataLoader, Dataset
from src.datasets.sen12ms import SEN12MSDataset
from src.datasets.collate import collate_fn
from src.utils import paths
from src.preprocessing.transform import transform

def test_dataloader(dataset: Dataset, batch_size: int = 4, shuffle: bool = False, num_workers: int = 0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    print(f"Testing DataLoader with batch size {batch_size} and shuffle={shuffle}")
    expected_keys = [sensor for sensor in dataset.sensors.keys()]+[f"{sensor}_metadata" for sensor in dataset.sensors.keys()]+(["ndvi"] if dataset.compute_ndvi else [])
    for batch in dataloader:
        if batch is None:
            raise ValueError("Batch is None. Please check the DataLoader for issues with data loading.")
        if not isinstance(batch, dict):
            raise TypeError(f"Batch is not a dictionary. Please check the DataLoader for incorrect data types. Got {type(batch)}")
        print(f"Batch keys: {list(batch.keys())}")
        if not all(key in batch for key in expected_keys):
            raise ValueError(f"Batch is missing required keys. Please check the DataLoader for incomplete data. Missing keys: {[key for key in expected_keys if key not in batch]}")
        print("batch diagnostics:\n")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} with shape {value.shape} and dtype {value.dtype}", end="\n")
            else:
                print(f"{key} with shape {len(value)} and dtype {type(value)}", end="\n")
        print("Batch validation passed.")
        break  # Check only the first batch for testing purposes

if __name__ == "__main__":
    dataset = SEN12MSDataset(
        manifest_path=paths.ALL_LABELS_PATH,
        transform=transform,  # Use the defined transform for testing
        sensors={"s2": [2, 3, 4, 8], "s1": [1, 2], "lc": [1]}  # Test with a subset of bands
    )
    test_dataloader(dataset, batch_size=4, shuffle=False, num_workers=4)