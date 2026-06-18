from src.datasets.sample import SEN12MSSample
from src.utils import io, metadata, paths
from src.indices import vegetation
from pathlib import Path
import rasterio as rio
import torch
from torch.utils.data import Dataset
from typing import Dict
import yaml
import numpy as np
from src.preprocessing.transform import transform
import csv

with open(paths.CONFIG_DIR/"dataset.yaml", "r") as f:
    config = yaml.safe_load(f)

class SEN12MSDataset(Dataset):
    def __init__(
            self, 
            manifest_path: Path = paths.ALL_LABELS_PATH,
            transform: callable = None, 
            sensors: Dict[str, list[int]] = None, 
            compute_ndvi: bool = False
            ):
        self.__manifest_path = manifest_path
        self.__transform = transform
        self.sensors = sensors if sensors is not None else {sensor: config["sensor"]["bands"] for sensor in config["dataset"]["sensors"]}
        self.compute_ndvi = compute_ndvi
        self.samples = self.__load_samples()
    
    def __load_samples(self):
        if not self.__manifest_path.is_file():
            raise FileNotFoundError(f"No manifest file found at {self.__manifest_path}. Please check the directory and ensure it contains the expected data.")
        with open(self.__manifest_path, "r") as f:
            dict_reader = csv.DictReader(f)
            
            file_paths = []
            for row in dict_reader:
                file_paths.append(row)

        if not file_paths:
            raise ValueError(f"No file paths found in the manifest file: {self.__manifest_path}. Generate labels or splits to populate the file.")

        samples = []
        for path in file_paths:
            try:
                sample = SEN12MSSample(Path(path["s2_path"]))
                if sample.s1_path.exists() and sample.lc_path.exists():
                    sample.label = int(path["label"]=='True')
                    samples.append(sample)
            except Exception as e:
                print(f"Error loading sample from {path}: {e}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx<0 or idx>=len(self.samples):
            raise IndexError(f"Index {idx} out of bounds")
        
        sample = self.samples[idx]

        sample_item = dict()
        if "s2" in  self.sensors.keys() or self.compute_ndvi:
            with rio.open(sample.s2_path) as s2_src:
                s2_data = io.load_raster(s2_src)
                if self.compute_ndvi:
                    sample_item["ndvi"] = vegetation.calculate_ndvi(s2_data)
                if "s2" in self.sensors.keys():
                    s2_data_idx = list(map(lambda x: x-1, self.sensors["s2"]))
                    sample_item["s2"] = s2_data[s2_data_idx]
                    sample_item["s2_metadata"] = io.get_raster_metadata(s2_src)

        if "s1" in self.sensors.keys():
            with rio.open(sample.s1_path) as s1_src:
                sample_item["s1"] = io.load_bands(s1_src, self.sensors["s1"])
                sample_item["s1_metadata"] = io.get_raster_metadata(s1_src)
        if "lc" in self.sensors.keys():
            with rio.open(sample.lc_path) as lc_src:
                sample_item["lc"] = io.load_bands(lc_src, self.sensors["lc"])
                sample_item["lc_metadata"] = io.get_raster_metadata(lc_src)
        sample_item["label"] = sample.label
        if self.__transform:
            sample_item = self.__transform(sample_item)
        return sample_item
    
if __name__ == "__main__":
    dataset = SEN12MSDataset(
        manifest_path=paths.ALL_LABELS_PATH,
        transform=transform,  # Use the defined transform for testing
        sensors={"s2": [2, 3, 4, 8], "s1": [1, 2], "lc": [1]}  # Test with a subset of bands
    )
    print(f"Dataset loaded with {len(dataset)} samples.")
    print(f"Dataset attributes: {dataset.__dict__.keys()}")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    for key, value in sample.items():
        if value is not None:
            print(f"{key} shape: {value.shape}, dtype: {value.dtype}")