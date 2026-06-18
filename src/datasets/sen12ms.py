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

with open(paths.CONFIG_DIR/"dataset.yaml", "r") as f:
    config = yaml.safe_load(f)

class SEN12MSDataset(Dataset):
    def __init__(
            self, 
            multi_spectral_dir: Path, 
            sar_dir: Path, 
            land_cover_dir: Path, 
            transform: callable = None, 
            filter_agricultural: bool = True, 
            sensors: Dict[str, list[int]] = None, 
            compute_ndvi: bool = False
            ):
        self.__multi_spectral_dir = multi_spectral_dir
        self.__sar_dir = sar_dir
        self.__land_cover_dir = land_cover_dir
        self.__transform = transform
        self.filter_agricultural = filter_agricultural
        self.sensors = sensors if sensors is not None else {sensor: config["sensor"]["bands"] for sensor in config["dataset"]["sensors"]}
        self.compute_ndvi = compute_ndvi
        self.samples = self.__load_samples()
    
    def __load_samples(self):
        file_paths = []
        if self.__multi_spectral_dir.exists():
            file_paths = sorted(list(self.__multi_spectral_dir.rglob("*.tif")))

        if not file_paths:
            raise FileNotFoundError(f"No .tif files found in {self.__multi_spectral_dir}. Please check the directory and ensure it contains the expected data.")

        samples = []
        for path in file_paths:
            try:
                sample = SEN12MSSample(path)
                if sample.s1_path.exists() and sample.lc_path.exists():
                    if self.filter_agricultural:
                        with rio.open(sample.lc_path) as lc_src:
                            lc_raster = io.load_bands(lc_src, config["lc"]["igbp_band"]).flatten()
                            agri_score = (np.isin(lc_raster, config["lc"]["agricultural_classes"])).sum() / lc_raster.size
                            if agri_score > config["agriculture"]["threshold"]:
                                samples.append(sample)
                    else:
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
                sample_item["lc"] = np.isin(io.load_bands(lc_src, config["lc"]["igbp_band"]), config["lc"]["agricultural_classes"]) if self.filter_agricultural else io.load_bands(lc_src, self.sensors["lc"])
                sample_item["lc_metadata"] = io.get_raster_metadata(lc_src)

        if self.__transform:
            sample_item = self.__transform(sample_item)
        return sample_item
    
if __name__ == "__main__":
    dataset = SEN12MSDataset(
        multi_spectral_dir=paths.MULTI_SPECTRAL_DIR,
        sar_dir=paths.SAR_DIR,
        land_cover_dir=paths.LAND_COVER_DIR,
        transform=transform,  # Use the defined transform for testing
        filter_agricultural=True,
        sensors={"s2": [2, 3, 4, 8], "s1": [1, 2], "lc": [1]}  # Test with a subset of bands
    )
    print(f"Dataset loaded with {len(dataset)} samples.")
    print(f"Dataset attributes: {dataset.__dict__.keys()}")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    for key, value in sample.items():
        if value is not None:
            print(f"{key} shape: {value.shape}, dtype: {value.dtype}")