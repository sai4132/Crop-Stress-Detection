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
        self.__filter_agricultural = filter_agricultural
        self.__sensors = sensors if sensors is not None else {sensor: config["sensor"]["bands"] for sensor in config["dataset"]["sensors"]}
        self.__compute_ndvi = compute_ndvi
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
                    if self.__filter_agricultural:
                        with rio.open(sample.lc_path) as lc_src:
                            lc_raster = io.load_bands(lc_src, config["lc"]["igbp_band"]).flatten()
                            agri_score = (np.isin(lc_raster, config["lc"]["agricultural_classes"])).sum() / lc_raster.size
                            if agri_score > config["lc"]["agricultural_threshold"]:
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

        s2_data, s1_data, lc_data = None, None, None
        s2_raster_metadata, s1_raster_metadata, lc_raster_metadata = None, None, None
        ndvi = None
        if "s2" in  self.__sensors.keys() or self.__compute_ndvi:
            with rio.open(sample.s2_path) as s2_src:
                s2_data = io.load_raster(s2_src)
                if self.__compute_ndvi:
                    ndvi = vegetation.compute_ndvi(s2_data)
                if "s2" not in self.__sensors.keys():
                    s2_data = None
                else:
                    s2_data_idx = list(map(lambda x: x-1, self.__sensors["s2"]))
                    s2_data = s2_data[s2_data_idx]
                    s2_raster_metadata = io.get_raster_metadata(s2_src)

        if "s1" in self.__sensors.keys():
            with rio.open(sample.s1_path) as s1_src:
                s1_data = io.load_bands(s1_src, self.__sensors["s1"])
                s1_raster_metadata = io.get_raster_metadata(s1_src)
        if "lc" in self.__sensors.keys():
            with rio.open(sample.lc_path) as lc_src:
                lc_data = io.load_bands(lc_src, config["lc"]["igbp_band"])
                lc_raster_metadata = io.get_raster_metadata(lc_src)

        sample_item = {
            "s2": s2_data,
            "s1": s1_data,
            "lc": lc_data,
            "ndvi": ndvi,
            "s2_metadata": s2_raster_metadata,
            "s1_metadata": s1_raster_metadata,
            "lc_metadata": lc_raster_metadata
        }

        if self.__transform:
            sample_item = self.__transform(sample_item)
        return sample_item