from src.utils import paths, io
from src.indices import vegetation
from src.datasets.sample import SEN12MSSample
from src.preprocessing.transform import transform
import csv
import yaml
from pathlib import Path
import rasterio as rio
import numpy as np

with open(paths.CONFIG_DIR/"train.yaml", "r") as f:
    train_config = yaml.safe_load(f)

def generate_ndvi_labels(data_dir: Path = paths.MULTI_SPECTRAL_DIR, threshold: dict = train_config["threshold"], filter_agricultural: bool = True):
    file_paths = []
    if data_dir.exists():
        file_paths = sorted(list(data_dir.rglob("*.tif")))

    if not file_paths:
        raise FileNotFoundError(f"No .tif files found in {data_dir}. Please check the directory and ensure it contains the expected data.")

    headers = ["sample_id", "s2_path", "roi", "season", "year", "agri_fraction", "ndvi_mean", "ndvi_std", "stress_ratio", "label"]
    label_list = []

    for file_path in file_paths:
        sample = SEN12MSSample(file_path)

        with rio.open(sample.lc_path) as lc_src:
            agri_mask = np.isin(io.load_bands(lc_src, train_config["lc"]["igbp_band"]), train_config["lc"]["agricultural_classes"])[0]

        if filter_agricultural:
            agri_score = agri_mask.mean()
            if agri_score <= train_config["agriculture"]["threshold"]:
                continue
        
        with rio.open(sample.s2_path) as raster:
            raster_data = io.load_raster(raster)
            ndvi_data = vegetation.calculate_ndvi(raster_data)
        stress_mask = (ndvi_data < threshold["ndvi_threshold"])

        stress_ratio = (stress_mask[agri_mask]).mean()
        label = stress_ratio > threshold["stress_threshold"]

        label_list.append(
            {
                "sample_id": sample.sample_id,
                "s2_path": str(sample.s2_path),
                "roi": sample.metadata.ROI,
                "season": sample.metadata.season,
                "year": sample.metadata.year,
                "agri_fraction": float(agri_mask.mean()),
                "ndvi_mean": float(ndvi_data[agri_mask].mean()),
                "ndvi_std": float(ndvi_data[agri_mask].std()),
                "stress_ratio": float(stress_ratio),
                "label": bool(label)
            }
        )

    
    with open(paths.ALL_LABELS_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = headers)
        writer.writeheader()
        writer.writerows(label_list)

if __name__ == "__main__":
    generate_ndvi_labels()