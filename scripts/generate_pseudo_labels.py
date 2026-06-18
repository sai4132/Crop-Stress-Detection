from src.datasets.sen12ms import SEN12MSDataset
from src.utils import paths
from src.preprocessing.transform import transform
import csv
import yaml

with open(paths.CONFIG_DIR/"train.yaml", "r") as f:
    train_config = yaml.safe_load(f)

def generate_ndvi_labels(dataset: SEN12MSDataset, threshold: dict = train_config["threshold"]):
    n_data = len(dataset)
    samples = dataset.samples

    headers = ["sample_id", "s2_path", "roi", "season", "year", "agri_fraction", "ndvi_mean", "ndvi_std", "stress_ratio", "label"]
    label_list = []

    for idx in range(n_data):
        sample = samples[idx]
        sample_data = dataset[idx]

        if sample_data["ndvi"] is None:
            raise ValueError(f"Sample at index {idx} is missing ndvi data. Please check the dataset for missing or corrupted data.")
        stress_mask = (sample_data["ndvi"] < threshold["ndvi_threshold"])
        agri_mask = sample_data["lc"][0]==1
        stress_ratio = (stress_mask[agri_mask]).mean()
        label = stress_ratio > threshold["stress_threshold"]

        label_list.append(
            {
                "sample_id": sample.sample_id,
                "s2_path": sample.s2_path,
                "roi": sample.metadata.ROI,
                "season": sample.metadata.season,
                "year": sample.metadata.year,
                "agri_fraction": float(agri_mask.mean()),
                "ndvi_mean": float(sample_data["ndvi"][agri_mask].mean()),
                "ndvi_std": float(sample_data["ndvi"][agri_mask].std()),
                "stress_ratio": float(stress_ratio),
                "label": bool(label)
            }
        )

    
    with open(paths.ALL_LABELS_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = headers)
        writer.writeheader()
        writer.writerows(label_list)

if __name__ == "__main__":
    dataset = SEN12MSDataset(
        multi_spectral_dir=paths.MULTI_SPECTRAL_DIR,
        sar_dir=paths.SAR_DIR,
        land_cover_dir=paths.LAND_COVER_DIR,
        transform=None,  # Use the defined transform for testing
        filter_agricultural=True,
        sensors={"lc": [1]},  # Test with a subset of bands
        compute_ndvi= True
    )

    generate_ndvi_labels(dataset=dataset)