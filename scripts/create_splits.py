from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from src.utils import paths
import yaml
from pathlib import Path

with open(paths.CONFIG_DIR/"train.yaml", "r") as f:
    train_config = yaml.safe_load(f)

def train_test_val_split(label_path: Path = paths.ALL_LABELS_PATH):
    if not label_path.is_file():
        raise FileNotFoundError(f"Labels at {label_path} are not found. Please generate labels before creating splits.")
    
    data = pd.read_csv(label_path)
    
    groups = data["roi"]
    y = data["label"]

    sgkf = StratifiedGroupKFold(
        n_splits=6,
        shuffle=True,
        random_state=11
    )

    folds = list(sgkf.split(data, y, groups=groups))
    val_idx, test_idx = folds[0][1], folds[1][1]
    train_idx = [idx for i in range(2,6) for idx in folds[i][1]]

    train_data, val_data, test_data = data.iloc[train_idx].copy(), data.iloc[val_idx].copy(), data.iloc[test_idx].copy()

    for key, value in {"train": train_data, "val": val_data, "test":test_data}.items():
        print(f"Diagnostics for {key} data:")
        print(f"Number of samples: {len(value)}")
        print(f"Stress ratio: {value['label'].mean()*100:.2f}%")
        print(f"ROIs count: {value['roi'].nunique()}")
        print(f"ROIs: {value['roi'].unique()}", end="\n\n")

    train_rois = set(train_data["roi"])
    val_rois = set(val_data["roi"])
    test_rois = set(test_data["roi"])

    assert len(train_rois & val_rois) == 0
    assert len(train_rois & test_rois) == 0
    assert len(val_rois & test_rois) == 0

    print("\nROI leakage check passed.")

    train_data.to_csv(paths.TRAIN_LABELS_PATH, index=False)
    val_data.to_csv(paths.VAL_LABELS_PATH, index=False)
    test_data.to_csv(paths.TEST_LABELS_PATH, index=False)

if __name__=="__main__":
    train_test_val_split()