from pathlib import Path
from src.utils import paths
import pandas as pd

def analyze_labels(label_file: Path):
    if not label_file.is_file:
        raise FileNotFoundError(f"The label file at path {label_file} doesn't exist.")
    label_data = pd.read_csv(label_file)
    print("The label data diagnostics are: \n")
    print(f'Total number of samples: {len(label_data)}\n')
    print(f'Percentage of stress croplands: {label_data["label"].mean()*100:.2f}%\n')
    print(f'Percentage of non-stress croplands: {100.00 - label_data["label"].mean()*100:.2f}%\n')
    for i in range(0, 110, 20):
        pct = i/100.0
        if pct==0.0:
            print(f'Minimum NDVI Mean: {label_data["ndvi_mean"].min():.2f}, \t Minimum NDVI STD: {label_data["ndvi_std"].min():.2f}, \t Minimum stress ratio: {label_data["stress_ratio"].min():.2f}', end="\n")
        elif pct==1.0:
            print(f'Maximum NDVI Mean: {label_data["ndvi_mean"].max():.2f}, \t Maximum NDVI STD: {label_data["ndvi_std"].max():.2f}, \t Maximum stress ratio: {label_data["stress_ratio"].max():.2f}', end="\n")
        else:
            print(f'{pct*100}% croplands are below NDVI Mean: {label_data["ndvi_mean"].quantile(pct):.2f}, \t NDVI STD: {label_data["ndvi_std"].quantile(pct):.2f}, \t stress ratio: {label_data["stress_ratio"].quantile(pct):.2f}', end="\n")
    
    unique_rois = list(label_data["roi"].unique())
    print(f"Total number of unique ROIs: {len(unique_rois)}\n")

    for roi in unique_rois:
        roi_label_data = label_data[label_data["roi"]==roi]
        print(f"Number of samples in ROI {roi}: {len(roi_label_data)}", end="\n")
        print(f"Percentage of stressed crops in ROI {roi}: {roi_label_data['label'].mean()*100:.2f}%", end="\n\n")


if __name__ == "__main__":
    analyze_labels(paths.ALL_LABELS_PATH)