from pathlib import Path
from dataclasses import dataclass
from src.utils import paths

@dataclass
class PatchMetadata:
    year: str
    season: str
    sensor: str
    ROI: int
    patch: int

  
def parse_patch_path(path: Path):
    path = path.resolve().stem
    path_metadata = path.split("_")
    if len(path_metadata) !=5:
        raise ValueError(f"Unexpected patch path format: {path}. ROIs[YEAR]_[SEASON]_[MODALITY]_[ROI_ID]_[PATCH_ID]")
    
    return PatchMetadata(
        year=path_metadata[0][-4:],
        season=path_metadata[1],
        sensor=path_metadata[2],
        ROI=int(path_metadata[3]),
        patch=int(path_metadata[4][1:])
    )
    
def get_s1_path_from_s2_path(s2_path: Path, s1_dir: Path = paths.SAR_DIR):
    s2_sample_metadata = parse_patch_path(s2_path)
    s1_path = s1_dir/(f"s1_{s2_sample_metadata.ROI}/ROIs{s2_sample_metadata.year}_{s2_sample_metadata.season}_s1_{s2_sample_metadata.ROI}_p{s2_sample_metadata.patch}.tif")
    # print(f"Derived s1 path from s2 path: {s1_path}")
    return s1_path if s1_path.exists() else None

def get_lc_path_from_s2_path(s2_path: Path, lc_dir: Path = paths.LAND_COVER_DIR):
    s2_sample_metadata = parse_patch_path(s2_path)
    lc_path = lc_dir/(f"lc_{s2_sample_metadata.ROI}/ROIs{s2_sample_metadata.year}_{s2_sample_metadata.season}_lc_{s2_sample_metadata.ROI}_p{s2_sample_metadata.patch}.tif")
    return lc_path if lc_path.exists() else None

def get_sensor_path_from_metadata(metadata: PatchMetadata) -> tuple[Path]:
    s1_path = paths.SAR_DIR/(f"s1_{metadata.ROI}/ROIs{metadata.year}_{metadata.season}_s1_{metadata.ROI}_p{metadata.patch}.tif")
    s2_path = paths.MULTI_SPECTRAL_DIR/(f"s2_{metadata.ROI}/ROIs{metadata.year}_{metadata.season}_s2_{metadata.ROI}_p{metadata.patch}.tif")
    lc_path = paths.LAND_COVER_DIR/(f"lc_{metadata.ROI}/ROIs{metadata.year}_{metadata.season}_lc_{metadata.ROI}_p{metadata.patch}.tif")
    return s1_path, s2_path, lc_path