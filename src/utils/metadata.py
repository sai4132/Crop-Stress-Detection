from pathlib import Path
  
def parse_patch_path(path: Path):
    path = path.resolve().stem
    path_metadata = path.split("_")
    if len(path_metadata) !=5:
        raise ValueError(f"Unexpected patch path format: {path}. ROIs[YEAR]_[SEASON]_[MODALITY]_[ROI_ID]_[PATCH_ID]")
    
    return {
        "year" : path_metadata[0][-4:],
        "season": path_metadata[1],
        "sensor": path_metadata[2],
        "ROI" : int(path_metadata[3]),
        "patch": int(path_metadata[4][1:])
    }
    
def get_s1_path_from_s2_path(s2_path: Path, s1_dir: Path):
    s2_sample_metadata = parse_patch_path(s2_path)
    s1_path = s1_dir/(f"s1_{s2_sample_metadata['ROI']}/ROIs{s2_sample_metadata['year']}_{s2_sample_metadata['season']}_s1_{s2_sample_metadata['ROI']}_p{s2_sample_metadata['patch']}.tif")
    # print(f"Derived s1 path from s2 path: {s1_path}")
    return s1_path if s1_path.exists() else None

def get_lc_path_from_s2_path(s2_path: Path, lc_dir: Path):
    s2_sample_metadata = parse_patch_path(s2_path)
    lc_path = lc_dir/(f"lc_{s2_sample_metadata['ROI']}/ROIs{s2_sample_metadata['year']}_{s2_sample_metadata['season']}_lc_{s2_sample_metadata['ROI']}_p{s2_sample_metadata['patch']}.tif")
    return lc_path if lc_path.exists() else None