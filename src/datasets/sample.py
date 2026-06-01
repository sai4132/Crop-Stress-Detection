from dataclasses import dataclass
from pathlib import Path
from src.utils.metadata import parse_patch_path, get_s1_path_from_s2_path, get_lc_path_from_s2_path, PatchMetadata

class SEN12MSSample:
    def __init__(self, s2_sample_path: Path):
        if not s2_sample_path.exists() or not s2_sample_path.is_file():
            raise ValueError("Invalid s2_sample_path")

        self.s2_path: Path = s2_sample_path
        self.s1_path: Path = get_s1_path_from_s2_path(s2_sample_path)
        self.lc_path: Path = get_lc_path_from_s2_path(s2_sample_path)
        self.metadata: PatchMetadata = parse_patch_path(s2_sample_path)

    @property
    def sample_id(self) -> str:
        return f"{self.metadata.year}_{self.metadata.season}_{self.metadata.ROI}_p{self.metadata.patch}"