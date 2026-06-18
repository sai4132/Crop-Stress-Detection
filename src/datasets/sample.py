from pathlib import Path
from src.utils.metadata import parse_patch_path, get_sensor_path_from_metadata, PatchMetadata

class SEN12MSSample:
    def __init__(self, sample_path: Path):
        if not sample_path.exists() or not sample_path.is_file():
            raise ValueError("Invalid sample_path")

        self.metadata: PatchMetadata = parse_patch_path(sample_path)
        self.s1_path, self.s2_path, self.lc_path = get_sensor_path_from_metadata(self.metadata)

    @property
    def sample_id(self) -> str:
        return f"{self.metadata.year}_{self.metadata.season}_{self.metadata.ROI}_p{self.metadata.patch}"