from pathlib import Path
import yaml

current_dir = Path(__file__).parent.resolve()

while not (current_dir / ".venv").exists() and current_dir != current_dir.parent:
    current_dir = current_dir.parent

PROJECT_ROOT = current_dir

with open(PROJECT_ROOT / "configs/paths.yaml", "r") as f:
    paths_config = yaml.safe_load(f)

DATA_DIR = PROJECT_ROOT/paths_config["DATA_DIR"]["ROOT"]
RAW_DATA_DIR = PROJECT_ROOT/paths_config["DATA_DIR"]["RAW"]
PROCESSED_DATA_DIR = PROJECT_ROOT/paths_config["DATA_DIR"]["PROCESSED"]
CACHE_DATA_DIR = PROJECT_ROOT/paths_config["DATA_DIR"]["CACHE"]
CONFIG_DIR = PROJECT_ROOT/paths_config["CONFIG_DIR"]
MULTI_SPECTRAL_DIR = RAW_DATA_DIR/paths_config["SENSORS"]["MULTI_SPECTRAL"]
SAR_DIR = RAW_DATA_DIR/paths_config["SENSORS"]["SAR"]
LAND_COVER_DIR = RAW_DATA_DIR/paths_config["SENSORS"]["LAND_COVER"]
OUTPUT_DIR = PROJECT_ROOT/paths_config["OUTPUT_DIR"]["ROOT"]
INSPECTION_OUTPUT_DIR = PROJECT_ROOT/paths_config["OUTPUT_DIR"]["INSPECTION"]