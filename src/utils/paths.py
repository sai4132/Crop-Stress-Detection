from pathlib import Path

current_dir = Path(__file__).parent.resolve()

while not (current_dir / ".venv").exists() and current_dir != current_dir.parent:
    current_dir = current_dir.parent

PROJECT_ROOT = current_dir
RAW_DATA_DIR = PROJECT_ROOT/"data/raw/"
PROCESSED_DATA_DIR = PROJECT_ROOT/"data/processed/"
CONFIG_DIR = PROJECT_ROOT/"configs/"