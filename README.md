# Crop Stress Detection using SEN12MS

Early-stage geospatial ML engineering pipeline for crop stress detection using Sentinel-1 and Sentinel-2 satellite imagery from the SEN12MS dataset.

Current focus is on:

* robust raster ingestion,
* geospatial metadata handling,
* visualization tooling,
* preprocessing infrastructure,
* dataset engineering.

Model training is intentionally deferred until data trustworthiness and preprocessing infrastructure are stabilized.

---

# Project Goals

* Build modular geospatial ingestion utilities
* Create reusable preprocessing pipelines
* Develop visualization/debugging tooling for satellite imagery
* Support Sentinel-1 SAR and Sentinel-2 multispectral workflows
* Maintain memory-efficient processing on local hardware
* Follow production-style ML engineering practices

---

# Dataset

Dataset used:

* SEN12MS

Expected raw data structure:

```text
data/
├── raw/
│   ├── s1/
│   ├── s2/
│   └── lc/
```

Examples:

* `ROIs2017_winter_s1_42_p103.tif`
* `ROIs2017_winter_s2_42_p103.tif`
* `ROIs2017_winter_lc_42_p103.tif`

Raw data is intentionally excluded from git tracking.

---

# Tech Stack

* Python
* PyTorch
* Rasterio
* NumPy
* Matplotlib
* UV package manager

---

# Environment Setup (macOS)

## Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Install Python 3.11

```bash
brew install python@3.11
```

## Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

# Project Setup

Clone repository:

```bash
git clone <repo-url>
cd crop-stress-detection
```

Create virtual environment:

```bash
uv venv --python 3.11
```

Activate environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

---

# Running Scripts

Currently scripts are executed using:

```bash
PYTHONPATH=. uv run python scripts/<script_name>.py
```

Example:

```bash
PYTHONPATH=. uv run python scripts/test_paths.py
```

---

# Current Repository Structure

```text
crop-stress-detection/
│
├── configs/
├── data/
├── notebooks/
├── outputs/
├── scripts/
├── src/
│   ├── datasets/
│   ├── indices/
│   ├── preprocessing/
│   ├── utils/
│   └── visualization/
├── tests/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Current Development Phase

Completed:

* project setup
* UV environment management
* git infrastructure
* centralized path management
* raster IO utilities
* metadata extraction
* raster validation

Upcoming:

* patch inspection tooling
* RGB/NDVI visualization
* histogram diagnostics
* preprocessing pipeline
* PyTorch dataset abstraction

---

# Notes

* Repository follows script-first engineering, not notebook-centric workflows.
* Visualization is treated as debugging infrastructure.
* Emphasis is placed on modularity, reproducibility, and geospatial correctness.
