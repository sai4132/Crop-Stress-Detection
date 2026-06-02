# Crop Stress Detection using SEN12MS

Early-stage geospatial ML engineering pipeline for crop stress detection using Sentinel-1 and Sentinel-2 satellite imagery from the SEN12MS dataset.

Current focus is on:

* robust raster ingestion,
* multimodal dataset engineering,
* geospatial metadata handling,
* visualization tooling,
* preprocessing infrastructure,
* tensor trustworthiness validation,
* operational batching and dataloader infrastructure.

Model training is intentionally deferred until data trustworthiness, preprocessing infrastructure, and batching reliability are stabilized.

---

# Project Goals

* Build modular geospatial ingestion utilities
* Create reusable preprocessing pipelines
* Develop visualization/debugging tooling for satellite imagery
* Support Sentinel-1 SAR and Sentinel-2 multispectral workflows
* Maintain memory-efficient processing on local hardware
* Build multimodal PyTorch dataset abstractions
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

# Current Engineering Features

Implemented:

## Geospatial IO Infrastructure

* Raster loading utilities
* Band-specific loading
* Metadata extraction
* Raster validation
* Semantic metadata parsing
* Path reconstruction utilities

## Visualization & Diagnostics

* RGB visualization
* NDVI visualization
* SAR visualization
* Land-cover visualization
* Histogram diagnostics
* Patch inspection tooling

## Dataset Engineering

* Lazy multimodal raster loading
* Aligned S1/S2/LC sample abstraction
* Agricultural filtering using MODIS IGBP labels
* Optional NDVI computation
* Configurable modality/channel selection
* Metadata-aware sample handling

## Preprocessing Infrastructure

* Tensor conversion
* Normalization transforms
* Percentile clipping
* Transform orchestration pipeline

## Validation & Batching

* Dataset validation tooling
* Tensor trustworthiness checks
* NaN/Inf diagnostics
* DataLoader integration
* Custom collate handling for metadata
* Batch loading benchmarking

---

# Operational Benchmark Results

Hardware:

* MacBook Air (Apple Silicon, 16GB Unified Memory)

Current best loading configuration:

```text
batch_size = 8
num_workers = 0
```

Observed performance:

| Metric                                | Value  |
| ------------------------------------- | ------ |
| Dataset initialization time           | ~27s   |
| Single sample load time               | ~0.02s |
| First batch iteration time            | ~0.13s |
| Approximate memory increase per batch | ~24 MB |

Observations:

* Multiprocessing overhead on macOS dominates current workload.
* `num_workers > 0` reduced performance for current raster sizes and preprocessing complexity.
* Current pipeline is lightweight and IO-efficient.

---

# Tech Stack

* Python
* PyTorch
* Rasterio
* NumPy
* Matplotlib
* PyYAML
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

Scripts are executed using:

```bash
PYTHONPATH=. uv run python scripts/<script_name>.py
```

Examples:

```bash
PYTHONPATH=. uv run python scripts/inspect_patch.py --random --ndvi
```

```bash
PYTHONPATH=. uv run python scripts/validate_dataset.py
```

```bash
PYTHONPATH=. uv run python scripts/benchmark_loading.py
```

---

# Repository Structure

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

# Current Development Philosophy

* Script-first engineering workflow
* Modular preprocessing and dataset abstractions
* Visualization treated as debugging infrastructure
* Emphasis on tensor trustworthiness before training
* Memory-conscious multimodal geospatial processing
* Production-style engineering patterns over notebook-centric experimentation
