from torch.utils.data import DataLoader
from src.datasets.sen12ms import SEN12MSDataset
from src.datasets.collate import collate_fn
from src.utils import paths
from src.preprocessing.transform import transform
from time import time
import argparse
import torch
from psutil import Process
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark dataset loading and dataloader performance")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for DataLoader")
    parser.add_argument("--s2_bands", nargs="+", type=int, default=[2, 3, 4, 8], help="List of Sentinel-2 bands to load")
    parser.add_argument("--s1_bands", nargs="+", type=int, default=[1, 2], help="List of Sentinel-1 bands to load")
    parser.add_argument("--compute-ndvi", action="store_true", help="Whether to compute NDVI from Sentinel-2 data")

    args = parser.parse_args()
    sensors = dict()
    if args.s2_bands:
        sensors["s2"] = args.s2_bands
    if args.s1_bands:
        sensors["s1"] = args.s1_bands

    start_time = time()

    dataset = SEN12MSDataset(
        multi_spectral_dir=paths.MULTI_SPECTRAL_DIR,
        sar_dir=paths.SAR_DIR,
        land_cover_dir=paths.LAND_COVER_DIR,
        transform=transform,  # Use the defined transform for testing
        filter_agricultural=True,
        sensors=sensors,  # Test with a subset of bands
        compute_ndvi=args.compute_ndvi
    )

    time_to_load_dataset = time() - start_time
    print(f"Time taken to load dataset with sensors {list(sensors.values())} and compute_ndvi={args.compute_ndvi}: {time_to_load_dataset:.2f} seconds")

    time_to_load_first_sample = time()
    first_sample = dataset[0]
    time_to_load_first_sample = time() - time_to_load_first_sample
    print(f"Time taken to load first sample: {time_to_load_first_sample:.2f} seconds")

    time_to_load_dataloader = time()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    time_to_load_dataloader = time() - time_to_load_dataloader
    print(f"Time taken to load dataloader with {args.num_workers} workers and batch_size={args.batch_size}: {time_to_load_dataloader:.2f} seconds")

    process = Process(os.getpid())
    memorymb = process.memory_info().rss/(1024**2)
    time_to_iterate_dataloader = time()
    for batch in dataloader:
        print(f"Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} with shape {value.shape} and dtype {value.dtype}", end="\n")
            else:
                print(f"{key} with shape {len(value)} and dtype {type(value)}", end="\n")
        break  # Check only the first batch for testing purposes
    time_to_iterate_dataloader = time() - time_to_iterate_dataloader
    print(f"Time taken to iterate through one batch of dataloader: {time_to_iterate_dataloader:.2f} seconds")
    memory_delta = process.memory_info().rss/(1024**2) - memorymb
    print(f"Memory Taken for this process to implement for one batch: {memory_delta:.2f} MB.")