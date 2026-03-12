#!/usr/bin/env python3
"""
Download and optionally downsample a Robomimic dataset.

Supported datasets (task × type):
  Tasks : lift, can, square, transport
  Types : ph (proficient-human), mh (mixed-human), mg (machine-generated)

Downloads the HDF5 image file from the official Robomimic hosting.

Usage:
    # Smallest dataset to get started (~300 MB)
    python scripts/download_dataset.py --task lift --type ph

    # Downsample images to 64×64 and keep only 200 demos
    python scripts/download_dataset.py --task lift --type ph \
        --img_size 64 --max_demos 200

    # Can task, proficient-human
    python scripts/download_dataset.py --task can --type ph
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# ── Official Robomimic download URLs ──────────────────────────────────────────
BASE_URL = "http://downloads.cs.stanford.edu/downloads/rt_benchmark"
DATASET_URLS = {
    ("lift", "ph"): "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5/lift/ph/demo_v15.hdf5?download=true",
    ("lift", "mh"): "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5/lift/mh/demo_v15.hdf5?download=true",
    ("can",  "ph"): "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5/can/ph/demo_v15.hdf5?download=true",
    ("can",  "mh"): "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5/can/mh/demo_v15.hdf5?download=true",
    ("square","ph"): "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5/square/ph/demo_v15.hdf5?download=true",
    ("transport","ph"): "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5/transport/ph/demo_v15.hdf5?download=true",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",      type=str, default="lift",
                   choices=["lift","can","square","transport"])
    p.add_argument("--type",      type=str, default="ph",
                   choices=["ph","mh","mg"])
    p.add_argument("--output_dir",type=str, default="data/robomimic")
    p.add_argument("--img_size",  type=int, default=84,
                   help="Resize images to img_size × img_size (default 84)")
    p.add_argument("--max_demos", type=int, default=None,
                   help="Keep only the first N demos after downloading")
    return p.parse_args()


# ── Download helper ────────────────────────────────────────────────────────────

class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


# ── Downsampling ──────────────────────────────────────────────────────────────

def downsample_dataset(src: Path, dst: Path, img_size: int, max_demos: int | None) -> None:
    """
    Create a new HDF5 file with:
      - images resized to (img_size, img_size)
      - at most max_demos demonstrations
    Preserves the Robomimic HDF5 schema.
    """
    try:
        import cv2
    except ImportError:
        print("opencv-python needed for resizing. Install with: pip install opencv-python")
        sys.exit(1)

    print(f"\nDownsampling → {dst}  (img_size={img_size}, max_demos={max_demos})")

    with h5py.File(src, "r") as src_f, h5py.File(dst, "w") as dst_f:
        # Copy top-level attrs and mask
        for k, v in src_f.attrs.items():
            dst_f.attrs[k] = v
        if "mask" in src_f:
            src_f.copy("mask", dst_f)

        data_grp = dst_f.create_group("data")

        demo_keys = sorted(src_f["data"].keys())
        if max_demos:
            demo_keys = demo_keys[:max_demos]

        for dk in tqdm(demo_keys, desc="Resampling demos"):
            src_demo = src_f["data"][dk]
            dst_demo = data_grp.create_group(dk)

            # Copy attrs
            for k, v in src_demo.attrs.items():
                dst_demo.attrs[k] = v

            # Copy non-image datasets verbatim
            for key in ("actions", "rewards", "dones", "states"):
                if key in src_demo:
                    src_demo.copy(key, dst_demo)

            # Resize images in obs
            obs_grp = dst_demo.create_group("obs")
            for obs_key in src_demo["obs"]:
                data = src_demo["obs"][obs_key][:]
                if data.ndim == 4 and data.shape[-1] == 3:
                    # Image: (T, H, W, 3) uint8
                    T = data.shape[0]
                    resized = np.stack([
                        cv2.resize(data[t], (img_size, img_size),
                                   interpolation=cv2.INTER_AREA)
                        for t in range(T)
                    ])  # (T, img_size, img_size, 3)
                    obs_grp.create_dataset(obs_key, data=resized,
                                           dtype=np.uint8, compression="gzip")
                else:
                    src_demo["obs"].copy(obs_key, obs_grp)

    size_mb = dst.stat().st_size / 1024**2
    print(f"✓ Saved downsampled dataset: {dst}  ({size_mb:.1f} MB)")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = (args.task, args.type)
    if key not in DATASET_URLS:
        print(f"No URL for {key}. Available: {list(DATASET_URLS.keys())}")
        sys.exit(1)

    url = DATASET_URLS[key]
    raw_path = out_dir / f"{args.task}_{args.type}_raw.hdf5"
    final_path = out_dir / f"{args.task}_{args.type}_{args.img_size}px.hdf5"

    # Download
    if not raw_path.exists():
        download_file(url, raw_path)
    else:
        print(f"Raw file already exists: {raw_path}")

    # Downsample / copy
    if args.img_size != 84 or args.max_demos is not None:
        downsample_dataset(raw_path, final_path, args.img_size, args.max_demos)
    else:
        if not final_path.exists():
            shutil.copy2(raw_path, final_path)
        print(f"✓ Dataset ready: {final_path}")

    print(f"\nDataset path to use in training:\n  {final_path}")
    print(f"\nQuick start:\n  python scripts/train_visual.py --data {final_path}")


if __name__ == "__main__":
    main()
