#!/usr/bin/env python3
"""Import a refiner's output frames into an exp_cosmos_transfer eval dir.

Frames are sorted by name, resized to the eval resolution and renamed
0000.png, 0001.png, ... so `exp_cosmos_transfer_refine.py eval` can score them
next to raw / difix / cosmos.

  python scripts/import_refiner_frames.py SRC_DIR METHOD_NAME \
      [--out-dir outputs/exp_cosmos_transfer/bb5389f1_f93] [--size 1280x704]
"""
import argparse
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path, help="directory with the refiner's frames")
ap.add_argument("method", help="eval method name, e.g. artifixer_1.3b_s0.1")
ap.add_argument("--out-dir", type=Path, default=Path("outputs/exp_cosmos_transfer/bb5389f1_f93"))
ap.add_argument("--size", default="1280x704")
args = ap.parse_args()

w, h = map(int, args.size.split("x"))
files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
n_ref = len(list((args.out_dir / "raw").glob("*.png")))
if len(files) != n_ref:
    raise SystemExit(f"{args.src}: {len(files)} frames, but raw/ has {n_ref}")
dst = args.out_dir / args.method
dst.mkdir(exist_ok=True)
for i, f in enumerate(files):
    Image.open(f).convert("RGB").resize((w, h), Image.LANCZOS).save(dst / f"{i:04d}.png")
print(f"imported {len(files)} frames -> {dst}")
