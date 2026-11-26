#!/usr/bin/env python3
"""Write the Wan-T5 prompt HDF5 for a scene exported by scripts/export_nurec_artifixer_scene.py.

Run INSIDE the ArtiFixer container (it needs ArtiFixer's captioning helpers and the Wan2.1 UMT5 encoder):

  python /workspace/ort/scripts/make_artifixer_prompt.py --scene-root /data/scenes/bb5389f1 --scene bb5389f1

Uses a hand-written scene caption instead of ArtiFixer's Qwen3-VL captioning pass (one fewer 8 GB model).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from data_processing.captioning.generate_captions import _write_wan_prompt_dataset, get_text_encoder_and_tokenizer
from data_processing.captioning.prompt_hdf5 import default_prompt_model_id
from model_training.prompt_formats import WAN_T5_PROMPT_FORMAT

CAPTIONS = {
    "bb5389f1": (
        "A camera moves smoothly forward along a wide suburban California arterial road at golden-hour sunset, "
        "seen from behind and above the road. The asphalt has a double yellow center line, white lane markings, a "
        "green bike lane and arrows. Tall eucalyptus trees and a low beige wall line the left side; palm trees, power "
        "lines, utility poles, houses and a parked silver sedan are on the right. A dark SUV drives ahead and a white "
        "car approaches in the opposite lane. Warm low sunlight, soft shadows, realistic textures."
    ),
}


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-root", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--caption", default=None)
    p.add_argument("--fps", type=float, default=16.0)
    p.add_argument("--max-sequence-length", type=int, default=512)
    args = p.parse_args()

    caption = args.caption or CAPTIONS[args.scene]
    split = json.loads((args.scene_root / "split.json").read_text())
    out_path = args.scene_root / split["test"][args.scene]["prompt_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anchors = np.array(json.loads((args.scene_root / "selected_indices.json").read_text()), dtype=np.int64)

    text_encoder, tokenizer = get_text_encoder_and_tokenizer(default_prompt_model_id(WAN_T5_PROMPT_FORMAT))
    with h5py.File(out_path, "w") as hf:
        _write_wan_prompt_dataset(
            hf, "caption_0000", caption, anchors, text_encoder, tokenizer, args.max_sequence_length, fps=args.fps
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
