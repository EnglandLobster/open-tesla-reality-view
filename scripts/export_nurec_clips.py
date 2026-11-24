#!/usr/bin/env python3
"""Export NVIDIA PhysicalAI-AV-NuRec USDZ scenes into standardized Stage 0 DrivingClip directories.

Renders synchronized 6-camera surround views + metric depths + third-person chase/orbit views
(with 3D Ego-Car compositing) directly on GPU via our zero-compiler Triton 3DGS rasterizer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from open_reality_view.data.nurec_renderer import NuRecSceneRenderer


def _save_png(tensor_hwc: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(tensor_hwc * 255.0, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a NuRec 26.10 USDZ into a Stage 0 DrivingClip.")
    parser.add_argument("--usdz", required=True, help="Path to NuRec .usdz scene file.")
    parser.add_argument("--out-dir", required=True, help="Output clip directory.")
    parser.add_argument("--num-frames", type=int, default=17, help="Must satisfy 1 + 4*k.")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--start-sec", type=float, default=1.0)
    parser.add_argument("--src-height", type=int, default=224)
    parser.add_argument("--src-width", type=int, default=384)
    parser.add_argument("--tgt-height", type=int, default=480)
    parser.add_argument("--tgt-width", type=int, default=832)
    parser.add_argument("--orbit-azimuth-deg", type=float, default=0.0)
    parser.add_argument("--save-depths", action="store_true", default=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    renderer = NuRecSceneRenderer(args.usdz, device="cuda")
    print(f"Loaded NuRec scene '{renderer.sequence_id}' in {time.perf_counter() - t0:.2f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_names = list(NuRecSceneRenderer.DEFAULT_SURROUND_CAMERAS)

    src_intrinsics_all: list[list] = []
    src_c2e_all: list[list] = []
    tgt_intrinsics_all: list[list] = []
    tgt_c2e_all: list[list] = []

    t_render = time.perf_counter()
    for f_idx in range(args.num_frames):
        ts_us = (args.start_sec + f_idx / args.fps) * 1e6
        bundle = renderer.render_frame_bundle(
            ts_us,
            surround_cameras=tuple(cam_names),
            src_hw=(args.src_height, args.src_width),
            tgt_hw=(args.tgt_height, args.tgt_width),
            orbit_azimuth_deg=args.orbit_azimuth_deg,
        )

        for cam in cam_names:
            _save_png(bundle["surround_rgb"][cam].cpu().numpy(), out_dir / "surround" / cam / f"{f_idx:06d}.png")
            if args.save_depths:
                d_path = out_dir / "depths" / cam / f"{f_idx:06d}.npy"
                d_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(d_path, bundle["surround_depth"][cam].cpu().numpy().astype(np.float32))

        _save_png(bundle["chase_rgb"].cpu().numpy(), out_dir / "chase_gt" / f"{f_idx:06d}.png")
        src_intrinsics_all.append(bundle["src_intrinsics"].cpu().tolist())
        src_c2e_all.append(bundle["src_c2e"].cpu().tolist())
        tgt_intrinsics_all.append(bundle["tgt_intrinsics"].cpu().tolist())
        tgt_c2e_all.append(bundle["tgt_c2e"].cpu().tolist())

        if (f_idx + 1) % 5 == 0 or f_idx == args.num_frames - 1:
            print(f"  Rendered frame {f_idx + 1}/{args.num_frames} ({time.perf_counter() - t_render:.1f}s)")

    meta = {
        "clip_id": renderer.sequence_id,
        "camera_names": cam_names,
        "fps": args.fps,
        "ego_centroid": list(renderer.ego_centroid),
        "ego_dim_lwh": list(renderer.ego_dim_lwh),
        "src_intrinsics": src_intrinsics_all,
        "src_c2e": src_c2e_all,
        "tgt_intrinsics": tgt_intrinsics_all,
        "tgt_c2e": tgt_c2e_all,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"Exported {args.num_frames}-frame clip to {out_dir} in {time.perf_counter() - t_render:.2f}s")


if __name__ == "__main__":
    main()
