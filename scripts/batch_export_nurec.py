#!/usr/bin/env python3
"""Batch-export top-PSNR NVIDIA PhysicalAI-AV-NuRec 26.10 scenes into Stage 0 DrivingClips.

Reads ``benchmark_26.04_vs_26.10.csv``, selects diverse high-PSNR scenes across day/night/evening
and ego speeds, downloads each ``.usdz`` scene, renders 6 surround views + metric depths +
chase/orbit third-person GT (with watertight 3D ego sedan), and optionally removes raw ``.usdz``
files after export to keep disk usage compact.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from open_reality_view.data.nurec_renderer import NuRecSceneRenderer

NUREC_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"


def _save_png(arr_hwc: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(arr_hwc * 255.0, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(img).save(path)


def select_top_scenes(
    csv_path: Path,
    *,
    max_scenes: int,
    min_psnr: float,
    day_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    """Select high-PSNR scenes balanced between daytime and nighttime/evening."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    valid = []
    for r in rows:
        psnr_str = r.get("camera_psnr_26_10", "")
        if not psnr_str:
            continue
        psnr = float(psnr_str)
        if psnr >= min_psnr:
            r["_psnr"] = psnr  # type: ignore[assignment]
            valid.append(r)

    valid.sort(key=lambda r: float(r["_psnr"]), reverse=True)
    day_rows = [r for r in valid if r.get("time_of_day") == "day"]
    other_rows = [r for r in valid if r.get("time_of_day") != "day"]

    n_day = min(len(day_rows), round(max_scenes * day_ratio))
    n_other = min(len(other_rows), max_scenes - n_day)
    selected = day_rows[:n_day] + other_rows[:n_other]

    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


def export_one_clip(
    renderer: NuRecSceneRenderer,
    out_dir: Path,
    *,
    num_frames: int,
    fps: float,
    start_sec: float,
    src_hw: tuple[int, int],
    tgt_hw: tuple[int, int],
    orbit_azimuth_deg: float,
    save_depths: bool,
) -> None:
    cam_names = list(NuRecSceneRenderer.DEFAULT_SURROUND_CAMERAS)
    src_k_all: list[list] = []
    src_c2e_all: list[list] = []
    tgt_k_all: list[list] = []
    tgt_c2e_all: list[list] = []

    for f_idx in range(num_frames):
        ts_us = (start_sec + f_idx / fps) * 1e6
        bundle = renderer.render_frame_bundle(
            ts_us,
            surround_cameras=tuple(cam_names),
            src_hw=src_hw,
            tgt_hw=tgt_hw,
            orbit_azimuth_deg=orbit_azimuth_deg,
        )
        for cam in cam_names:
            _save_png(bundle["surround_rgb"][cam].cpu().numpy(), out_dir / "surround" / cam / f"{f_idx:06d}.png")
            if save_depths:
                d_path = out_dir / "depths" / cam / f"{f_idx:06d}.npy"
                d_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(d_path, bundle["surround_depth"][cam].cpu().numpy().astype(np.float32))
        _save_png(bundle["chase_rgb"].cpu().numpy(), out_dir / "chase_gt" / f"{f_idx:06d}.png")
        src_k_all.append(bundle["src_intrinsics"].cpu().tolist())
        src_c2e_all.append(bundle["src_c2e"].cpu().tolist())
        tgt_k_all.append(bundle["tgt_intrinsics"].cpu().tolist())
        tgt_c2e_all.append(bundle["tgt_c2e"].cpu().tolist())

    meta = {
        "clip_id": out_dir.name,
        "source_sequence_id": renderer.sequence_id,
        "camera_names": cam_names,
        "fps": fps,
        "orbit_azimuth_deg": orbit_azimuth_deg,
        "ego_centroid": list(renderer.ego_centroid),
        "ego_dim_lwh": list(renderer.ego_dim_lwh),
        "src_intrinsics": src_k_all,
        "src_c2e": src_c2e_all,
        "tgt_intrinsics": tgt_k_all,
        "tgt_c2e": tgt_c2e_all,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-export top NuRec 26.10 scenes for Stage 0.")
    parser.add_argument("--out-clips-dir", default="data/nurec_clips")
    parser.add_argument("--usdz-cache-dir", default="data/nurec_usdz_cache")
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--min-psnr", type=float, default=28.0)
    parser.add_argument("--day-ratio", type=float, default=0.7)
    parser.add_argument("--num-frames", type=int, default=17, help="Must satisfy 1 + 4*k.")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--src-height", type=int, default=224)
    parser.add_argument("--src-width", type=int, default=384)
    parser.add_argument("--tgt-height", type=int, default=480)
    parser.add_argument("--tgt-width", type=int, default=832)
    parser.add_argument("--orbit-angles", type=float, nargs="+", default=[0.0, -20.0, 20.0])
    parser.add_argument("--delete-usdz-after", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(hf_hub_download(NUREC_REPO, "benchmark_26.04_vs_26.10.csv", repo_type="dataset"))
    scenes = select_top_scenes(
        csv_path,
        max_scenes=args.max_scenes,
        min_psnr=args.min_psnr,
        day_ratio=args.day_ratio,
        seed=args.seed,
    )
    print(f"Selected {len(scenes)} NuRec 26.10 scenes (min_psnr={args.min_psnr}, day_ratio={args.day_ratio})")

    out_root = Path(args.out_clips_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(scenes, start=1):
        clip_id = row["clip_id"]
        rel_path = f"sample_set/26.10_release/{clip_id}/{clip_id}.usdz"
        print(
            f"[{idx}/{len(scenes)}] {clip_id} ({row['time_of_day']}, {row['ego_speed']}, "
            f"PSNR={float(row['camera_psnr_26_10']):.2f} dB)"
        )
        usdz_path = Path(
            hf_hub_download(
                NUREC_REPO,
                rel_path,
                repo_type="dataset",
                local_dir=args.usdz_cache_dir,
            )
        )
        t0 = time.perf_counter()
        renderer = NuRecSceneRenderer(usdz_path, device="cuda")

        # Export multiple windows / orbit angles per 20-second scene
        for angle_idx, az in enumerate(args.orbit_angles):
            start_sec = 1.0 + angle_idx * 4.5
            if start_sec + (args.num_frames / args.fps) > 19.0:
                start_sec = 1.0
            clip_out = out_root / f"{clip_id}_az{round(az):+03d}_t{int(start_sec):02d}"
            if (clip_out / "metadata.json").is_file():
                print(f"  Skipping existing {clip_out.name}")
                continue
            export_one_clip(
                renderer,
                clip_out,
                num_frames=args.num_frames,
                fps=args.fps,
                start_sec=start_sec,
                src_hw=(args.src_height, args.src_width),
                tgt_hw=(args.tgt_height, args.tgt_width),
                orbit_azimuth_deg=az,
                save_depths=True,
            )
            print(f"  Exported {clip_out.name} ({time.perf_counter() - t0:.1f}s elapsed)")

        del renderer
        if args.delete_usdz_after and usdz_path.is_file():
            usdz_path.unlink()


if __name__ == "__main__":
    main()
