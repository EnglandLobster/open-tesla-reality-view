#!/usr/bin/env python3
"""Export a NuRec 26.10 scene as an ArtiFixer `reconstructed_colmap` scene (no COLMAP / 3DGRUT re-fit).

ArtiFixer (nv-tlabs/ArtiFixer) normally prepares a scene from COLMAP, retrains 3DGRUT and renders the
target path. We already have NVIDIA's NuRec Gaussians, so this script writes the prepared layout
directly from our renderer:

  <out>/
    images/a_XXXXX.png        anchors: NuRec renders of the surround cameras (near-real training views)
    renders/{:05d}.png        3DGS render for EVERY frame (anchors + chase targets)
    opacity/{:05d}.png        render alpha (0..255)
    transforms.json           OPENCV intrinsics + OpenGL camera-to-world (NuRec world, metres)
    selected_indices.json     anchor frame indices (reference views)
    target_indices.json       chase-camera frame indices to generate
    reconstruction.usdz       symlink to the source scene (required field; not re-fit)
    split.json                ArtiFixer split entry (prompt path filled by make_artifixer_prompt.py)

Frames 0..A-1 are anchors, A..A+T-1 are the Tesla-framed chase targets (same timestamps and framing as
scripts/exp_cosmos_transfer_refine.py, at a smaller size that fits a 20 GB card).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from open_reality_view.data.nurec_renderer import NuRecSceneRenderer
from open_reality_view.geometry.projector import build_orbit_chase_pose

SCENES = {
    "bb5389f1": "data/nurec_samples/sample_set/26.10_release/"
    "bb5389f1-9514-4cfa-8322-dae897ed3786/bb5389f1-9514-4cfa-8322-dae897ed3786.usdz",
    "3582a449": "data/nurec_samples/sample_set/26.10_release/"
    "3582a449-6e1a-4dca-8c53-efc49a623b88/3582a449-6e1a-4dca-8c53-efc49a623b88.usdz",
}
TESLA_CHASE_RADIUS, TESLA_CHASE_ELEV_DEG, TESLA_CHASE_FOV_X_DEG = 13.8, 11.5, 74.0
TESLA_LOOK_AHEAD_X, TESLA_LOOK_AT_Z = 4.5, 1.25
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _save_rgb(t: torch.Tensor, path: Path) -> None:
    Image.fromarray((t.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def _save_alpha(t: torch.Tensor, path: Path) -> None:
    Image.fromarray((t.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8), mode="L").save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", choices=sorted(SCENES), default="bb5389f1")
    p.add_argument("--out-dir", default=None, help="default: outputs/artifixer_scenes/<scene>")
    p.add_argument("--num-frames", type=int, default=93)
    p.add_argument("--fps", type=float, default=16.0)
    p.add_argument("--start-sec", type=float, default=0.85)
    p.add_argument("--height", type=int, default=528, help="multiple of 16")
    p.add_argument("--width", type=int, default=960, help="multiple of 16")
    p.add_argument("--anchor-times", type=int, default=2, help="timestamps per surround camera used as anchors")
    args = p.parse_args()
    assert args.height % 16 == 0 and args.width % 16 == 0, "Wan2.1 ArtiFixer needs multiples of 16"

    out = Path(args.out_dir or f"outputs/artifixer_scenes/{args.scene}")
    for sub in ("images", "renders", "opacity"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    renderer = NuRecSceneRenderer(SCENES[args.scene])
    dev, h, w = renderer.device, args.height, args.width

    frames: list[dict] = []

    def add_frame(ts_us: float, t_cam_rig: torch.Tensor, k: torch.Tensor, image_name: str | None) -> int:
        idx = len(frames)
        res = renderer.render_camera(ts_us, t_cam_rig, k, h, w, composite_ego=False)
        _save_rgb(res.rgb, out / "renders" / f"{idx:05d}.png")
        _save_alpha(res.alpha, out / "opacity" / f"{idx:05d}.png")
        if image_name is not None:  # anchors: the render of a fitted training view stands in for the photo
            _save_rgb(res.rgb, out / "images" / image_name)
        c2w_cv = (renderer.rig_to_nurec(ts_us) @ t_cam_rig.to(dev, torch.float32)).cpu().numpy().astype(np.float64)
        kk = k.cpu().numpy()
        frames.append(
            {
                "file_path": f"images/{image_name}" if image_name else f"renders/{idx:05d}.png",
                "transform_matrix": (c2w_cv @ CV_TO_GL).tolist(),
                "fl_x": float(kk[0, 0]), "fl_y": float(kk[1, 1]),
                "cx": float(kk[0, 2]), "cy": float(kk[1, 2]),
                "w": w, "h": h,
            }
        )
        return idx

    # Anchors: every surround camera at a few timestamps spread over the clip
    clip_len = (args.num_frames - 1) / args.fps
    anchor_ts = [args.start_sec + clip_len * i / max(args.anchor_times - 1, 1) for i in range(args.anchor_times)]
    anchors = []
    for ts in anchor_ts:
        for cam in NuRecSceneRenderer.DEFAULT_SURROUND_CAMERAS:
            t_cam_rig = torch.from_numpy(renderer.cameras[cam].T_sensor_rig).to(dev)
            k = renderer.camera_intrinsics(cam, h, w)
            anchors.append(add_frame(ts * 1e6, t_cam_rig, k, f"a_{len(frames):05d}.png"))
    print(f"[export] {len(anchors)} anchors", flush=True)

    # Targets: Tesla-framed chase camera
    fx = (w * 0.5) / math.tan(math.radians(TESLA_CHASE_FOV_X_DEG * 0.5))
    k_tgt = torch.tensor([[fx, 0.0, w * 0.5], [0.0, fx, h * 0.5], [0.0, 0.0, 1.0]], device=dev)
    chase = build_orbit_chase_pose(
        radius=TESLA_CHASE_RADIUS, elevation_deg=TESLA_CHASE_ELEV_DEG, azimuth_deg=0.0,
        look_at_xyz=(renderer.ego_centroid[0] + TESLA_LOOK_AHEAD_X, 0.0, TESLA_LOOK_AT_Z), device=dev,
    )
    targets = [add_frame((args.start_sec + i / args.fps) * 1e6, chase, k_tgt, None) for i in range(args.num_frames)]
    print(f"[export] {len(targets)} chase targets", flush=True)

    (out / "transforms.json").write_text(json.dumps({"camera_model": "OPENCV", "frames": frames}, indent=1))
    (out / "selected_indices.json").write_text(json.dumps(anchors))
    (out / "target_indices.json").write_text(json.dumps(targets))
    link = out / "reconstruction.usdz"
    if not link.exists():
        os.symlink(Path(SCENES[args.scene]).resolve(), link)
    split = {
        "test": {
            args.scene: {
                "scene_id": args.scene,
                "transforms_path": "transforms.json",
                "image_root": ".",
                "render_dir": "renders",
                "opacity_dir": "opacity",
                "selected_indices_path": "selected_indices.json",
                "target_indices_path": "target_indices.json",
                "prompt_path": "prompt/caption_wan_t5.h5",
                "reconstruction_checkpoint": "reconstruction.usdz",
                "camera_scale": 1.0,  # NuRec world is already metric
                "has_gt": False,
            }
        }
    }
    (out / "split.json").write_text(json.dumps(split, indent=2))
    (out / "export_meta.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[export] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
