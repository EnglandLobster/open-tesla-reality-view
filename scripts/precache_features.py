#!/usr/bin/env python3
"""Stage 0 offline feature pre-cacher for high-throughput Colab A100 / PRO 6000 / H100 training.

Pre-computes frozen representations per clip and saves compact BF16 ``.pt`` shards:
  - ``clean_latents``:  [16, F_lat, H_lat, W_lat]  (AutoencoderKLWan on chase GT)
  - ``vace_context``:   [96, F_lat, H_lat, W_lat]  (3D projection + ego car + hole mask)
  - ``vision_tokens``:  [F_lat, K, H_p, W_p, 1152] (frozen Alpamayo2-Edge SigLIP2)
  - ``plucker_rays``:   [F_lat, K, 6, H_p, W_p]    (target-camera-frame Plücker rays)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan

from open_reality_view.config import load_config
from open_reality_view.data import (
    ClipDirectoryDataset,
    DrivingClip,
    generate_synthetic_driving_clip,
    save_cached_shard,
)
from open_reality_view.encoders import Alpamayo2EdgeVisionEncoder
from open_reality_view.geometry import (
    EgoVehicleCompositor,
    MultiViewDepthProjector,
    Temporal3DCacheProjector,
    compute_plucker_rays,
)
from open_reality_view.models import encode_vace_control, encode_wan_latents


@torch.no_grad()
def precache_clip(
    clip: DrivingClip,
    *,
    encoder: Alpamayo2EdgeVisionEncoder,
    vae: AutoencoderKLWan,
    projector: MultiViewDepthProjector,
    ego_comp: EgoVehicleCompositor,
    out_dir: Path,
    device: torch.device,
    temporal_stride: int = 4,
    spatial_stride: int = 8,
) -> Path:
    if clip.surround_depth is None:
        raise ValueError(
            f"Clip '{clip.clip_id}' has no depth maps. Pre-compute metric depth via DA3-BASE first."
        )

    num_frames = clip.surround_rgb.shape[0]
    h_src, w_src = clip.surround_rgb.shape[-2:]
    h_tgt, w_tgt = clip.chase_gt_rgb.shape[-2:]

    surround_rgb = clip.surround_rgb.unsqueeze(0).to(device)
    surround_depth = clip.surround_depth.unsqueeze(0).to(device)
    src_k = clip.src_intrinsics.unsqueeze(0).to(device)
    src_c2e = clip.src_c2e.unsqueeze(0).to(device)
    tgt_k = clip.tgt_intrinsics.unsqueeze(0).to(device)
    tgt_c2e = clip.tgt_c2e.unsqueeze(0).to(device)
    gt_video_m11 = clip.chase_gt_rgb.permute(1, 0, 2, 3).unsqueeze(0).to(device) * 2.0 - 1.0

    coarse_frames, mask_frames = [], []
    use_temporal_cache = (
        isinstance(projector, Temporal3DCacheProjector) and clip.rig_to_world is not None
    )
    rig_to_world = clip.rig_to_world.unsqueeze(0).to(device) if clip.rig_to_world is not None else None
    cache_window = 6

    for f_idx in range(num_frames):
        if use_temporal_cache and rig_to_world is not None:
            w_start = max(0, f_idx - cache_window + 1)
            proj = projector.project_temporal_window(
                surround_rgb[:, w_start : f_idx + 1],
                surround_depth[:, w_start : f_idx + 1],
                src_k[:, w_start : f_idx + 1],
                src_c2e[:, w_start : f_idx + 1],
                rig_to_world[:, w_start : f_idx + 1],
                tgt_k[:, f_idx],
                tgt_c2e[:, f_idx],
                target_hw=(h_tgt, w_tgt),
            )
        else:
            proj = projector(
                surround_rgb[:, f_idx],
                surround_depth[:, f_idx],
                src_k[:, f_idx],
                src_c2e[:, f_idx],
                tgt_k[:, f_idx],
                tgt_c2e[:, f_idx],
                target_hw=(h_tgt, w_tgt),
            )
        proj = ego_comp(proj, tgt_k[:, f_idx], tgt_c2e[:, f_idx])
        coarse_frames.append(proj.coarse_rgb)
        mask_frames.append(proj.hole_mask)

    coarse_m11 = torch.stack(coarse_frames, dim=2) * 2.0 - 1.0
    hole_mask = torch.stack(mask_frames, dim=2)

    clean_latents = encode_wan_latents(vae, gt_video_m11).squeeze(0)
    vace_context = encode_vace_control(
        vae,
        coarse_m11,
        hole_mask,
        spatial_stride=spatial_stride,
        temporal_stride=temporal_stride,
    ).squeeze(0)

    key_idx = list(range(0, num_frames, temporal_stride))
    enc_out = encoder.encode_multiview(surround_rgb[:, key_idx])
    h_p, w_p = enc_out.grid_hw
    tgt_e2c = torch.linalg.inv(tgt_c2e[:, key_idx])[:, :, None]
    plucker = compute_plucker_rays(
        src_k[:, key_idx],
        src_c2e[:, key_idx],
        grid_height=h_p,
        grid_width=w_p,
        orig_height=h_src,
        orig_width=w_src,
        target_w2c=tgt_e2c,
    ).squeeze(0)

    out_path = out_dir / f"{clip.clip_id}.pt"
    save_cached_shard(
        out_path,
        clean_latents=clean_latents,
        vace_context=vace_context,
        vision_tokens=enc_out.tokens.squeeze(0),
        plucker_rays=plucker,
        clip_id=clip.clip_id,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache frozen Stage 0 features into .pt shards.")
    parser.add_argument("--config", default="configs/v1_edge_wan21.yaml")
    parser.add_argument("--clips-dir", type=str, default=None, help="Directory of standardized clips.")
    parser.add_argument("--synthetic-demo", type=int, default=0, help="Generate N synthetic clips.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_frames = args.num_frames or cfg.data.clip_num_frames
    tgt_h = args.height or cfg.data.target_height
    tgt_w = args.width or cfg.data.target_width

    encoder = Alpamayo2EdgeVisionEncoder.from_pretrained(
        cfg.encoder.model_id,
        device=device,
        dtype=dtype,
        target_hw=(224, 384),
    )
    vae = AutoencoderKLWan.from_pretrained(
        cfg.generator.model_id,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    projector: MultiViewDepthProjector = Temporal3DCacheProjector(
        min_depth_m=cfg.geometry.min_depth_m,
        max_depth_m=cfg.geometry.max_depth_m,
        splat_radius_px=cfg.geometry.splat_radius_px,
    ).to(device)
    ego_glb = Path("assets/ego_car_concept.glb")
    ego_comp = EgoVehicleCompositor(
        ego_box_lwh=tuple(cfg.data.ego_box_lwh),
        glb_path=ego_glb if ego_glb.is_file() else None,
    ).to(device)

    clips: list[DrivingClip] = []
    if args.synthetic_demo > 0:
        for i in range(args.synthetic_demo):
            clips.append(
                generate_synthetic_driving_clip(
                    clip_id=f"synthetic_{i:04d}",
                    num_frames=num_frames,
                    tgt_hw=(tgt_h, tgt_w),
                    chase_offset_xyz=tuple(cfg.data.default_chase_offset_xyz),
                    chase_pitch_deg=cfg.data.default_chase_pitch_deg,
                )
            )
    elif args.clips_dir is not None:
        ds = ClipDirectoryDataset(args.clips_dir, num_frames=num_frames, tgt_hw=(tgt_h, tgt_w))
        clips = [ds[i] for i in range(len(ds))]
    else:
        raise ValueError("Provide either --clips-dir or --synthetic-demo N.")

    for clip in clips:
        saved = precache_clip(
            clip,
            encoder=encoder,
            vae=vae,
            projector=projector,
            ego_comp=ego_comp,
            out_dir=out_dir,
            device=device,
            temporal_stride=cfg.generator.vae_temporal_stride,
            spatial_stride=cfg.generator.vae_spatial_stride,
        )
        print(f"Cached {clip.clip_id} -> {saved}")


if __name__ == "__main__":
    main()
