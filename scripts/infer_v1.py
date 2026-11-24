#!/usr/bin/env python3
"""Run end-to-end v1 Reality View inference and save side-by-side preview frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from PIL import Image

from open_reality_view.config import load_config
from open_reality_view.data import generate_synthetic_driving_clip
from open_reality_view.encoders import Alpamayo2EdgeVisionEncoder
from open_reality_view.geometry import (
    EgoVehicleCompositor,
    MultiViewDepthProjector,
    build_orbit_chase_pose,
)
from open_reality_view.models import AlpamayoContextAdapter, RealityViewVACEModel
from open_reality_view.pipelines import RealityViewVACEPipeline


def _to_uint8_hwc(tensor_chw: torch.Tensor) -> np.ndarray:
    arr = tensor_chw.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end v1 Reality View inference demo.")
    parser.add_argument("--config", default="configs/v1_edge_wan21.yaml")
    parser.add_argument("--adapter-ckpt", type=str, default=None)
    parser.add_argument("--output-dir", default="outputs/v1_preview")
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--orbit-azimuth-deg", type=float, default=0.0, help="Interactive orbit yaw.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    encoder = Alpamayo2EdgeVisionEncoder.from_pretrained(
        cfg.encoder.model_id,
        device=device,
        dtype=dtype,
        target_hw=(224, 384),
    )
    vae = AutoencoderKLWan.from_pretrained(cfg.generator.model_id, subfolder="vae", torch_dtype=dtype).to(device)
    adapter = AlpamayoContextAdapter(
        in_dim=cfg.encoder.vision_hidden_size,
        adapter_dim=cfg.context_adapter.adapter_dim,
        out_dim=cfg.context_adapter.output_dim,
        num_latents=cfg.context_adapter.num_latents,
        num_layers=cfg.context_adapter.num_layers,
        num_heads=cfg.context_adapter.num_heads,
    ).to(device=device, dtype=dtype)
    if args.adapter_ckpt:
        adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location="cpu", weights_only=True))

    model = RealityViewVACEModel.from_pretrained(
        cfg.generator.model_id,
        context_adapter=adapter,
        dtype=dtype,
        flow_shift=cfg.generator.flow_shift,
        gradient_checkpointing=False,
    ).to(device)
    model.eval()

    scheduler = UniPCMultistepScheduler.from_pretrained(cfg.generator.model_id, subfolder="scheduler")
    scheduler = UniPCMultistepScheduler.from_config(scheduler.config, flow_shift=cfg.generator.flow_shift)

    pipe = RealityViewVACEPipeline(
        vision_encoder=encoder,
        model=model,
        vae=vae,
        scheduler=scheduler,
        projector=MultiViewDepthProjector(),
        ego_compositor=EgoVehicleCompositor(ego_box_lwh=tuple(cfg.data.ego_box_lwh)),
        device=device,
        dtype=dtype,
    )

    clip = generate_synthetic_driving_clip(
        num_frames=args.num_frames,
        tgt_hw=(args.height, args.width),
    )
    tgt_c2e = clip.tgt_c2e.unsqueeze(0)
    if args.orbit_azimuth_deg != 0.0:
        orbit_pose = build_orbit_chase_pose(azimuth_deg=args.orbit_azimuth_deg)
        tgt_c2e = orbit_pose[None, None].expand(1, args.num_frames, 4, 4).clone()

    out = pipe(
        clip.surround_rgb.unsqueeze(0),
        clip.surround_depth.unsqueeze(0),
        clip.src_intrinsics.unsqueeze(0),
        clip.src_c2e.unsqueeze(0),
        clip.tgt_intrinsics.unsqueeze(0),
        tgt_c2e,
        target_hw=(args.height, args.width),
        num_inference_steps=args.steps,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in range(args.num_frames):
        coarse_img = _to_uint8_hwc(out.coarse_rgb[0, :, t])
        gen_img = _to_uint8_hwc(out.generated_rgb[0, :, t])
        gt_img = _to_uint8_hwc(clip.chase_gt_rgb[t])
        panel = np.concatenate([coarse_img, gen_img, gt_img], axis=1)
        Image.fromarray(panel).save(out_dir / f"frame_{t:04d}.png")
    print(f"Saved {args.num_frames} side-by-side panels (coarse | generated | gt) to {out_dir}")


if __name__ == "__main__":
    main()
