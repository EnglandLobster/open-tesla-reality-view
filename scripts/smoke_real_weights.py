#!/usr/bin/env python3
"""Download and smoke-test REAL Alpamayo2-Edge + Wan2.1-VACE-1.3B weights on GPU.

Verifies:
  1. Exact 437-key strict weight load of ``nvidia/Alpamayo2-Edge`` vision tower (~0.82 GB BF16)
  2. Real ``AutoencoderKLWan`` + ``WanVACETransformer3DModel`` load (~3.1 GB BF16, no UMT5!)
  3. Full synthetic multi-camera projection + EgoVehicleCompositor + Stage 1A forward/backward pass
  4. Peak VRAM and per-stage wall-clock latency on RTX 3080 20GB / Colab A100 / PRO 6000 / H100
"""

from __future__ import annotations

import argparse
import time

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler

from open_reality_view.config import load_config
from open_reality_view.data import generate_synthetic_driving_clip
from open_reality_view.encoders import Alpamayo2EdgeVisionEncoder
from open_reality_view.geometry import (
    EgoVehicleCompositor,
    MultiViewDepthProjector,
    compute_plucker_rays,
)
from open_reality_view.models import (
    AlpamayoContextAdapter,
    RealityViewVACEModel,
    encode_vace_control,
    encode_wan_latents,
)
from open_reality_view.pipelines import RealityViewVACEPipeline


def _vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-weight GPU smoke test for v1 stack.")
    parser.add_argument("--config", default="configs/v1_edge_wan21.yaml")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=9, help="Must satisfy 1 + 4*k.")
    parser.add_argument("--denoise-steps", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.perf_counter()
    print(f"[1/4] Loading frozen vision tower from {cfg.encoder.model_id} ...")
    encoder = Alpamayo2EdgeVisionEncoder.from_pretrained(
        cfg.encoder.model_id,
        device=device,
        dtype=dtype,
        target_hw=(224, 384),
    )
    print(f"      Loaded 437 SigLIP2 vision tensors strictly in {time.perf_counter() - t0:.1f}s | Peak VRAM: {_vram_gb():.2f} GB")

    t1 = time.perf_counter()
    print(f"[2/4] Loading Wan2.1-VACE-1.3B VAE & DiT from {cfg.generator.model_id} (no UMT5) ...")
    vae = AutoencoderKLWan.from_pretrained(
        cfg.generator.model_id,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    adapter = AlpamayoContextAdapter(
        in_dim=cfg.encoder.vision_hidden_size,
        adapter_dim=cfg.context_adapter.adapter_dim,
        out_dim=cfg.context_adapter.output_dim,
        num_latents=cfg.context_adapter.num_latents,
        num_layers=cfg.context_adapter.num_layers,
        num_heads=cfg.context_adapter.num_heads,
    ).to(device=device, dtype=dtype)

    model = RealityViewVACEModel.from_pretrained(
        cfg.generator.model_id,
        context_adapter=adapter,
        dtype=dtype,
        flow_shift=cfg.generator.flow_shift,
        vace_context_scale=cfg.generator.vace_context_scale,
        gradient_checkpointing=True,
    ).to(device)
    model.freeze_backbone_for_stage1a()
    print(f"      Loaded Wan2.1-VACE + Adapter in {time.perf_counter() - t1:.1f}s | Peak VRAM: {_vram_gb():.2f} GB")

    print(f"[3/4] Running Stage 1A training forward + backward step ({args.num_frames}f @ {args.height}x{args.width}) ...")
    clip = generate_synthetic_driving_clip(
        num_frames=args.num_frames,
        src_hw=(224, 384),
        tgt_hw=(args.height, args.width),
    )
    projector = MultiViewDepthProjector().to(device)
    ego_comp = EgoVehicleCompositor().to(device)

    surround_rgb = clip.surround_rgb.unsqueeze(0).to(device)
    surround_depth = clip.surround_depth.unsqueeze(0).to(device)
    src_k = clip.src_intrinsics.unsqueeze(0).to(device)
    src_c2e = clip.src_c2e.unsqueeze(0).to(device)
    tgt_k = clip.tgt_intrinsics.unsqueeze(0).to(device)
    tgt_c2e = clip.tgt_c2e.unsqueeze(0).to(device)
    gt_video_m11 = clip.chase_gt_rgb.permute(1, 0, 2, 3).unsqueeze(0).to(device) * 2.0 - 1.0

    t2 = time.perf_counter()
    coarse_frames, mask_frames = [], []
    for f_idx in range(args.num_frames):
        p = projector(
            surround_rgb[:, f_idx],
            surround_depth[:, f_idx],
            src_k[:, f_idx],
            src_c2e[:, f_idx],
            tgt_k[:, f_idx],
            tgt_c2e[:, f_idx],
            target_hw=(args.height, args.width),
        )
        p = ego_comp(p, tgt_k[:, f_idx], tgt_c2e[:, f_idx])
        coarse_frames.append(p.coarse_rgb)
        mask_frames.append(p.hole_mask)

    coarse_m11 = torch.stack(coarse_frames, dim=2) * 2.0 - 1.0
    hole_mask = torch.stack(mask_frames, dim=2)
    clean_latents = encode_wan_latents(vae, gt_video_m11).to(dtype=dtype)
    vace_ctrl = encode_vace_control(vae, coarse_m11, hole_mask).to(dtype=dtype)

    key_idx = list(range(0, args.num_frames, 4))
    enc_out = encoder.encode_multiview(surround_rgb[:, key_idx])
    h_p, w_p = enc_out.grid_hw
    tgt_e2c = torch.linalg.inv(tgt_c2e[:, key_idx])[:, :, None]
    plucker = compute_plucker_rays(
        src_k[:, key_idx],
        src_c2e[:, key_idx],
        grid_height=h_p,
        grid_width=w_p,
        orig_height=224,
        orig_width=384,
        target_w2c=tgt_e2c,
    ).to(dtype=dtype)

    stats = model.compute_flow_matching_loss(
        clean_latents=clean_latents,
        vace_context=vace_ctrl,
        vision_tokens=enc_out.tokens,
        plucker_rays=plucker,
    )
    stats["loss"].backward()
    assert model.context_adapter.out_proj.weight.grad is not None
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(
        f"      Stage 1A loss={stats['loss'].item():.4f} | "
        f"Time: {time.perf_counter() - t2:.2f}s | Peak VRAM: {_vram_gb():.2f} GB"
    )

    print(f"[4/4] Running {args.denoise_steps}-step Flow-Matching inference smoke pass ...")
    model.zero_grad(set_to_none=True)
    model.eval()
    scheduler = UniPCMultistepScheduler.from_pretrained(
        cfg.generator.model_id,
        subfolder="scheduler",
    )
    scheduler = UniPCMultistepScheduler.from_config(scheduler.config, flow_shift=cfg.generator.flow_shift)
    pipe = RealityViewVACEPipeline(
        vision_encoder=encoder,
        model=model,
        vae=vae,
        scheduler=scheduler,
        projector=projector,
        ego_compositor=ego_comp,
        device=device,
        dtype=dtype,
    )
    t3 = time.perf_counter()
    out = pipe(
        surround_rgb,
        surround_depth,
        src_k,
        src_c2e,
        tgt_k,
        tgt_c2e,
        target_hw=(args.height, args.width),
        num_inference_steps=args.denoise_steps,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(
        f"      Output video: {tuple(out.generated_rgb.shape)} in {time.perf_counter() - t3:.2f}s | "
        f"Peak VRAM: {_vram_gb():.2f} GB"
    )
    print("REAL-WEIGHT SMOKE TEST PASSED!")


if __name__ == "__main__":
    main()
