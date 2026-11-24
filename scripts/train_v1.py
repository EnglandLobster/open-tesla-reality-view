#!/usr/bin/env python3
"""v1 Quality-First training script (Stage 1A Adapter Warmup & Stage 1B Joint VACE + LoRA/Full FT).

Designed for Google Colab (A100 80GB / RTX PRO 6000 96GB / H100 80GB) and local RTX 3080 20GB
debugging using pre-cached ``.pt`` shards produced by ``scripts/precache_features.py``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from open_reality_view.config import load_config
from open_reality_view.data import CachedShardDataset
from open_reality_view.models import AlpamayoContextAdapter, RealityViewVACEModel


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Reality View v1 (Alpamayo2-Edge + Wan2.1-VACE).")
    parser.add_argument("--config", default="configs/v1_edge_wan21.yaml")
    parser.add_argument("--shards-dir", required=True, help="Directory of pre-cached .pt shards.")
    parser.add_argument("--output-dir", default="runs/v1_edge_wan21")
    parser.add_argument("--stage", choices=["1a", "1b_lora", "1b_full"], default="1a")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--resume-adapter", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    dataset = CachedShardDataset(args.shards_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    adapter = AlpamayoContextAdapter(
        in_dim=cfg.encoder.vision_hidden_size,
        adapter_dim=cfg.context_adapter.adapter_dim,
        out_dim=cfg.context_adapter.output_dim,
        num_latents=cfg.context_adapter.num_latents,
        num_layers=cfg.context_adapter.num_layers,
        num_heads=cfg.context_adapter.num_heads,
        plucker_freq_bands=cfg.context_adapter.plucker_freq_bands,
        dropout=cfg.context_adapter.dropout,
    )
    if args.resume_adapter:
        adapter.load_state_dict(torch.load(args.resume_adapter, map_location="cpu", weights_only=True))
        print(f"Loaded adapter checkpoint from {args.resume_adapter}")

    model = RealityViewVACEModel.from_pretrained(
        cfg.generator.model_id,
        context_adapter=adapter,
        dtype=dtype,
        flow_shift=cfg.generator.flow_shift,
        vace_context_scale=cfg.generator.vace_context_scale,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    ).to(device)

    param_groups: list[dict] = []
    if args.stage == "1a":
        model.freeze_backbone_for_stage1a()
        total_steps = args.max_steps or cfg.training.stage1a_steps
        param_groups.append(
            {
                "params": list(model.context_adapter.parameters()),
                "lr": cfg.training.learning_rate_adapter,
                "base_lr": cfg.training.learning_rate_adapter,
            }
        )
    elif args.stage == "1b_lora":
        model.configure_stage1b_lora(
            rank=cfg.generator.lora_rank,
            alpha=cfg.generator.lora_alpha,
            train_vace_blocks=cfg.generator.train_vace_blocks,
        )
        total_steps = args.max_steps or cfg.training.stage1b_steps
        param_groups.extend(
            [
                {
                    "params": list(model.context_adapter.parameters()),
                    "lr": cfg.training.learning_rate_adapter * 0.5,
                    "base_lr": cfg.training.learning_rate_adapter * 0.5,
                },
                {
                    "params": [p for p in model.transformer.parameters() if p.requires_grad],
                    "lr": cfg.training.learning_rate_dit,
                    "base_lr": cfg.training.learning_rate_dit,
                },
            ]
        )
    else:  # 1b_full (Colab RTX PRO 6000 96GB / H100 80GB)
        model.requires_grad_(True)
        total_steps = args.max_steps or cfg.training.stage1b_steps
        param_groups.extend(
            [
                {
                    "params": list(model.context_adapter.parameters()),
                    "lr": cfg.training.learning_rate_adapter * 0.5,
                    "base_lr": cfg.training.learning_rate_adapter * 0.5,
                },
                {
                    "params": list(model.transformer.parameters()),
                    "lr": cfg.training.learning_rate_dit,
                    "base_lr": cfg.training.learning_rate_dit,
                },
            ]
        )

    trainable_params = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"Stage {args.stage} | Trainable params: {trainable_params / 1e6:.2f}M | Steps: {total_steps}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.training.weight_decay)
    out_dir = Path(args.output_dir) / f"stage_{args.stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = min(200, max(1, total_steps // 10))
    step = 0
    accum_loss = 0.0
    t_start = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    data_iter = iter(loader)
    while step < total_steps:
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            clean_latents = batch["clean_latents"].to(device=device, dtype=dtype)
            vace_context = batch["vace_context"].to(device=device, dtype=dtype)
            vision_tokens = batch["vision_tokens"].to(device=device, dtype=dtype)
            plucker_rays = batch["plucker_rays"].to(device=device, dtype=dtype)

            # Conditioning dropout on coarse 3D RGB channels (0:32) so model learns semantic recovery
            if cfg.geometry.coarse_rgb_dropout_prob > 0.0:
                drop = torch.rand(clean_latents.shape[0], device=device) < cfg.geometry.coarse_rgb_dropout_prob
                if drop.any():
                    vace_context = vace_context.clone()
                    vace_context[drop, :32] = 0.0
                    vace_context[drop, 32:] = 1.0

            with torch.amp.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda"),
            ):
                stats = model.compute_flow_matching_loss(
                    clean_latents=clean_latents,
                    vace_context=vace_context,
                    vision_tokens=vision_tokens,
                    plucker_rays=plucker_rays,
                )
            loss = stats["loss"] / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        for group in optimizer.param_groups:
            group["lr"] = _cosine_lr(step, total_steps, warmup_steps, group["base_lr"])

        torch.nn.utils.clip_grad_norm_(
            [p for g in param_groups for p in g["params"]],
            cfg.training.grad_clip_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0 or step == 1 or step == total_steps:
            elapsed = time.perf_counter() - t_start
            vram = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
            print(
                f"step {step:5d}/{total_steps} | loss={accum_loss:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | vram={vram:.2f}GB | {elapsed:.1f}s"
            )
        accum_loss = 0.0

        if step % args.save_every == 0 or step == total_steps:
            torch.save(model.context_adapter.state_dict(), out_dir / f"adapter_step{step:06d}.pt")
            if args.stage == "1b_lora":
                model.transformer.save_pretrained(out_dir / f"lora_step{step:06d}")
            elif args.stage == "1b_full":
                model.transformer.save_pretrained(out_dir / f"transformer_step{step:06d}")
            print(f"Saved checkpoint at step {step} -> {out_dir}")


if __name__ == "__main__":
    main()
