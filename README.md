# Open Tesla Reality View

An open-source, research-grade replication of **Tesla FSD v15 Reality View** — a real-time generative world-rendering system that turns surround-view vehicle cameras into a photorealistic, geometrically grounded third-person view.

> **v1 Stack (Quality-First Flow-Matching Diffusion):**
> - **Shared Driving Vision Encoder (UVE Equivalent):** [`nvidia/Alpamayo2-Edge`](https://huggingface.co/nvidia/Alpamayo2-Edge) (`cosmos3_edge_vision` / SigLIP2, 0.41B, `hidden_size=1152`, natively trained on multi-camera driving + NuRec; pluggable interface for `DM0` / `DINOv3` ablations)
> - **3D Geometry Anchor:** [`depth-anything/DA3-BASE`](https://huggingface.co/depth-anything/DA3-BASE) (`DA3-Streaming`) multi-view metric depth projection + 3D Ego-Car composite + blind-spot occlusion mask
> - **Generative Video Backbone:** [`Wan-AI/Wan2.1-VACE-1.3B-diffusers`](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers) (`WanVACETransformer3DModel`, Flow-Matching DiT, `hidden_size=1536`, `text_dim=4096`, `vace_in_channels=96`)
> - **v2 Real-Time Upgrade Path:** Distill the v1 multi-step Flow-Matching teacher into a **1–2 step Causal Mean-Flow** streaming model via [`NVlabs/rcm`](https://github.com/NVlabs/rcm) (`Causal-rCM`) + lightweight `TAEHV` video decoding.

## Architecture Overview

```text
K Surround Cameras (6-7 views) + Intrinsics/Extrinsics + Target Chase-Cam Pose
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
[Branch A: 3D Geometry Anchor]         [Branch B: Shared Driving Encoder (UVE)]
  Depth Anything 3 (DA3-Streaming)       Frozen nvidia/Alpamayo2-Edge Vision Tower
  Multi-view depth unprojection          (SigLIP2, 0.41B, D=1152, BF16 ~0.8 GB)
  + 3D Ego-Car mesh + Hole Mask                    │
         │                              ┌──────────┴──────────┐
         ▼                              ▼                     ▼
  Coarse Target-View RGB Video   [AlpamayoContextAdapter]  [Optional Dual Branch]
  + Blind-Spot Mask (T×H×W)      Plücker Ray + Cam ID +    Alpamayo2-Edge Action Expert
         │                       Perceiver Resampler       → 64-waypoint Ego Trajectory
         ▼                       (D=1152 → D=4096)         (Overlay on Reality View)
  AutoencoderKLWan (8×8×4 VAE)          │
  → VACE Control Latent (96 ch)         │ Cross-Attention Context [B, N_ctx, 4096]
         │                              │ (replaces 11 GB UMT5-XXL text encoder)
         └──────────────┬───────────────┘
                        ▼
       [Wan2.1-VACE-1.3B Flow-Matching DiT]
       - v1 (Quality-First): 15-30 step UniPC/Euler Flow Matching
       - v2 (Real-Time): 1-2 step Causal-rCM (Mean-Flow + Self-Forcing) + TAEHV
                        │
                        ▼
        Photorealistic Third-Person Reality View Stream
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full mathematical specifications, Stage 0 data construction (`PhysicalAI-Autonomous-Vehicles-NuRec`, `OmniRe + Difix3D+`, `CARLA`), and the 500–600 A100h training budget.

## Roadmap

- [x] **Milestone 0 — Architecture & Core Bridge Skeleton**
  - Pluggable `SharedVisionEncoder` interface & `Alpamayo2EdgeVisionEncoder` wrapper
  - Multi-camera `PluckerRayEncoder` & differentiable z-buffer `MultiViewProjector`
  - `AlpamayoContextAdapter` (`1152 → 4096`) bridging frozen Alpamayo2-Edge tokens into `WanVACETransformer3DModel`
  - Unit tests verifying exact tensor contracts without downloading multi-GB weights
- [ ] **Milestone 1 — Stage 0 Third-Person GT Data Pipeline**
  - `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (1,601 pre-reconstructed 3DGS driving scenes) renderer + 3D Ego-Car compositor
  - `CARLA` synthetic cold-start pipeline (zero-hole third-person chase camera GT)
  - Optional `OmniRe` (`drivestudio`) + `Difix3D+` artifact cleaner for `nuScenes` / `Waymo`
- [ ] **Milestone 2 — v1 Quality-First Training (Colab A100 / PRO 6000 / H100)**
  - Pre-cache frozen Alpamayo2-Edge vision tokens & Wan-VAE latents
  - Train `AlpamayoContextAdapter` + high-rank LoRA / full fine-tuning on `Wan2.1-VACE-1.3B`
  - Demo inference + interactive camera orbiting on RTX 3080 20GB
- [ ] **Milestone 3 — v2 Real-Time Causal Mean-Flow & Dual-Branch Demo**
  - Causal chunking (4-frame chunk = 1 latent frame) + short sliding-window KV cache
  - 1–2 step Mean-Flow / sCM + Self-Forcing distillation (`Causal-rCM`) + `TAEHV` decoder
  - Dual-branch overlay: live `Alpamayo2-Edge` planned trajectory + Reality View stream

## Quick Start

```bash
# Install editable package (requires PyTorch >= 2.4)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify local hardware & dependencies
python scripts/check_env.py

# Run architecture tensor-contract tests
pytest -v
```

## License

Code in this repository is released under the Apache-2.0 license. Pretrained weights (`nvidia/Alpamayo2-Edge`, `Wan-AI/Wan2.1-VACE-1.3B-diffusers`, `depth-anything/DA3-BASE`) and datasets (`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`) remain subject to their respective upstream licenses.
