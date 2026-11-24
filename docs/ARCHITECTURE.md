# Technical Architecture & Roadmap (`v1: Alpamayo2-Edge + Wan2.1-VACE-1.3B`)

## 1. Design Goals & Hardware Envelope

This project targets a **research-grade open-source replication** of Tesla FSD v15 Reality View under realistic academic/indie compute constraints:

- **v1 Priority:** Maximum visual fidelity, strict 3D geometric faithfulness (zero safety-critical vehicle drift/hallucination), and free virtual camera orbiting. Real-time FPS optimization is deferred to **v2**.
- **Training Budget:** ~500–600 A100 GPU-hours (Google Colab A100 / RTX PRO 6000 96GB / H100 80GB).
- **Local Development & Demo Hardware:** Single RTX 3080 20GB (Ampere, BF16/FP16, no native FP8 Tensor Cores).

---

## 2. Why Dual-Branch Pseudo-UVE + Flow-Matching VACE?

### 2.1 Tesla UVE vs. Open-Source VLA Vision Towers

Tesla's Universal Vision Encoder (UVE) jointly encodes 7–8 calibrated surround cameras into 3D-consistent spatiotemporal tokens shared by the driving policy and the ~1B Conditional Mean-Flow renderer.

Open-source driving VLAs such as `nvidia/Alpamayo2-Edge` provide an exceptional **driving-domain semantic vision encoder** (`cosmos3_edge_vision` / SigLIP2, 0.41B, `hidden_size=1152`), natively co-trained on real multi-camera driving datasets (`PhysicalAI-Autonomous-Vehicles` & `NuRec`). However, because each camera image is processed independently as a 2D view, relying *solely* on cross-attention from frozen 2D tokens forces the diffusion backbone to learn multi-camera projective geometry from scratch — which does not converge cleanly under 500–600 A100h.

### 2.2 The Dual-Branch Solution

We decompose the pseudo-UVE front-end into two complementary branches that map cleanly onto the **two native conditioning ports** of `Wan2.1-VACE-1.3B`:

1. **Branch A — Pixel-Aligned 3D Geometry Anchor (→ VACE Control Port, 96 ch):**
   - Multi-view metric depth from [`depth-anything/DA3-BASE`](https://huggingface.co/depth-anything/DA3-BASE) (`DA3-Streaming`, Apache-2.0) unprojects surround-view pixels into ego 3D coordinates and projects/splats them into the exact target third-person chase camera.
   - A 3D **Ego-Vehicle Asset** (with dynamic brake/turn-signal state and ground contact shadow) is composited into the target view — solving the fundamental "invisible ego-car hole" present in all sensor-only driving reconstructions.
   - Produces a pixel-aligned **coarse RGB video** `V_coarse` and a **blind-spot / occlusion mask** `M_hole` (`0` = directly observed by surround cameras, `1` = occluded roof / sky / unprojected blind spot).
   - Encoded by `AutoencoderKLWan` (`8×8×4`) directly into `WanVACETransformer3DModel`'s native `vace_in_channels=96` control pathway (`16ch inactive + 16ch reactive + 64ch spatial mask`), injected across 15 even DiT blocks (`vace_layers = [0, 2, ..., 28]`).

2. **Branch B — Shared Driving Semantic Encoder (→ Cross-Attention Port, `text_dim=4096`):**
   - Frozen `nvidia/Alpamayo2-Edge` vision tower (`cosmos3_edge_vision`, 27 layers, `patch_size=16`, `spatial_merge_size=2`, `hidden_size=1152`, ~0.82 GB in BF16).
   - **UMT5-XXL Removal:** Instead of loading Wan2.1's 11 GB text encoder, a trainable **`AlpamayoContextAdapter`** enriches multi-camera vision tokens with:
     - Per-camera **Plücker Ray Embeddings** `(o × d, d)` expressed in the target chase camera frame,
     - Learnable **Camera ID** and **Relative Frame Time** embeddings,
     - A compact **Perceiver Resampler** (`N_latents = 256`) projecting `1152 → 4096` (`WanVACETransformer3DModel.text_dim`).
   - **Dual-Branch Policy Showcase:** Because the vision tower stays frozen, the full `Alpamayo2-Edge` (~2.65B, ~5.3 GB BF16) can optionally run alongside Reality View on a single 20 GB GPU to output 64-waypoint future trajectories rendered directly onto the generated third-person road!
   - **Ablation-Ready Interface:** `SharedVisionEncoder` is modular so we can directly ablate `Alpamayo2-Edge` vs. `DM0 / DM0.5` (`OpenDM`) vs. `DINOv3` without changing downstream code.

---

## 3. Verified Model Dimensions & VRAM Footprint (BF16)

| Component | Upstream ID | Verified Key Dimensions | BF16 Weight Size | Status in v1 |
| :--- | :--- | :--- | :---: | :--- |
| **Shared Vision Encoder** | `nvidia/Alpamayo2-Edge` (`vlm.vision_tower`) | 27L SigLIP2, `hidden=1152`, `patch=16`, `merge=2` | ~0.82 GB | Frozen (pre-cached for training) |
| **Optional Policy Branch** | `nvidia/Alpamayo2-Edge` (full VLA) | 2.65B total (`expert_config`, 64 waypoints @ 10Hz) | ~5.30 GB | Optional dual-branch demo |
| **3D Geometry Anchor** | `depth-anything/DA3-BASE` | Multi-view metric depth-ray transformer | ~0.45 GB | Frozen |
| **Semantic Bridge** | `AlpamayoContextAdapter` (ours) | Perceiver Resampler `1152 → 4096`, `256` tokens | ~0.12 GB | **Trainable** |
| **Diffusion Backbone** | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | 30L DiT, `hidden=1536`, `text_dim=4096`, `vace_in=96` | ~2.60 GB | **Trainable** (LoRA → Full FT) |
| **Causal 3D VAE** | `AutoencoderKLWan` (`8×8×4`) | `z_dim=16`, `4×` temporal = 1 latent per 4 RGB frames | ~0.50 GB | Frozen (v1) / `TAEHV` (v2) |
| **Total Inference Weights** | — | Without UMT5-XXL text encoder | **~4.5 GB** (~9 GB w/ full policy) | Fits comfortably on RTX 3080 20GB |

---

## 4. Stage 0: Third-Person Ground Truth (GT) Data Strategy

A major breakthrough for open-source replication is combining **pre-reconstructed neural driving scenes** with **synthetic zero-hole ground truth**:

1. **Tier 1 (Real-World Primary) — `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (`26.10` release):**
   - Contains **1,601 pre-reconstructed 20-second 3D neural driving scenes** (`.usdz` 3DGUT + `.xodr` HD maps + surface meshes, 7 cameras) — the exact dataset family used to train `Alpamayo2-Edge`!
   - Eliminates hundreds of GPU hours of per-scene 4DGS fitting. We render both synchronized ego surround views and randomized third-person chase/orbit cameras, composite a physically lit 3D ego-car mesh, and pass residual high-elevation artifacts through **`Difix3D+`** (single-step 3DGS artifact fixer).
2. **Tier 2 (Cold-Start / Exact-Geometry Benchmark) — `CARLA` / UE5:**
   - Captures 5–10 hours of multi-camera surround feeds paired with a **real virtual chase camera** (zero reconstruction holes, exact ego vehicle, exact ground-truth metric depth and dynamic lighting). Used for rapid pipeline debugging and exact PSNR/SSIM/LPIPS/geometry evaluation.
3. **Tier 3 (Optional Public Benchmark Transfer) — `nuScenes` / `Waymo` via `OmniRe` (`ziyc/drivestudio`) + `Difix3D+`:**
   - Used for cross-dataset generalization testing (e.g., 30–50 curated urban clips reconstructed with OmniRe's rigid + SMPL pedestrian + cyclist scene graphs).

---

## 5. Training & Distillation Pipeline (~500–600 A100h Budget)

### v1 — Quality-First Multi-Step Flow-Matching Teacher (~350–400 A100h)

1. **Stage 0: Dataset Rendering & Feature Pre-Caching (~80–100 A100h)**
   - Render paired surround + chase-view clips (`480×832`, 16–24 fps, 49–81 frame clips) + coarse DA3 projections + hole masks.
   - Pre-extract and shard `Alpamayo2-Edge` vision tokens and `AutoencoderKLWan` latents so training spends 100% of GPU VRAM on the DiT + adapter.
2. **Stage 1A: Adapter Warmup (~40–60 A100h, Colab A100 / PRO 6000)**
   - Freeze `WanVACETransformer3DModel`; train only `AlpamayoContextAdapter` + cross-attention projections with Flow-Matching velocity loss `E[||v_θ(x_t, t, c) - (x_1 - x_0)||^2]`.
3. **Stage 1B: Joint Quality Fine-Tuning (~180–220 A100h, Colab PRO 6000 96GB / H100)**
   - Unfreeze VACE control layers + DiT (high-rank LoRA `r=64` first, then full fine-tuning on PRO 6000 96GB) with **conditioning dropout** (randomly dropping/degrading coarse 3D projections so the model relies heavily on `Alpamayo2-Edge` tokens for blind spots, roofs, traffic lights, and lighting).
   - **v1 Inference:** 15–30 steps `UniPCMultistepScheduler` (`flow_shift=3.0` for 480p), outputting high-fidelity third-person videos with interactive viewpoint control.

### v2 — Real-Time Causal Mean-Flow Streaming (~150–200 A100h)

4. **Stage 2: Causal Chunking + `Causal-rCM` Mean-Flow Distillation (`NVlabs/rcm`)**
   - Convert v1 bidirectional teacher into a **4-frame causal chunk** model (`1` Wan latent frame per chunk = `133 ms` buffer @ 30 fps) with a **short sliding-window KV cache** (`2–4` latent chunks ≈ `0.5–1.0s`, avoiding stale-memory ghosting while maintaining temporal stability).
   - Apply **Causal-rCM** (continuous-time MeanFlow / sCM + Self-Forcing DMD) to compress sampling from 20–30 steps down to **1–2 steps**.
5. **Stage 3: Lightweight Decoder & Dual-Branch Live Demo**
   - Swap/distill `AutoencoderKLWan` decoder to a `TAEHV`-style fast decoder (`<10 ms` per 4-frame chunk) for real-time local inference on RTX 3080 20GB.
