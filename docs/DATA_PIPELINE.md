# Stage 0 Data Pipeline: Multi-View Surround → Third-Person GT

This document specifies how to produce standardized `DrivingClip` directories consumed by `scripts/precache_features.py` and `scripts/train_v1.py`.

---

## 1. Unified Clip Directory Contract

Every Stage 0 source (`PhysicalAI-AV-NuRec`, `CARLA`, `nuScenes/Waymo + OmniRe + Difix3D+`, or synthetic) exports to the exact same layout:

```text
data/clips/<clip_id>/
├── metadata.json
├── surround/
│   ├── cam_front/000000.png ... 000048.png
│   ├── cam_front_left/...
│   ├── cam_front_right/...
│   ├── cam_back/...
│   ├── cam_back_left/...
│   └── cam_back_right/...
├── chase_gt/
│   └── 000000.png ... 000048.png       # 480x832 third-person ground truth (with 3D ego car)
└── depths/                             # Optional; generated via scripts/estimate_depth_da3.py if absent
    └── <cam_name>/000000.npy ...       # float32 metric depth in meters, shape [H_src, W_src]
```

### `metadata.json` Schema

```json
{
  "camera_names": [
    "cam_front", "cam_front_left", "cam_front_right",
    "cam_back", "cam_back_left", "cam_back_right"
  ],
  "fps": 16,
  "src_intrinsics": "float[T, K, 3, 3]",
  "src_c2e": "float[T, K, 4, 4]  (OpenCV cam-to-ego; ego +X fwd, +Y left, +Z up)",
  "tgt_intrinsics": "float[T, 3, 3]",
  "tgt_c2e": "float[T, 4, 4]     (target virtual chase/orbit camera pose in ego frame)"
}
```

> **Frame Count Rule (`AutoencoderKLWan`):** Always export clips with `T = 1 + 4 * k` frames (e.g., `17`, `33`, `49`, or `81`) so Wan's 4× causal temporal compression aligns cleanly with `F_lat = 1 + k`.

---

## 2. Three Data Tiers (Recommended Progression)

### Tier 0: Zero-Download Pipeline Verification (Built-In)

Validates pre-caching, Stage 1A/1B training, and inference in minutes on a single RTX 3080 20GB:

```bash
python scripts/precache_features.py \
  --synthetic-demo 8 \
  --num-frames 17 \
  --height 480 --width 832 \
  --out-dir data/cached_shards_synth
```

### Tier 1 (Primary Real-World): `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`

- **Why Primary:** `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (`26.10` release) provides **1,601 pre-reconstructed 20-second 7-camera 3D neural driving scenes** (`.usdz` 3DGUT + `.xodr` HD maps + surface meshes) — the exact dataset ecosystem `nvidia/Alpamayo2-Edge` was trained on.
- **Workflow:**
  1. Request instant auto-approval on Hugging Face: `https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`.
  2. Render synchronized 6–7 ego surround views + randomized third-person chase/orbit cameras (`build_chase_camera_pose` / `build_orbit_chase_pose`) from each `.usdz` scene using NVIDIA NuRec / `3DGRUT`.
  3. Composite a physically lit 3D ego-car mesh (or use `EgoVehicleCompositor`) to fill the sensor blind-spot ego-car hole.
  4. Run single-step **`Difix3D+`** over high-elevation chase renders to strip roof/occlusion floaters before saving to `chase_gt/`.
  5. Run `scripts/estimate_depth_da3.py` on `surround/` (or export rendered depth directly from NuRec) and pre-cache with `scripts/precache_features.py`.

### Tier 2: `CARLA` Synthetic Chase-Camera Benchmark & `nuScenes` / `Waymo` (`OmniRe`)

- **CARLA:** Attach 6 calibrated RGB+Depth sensors on the ego vehicle plus 1 real virtual chase camera at `offset_xyz = (-7.5, 0.0, 4.2), pitch = -18°` (plus random orbit sweeps). Zero reconstruction holes, exact ego vehicle, ideal for quantitative PSNR/SSIM/LPIPS benchmarking.
- **nuScenes / Waymo:** Reconstruct 30–50 curated urban clips with `OmniRe` (`ziyc/drivestudio`, rigid cars + SMPL pedestrians + non-rigid cyclists), composite the ego vehicle, refine elevated views with `Difix3D+`, and export to the unified clip layout above.

---

## 3. Depth Estimation & Feature Pre-Caching

```bash
# Step A (if depths/ not exported from simulator/NuRec): predict metric depth with DA3
python scripts/estimate_depth_da3.py \
  --clips-dir data/clips \
  --model-id depth-anything/DA3-BASE

# Step B: Pre-cache frozen Alpamayo2-Edge tokens + Wan2.1 VAE / VACE control latents
python scripts/precache_features.py \
  --clips-dir data/clips \
  --out-dir data/cached_shards
```
