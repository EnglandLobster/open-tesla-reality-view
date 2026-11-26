#!/usr/bin/env python3
"""Experiment: video-level GT refinement with Cosmos-Transfer2.5-2B vs per-frame Difix3D+.

Renders a Tesla-framed third-person chase clip (pure background, no ego car) from a NuRec 26.10
scene, then compares three versions of the same clip:

  raw     NuRec 3DGS render (what Stage 0 produces before refinement)
  difix   per-frame Difix3D+ (current Stage 0 refiner, front-wide reference view)
  cosmos  Cosmos-Transfer2.5-2B, whole clip at once, conditioned on the render via a controlnet
          (``blur`` = softened render, ``depth`` = render depth)

There is no ground-truth chase view, so ``eval`` reports reference-free proxies:

  flicker    mean |o_t - (o_{t-1} + o_{t+1}) / 2| on luminance (temporal 2nd difference; lower = steadier)
  fidelity   PSNR between low-passed output and low-passed raw render (higher = keeps layout/colors,
             i.e. less hallucination)
  sharpness  mean variance of the Laplacian (higher = more fine detail)

Stages (run in order; each reads the previous stage's files from ``--out-dir``):

  python scripts/exp_cosmos_transfer_refine.py render --scene bb5389f1
  python scripts/exp_cosmos_transfer_refine.py cosmos --scene bb5389f1 --control blur
  python scripts/exp_cosmos_transfer_refine.py cosmos --scene bb5389f1 --control depth
  python scripts/exp_cosmos_transfer_refine.py eval   --scene bb5389f1

Cosmos needs gated access to ``nvidia/Cosmos-Transfer2.5-2B`` and ``nvidia/Cosmos-1.0-Guardrail``
(the guardrail is required by the NVIDIA Open Model License) plus ``pip install cosmos_guardrail``.
On a 20 GB card the 7B Qwen2.5-VL text encoder runs on CPU once and its embeddings are cached.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCENES = {
    "bb5389f1": "data/nurec_samples/sample_set/26.10_release/"
    "bb5389f1-9514-4cfa-8322-dae897ed3786/bb5389f1-9514-4cfa-8322-dae897ed3786.usdz",
    "3582a449": "data/nurec_samples/sample_set/26.10_release/"
    "3582a449-6e1a-4dca-8c53-efc49a623b88/3582a449-6e1a-4dca-8c53-efc49a623b88.usdz",
}

# Tesla FSD v15 drone follow-cam framing (kept in sync with scripts/serve_visualizer.py)
TESLA_CHASE_RADIUS = 13.8
TESLA_CHASE_ELEV_DEG = 11.5
TESLA_CHASE_FOV_X_DEG = 74.0
TESLA_LOOK_AHEAD_X = 4.5
TESLA_LOOK_AT_Z = 1.25

COSMOS_ID = "nvidia/Cosmos-Transfer2.5-2B"
DEFAULT_PROMPT = (
    "A photorealistic video filmed by a drone smoothly following behind and above a car on a real city "
    "street, looking forward down the road. The road surface, lane markings, curbs, sidewalks, trees, "
    "buildings, parked cars and traffic are sharp and natural, with realistic daylight, shadows and "
    "reflections. The camera moves forward steadily with no shaking. High quality dashcam-like realism."
)
SCENE_PROMPTS = {
    "bb5389f1": (
        "A photorealistic video filmed by a drone smoothly following a car from behind and above along a wide "
        "suburban California arterial road at golden-hour sunset. The asphalt road has a double yellow center "
        "line, white lane markings, a bike lane and arrows. Tall eucalyptus trees and a low wall line the left "
        "side; palm trees, power lines, utility poles, houses and a parked silver sedan are on the right. A dark "
        "SUV drives ahead and a white car approaches in the opposite lane. Warm low sunlight, long soft shadows, "
        "sharp natural textures, realistic colors, steady forward camera motion."
    ),
}


# --------------------------------------------------------------------------------------------- io
def _save_png(arr_hwc: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr_hwc * 255.0, 0, 255).round().astype(np.uint8)).save(path)


def _load_frames(folder: Path) -> np.ndarray:
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No frames in {folder}")
    return np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0 for f in files])


def _write_mp4(frames_thwc: np.ndarray, path: Path, fps: float) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, (np.clip(frames_thwc, 0, 1) * 255).astype(np.uint8), fps=fps, quality=9)


# ----------------------------------------------------------------------------------------- render
def stage_render(args: argparse.Namespace) -> None:
    from open_reality_view.data.nurec_renderer import NuRecSceneRenderer
    from open_reality_view.geometry.projector import build_orbit_chase_pose
    from open_reality_view.rendering import Difix3DRefiner

    out = Path(args.out_dir)
    renderer = NuRecSceneRenderer(SCENES[args.scene])
    dev = renderer.device
    h, w = args.height, args.width
    fx = (w * 0.5) / math.tan(math.radians(TESLA_CHASE_FOV_X_DEG * 0.5))
    k_tgt = torch.tensor([[fx, 0.0, w * 0.5], [0.0, fx, h * 0.5], [0.0, 0.0, 1.0]], device=dev)
    chase_pose = build_orbit_chase_pose(
        radius=TESLA_CHASE_RADIUS,
        elevation_deg=TESLA_CHASE_ELEV_DEG,
        azimuth_deg=0.0,
        look_at_xyz=(renderer.ego_centroid[0] + TESLA_LOOK_AHEAD_X, 0.0, TESLA_LOOK_AT_Z),
        device=dev,
    )
    front_c2e = torch.from_numpy(renderer.cameras["camera_front_wide_120fov"].T_sensor_rig).to(dev)
    front_k = renderer.camera_intrinsics("camera_front_wide_120fov", h, w)
    refiner = Difix3DRefiner.from_pretrained("nvidia/difix_ref", device=dev)

    t_render = t_difix = 0.0
    for i in range(args.num_frames):
        ts_us = (args.start_sec + i / args.fps) * 1e6
        t0 = time.perf_counter()
        res = renderer.render_camera(ts_us, chase_pose, k_tgt, h, w, composite_ego=False)
        torch.cuda.synchronize()
        t_render += time.perf_counter() - t0
        _save_png(res.rgb.cpu().numpy(), out / "raw" / f"{i:04d}.png")
        (out / "depth").mkdir(parents=True, exist_ok=True)
        np.save(out / "depth" / f"{i:04d}.npy", res.depth.cpu().numpy().astype(np.float32))

        t0 = time.perf_counter()
        ref = renderer.render_camera(ts_us, front_c2e, front_k, h, w, composite_ego=False).rgb
        fixed = refiner.refine(res.rgb, ref)
        torch.cuda.synchronize()
        t_difix += time.perf_counter() - t0
        _save_png(fixed.cpu().numpy(), out / "difix" / f"{i:04d}.png")
        print(f"[render] frame {i + 1}/{args.num_frames}", flush=True)

    meta = {
        "scene": args.scene,
        "usdz": SCENES[args.scene],
        "num_frames": args.num_frames,
        "fps": args.fps,
        "start_sec": args.start_sec,
        "height": h,
        "width": w,
        "framing": {
            "radius_m": TESLA_CHASE_RADIUS,
            "elevation_deg": TESLA_CHASE_ELEV_DEG,
            "fov_x_deg": TESLA_CHASE_FOV_X_DEG,
        },
        "timing_s": {"render_total": t_render, "difix_total": t_difix},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[render] done -> {out}  render {t_render:.1f}s  difix {t_difix:.1f}s", flush=True)


# ----------------------------------------------------------------------------------------- cosmos
def _blur_control(raw_thwc: np.ndarray, strength: str) -> np.ndarray:
    """Approximate Cosmos 'blur' control: bilateral smoothing at reduced resolution, then upsampled.

    The official Transfer2.5 preprocessor is not shipped with diffusers; this keeps edges and colors
    while removing high-frequency texture (including 3DGS needle/floater artifacts).
    """
    import cv2

    if strength == "none":
        return raw_thwc
    down, iters = {"low": (2, 1), "medium": (4, 2), "high": (6, 3)}[strength]
    out = []
    for f in raw_thwc:
        u8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        h, w = u8.shape[:2]
        small = cv2.resize(u8, (w // down, h // down), interpolation=cv2.INTER_AREA)
        for _ in range(iters):
            small = cv2.bilateralFilter(small, d=9, sigmaColor=60, sigmaSpace=9)
        out.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0)
    return np.stack(out)


def _depth_control(depth_dir: Path) -> np.ndarray:
    """Inverse depth, normalised over the whole clip (near = bright), as a 3-channel video."""
    d = np.stack([np.load(f) for f in sorted(depth_dir.glob("*.npy"))])
    valid = np.isfinite(d) & (d > 0.1)
    inv = np.where(valid, 1.0 / np.clip(d, 0.5, 300.0), 0.0)
    lo, hi = np.percentile(inv[valid], 1), np.percentile(inv[valid], 99)
    inv = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.repeat(inv[..., None], 3, axis=-1).astype(np.float32)


def stage_cosmos(args: argparse.Namespace) -> None:
    from diffusers import AutoModel, Cosmos2_5_TransferPipeline
    from diffusers.pipelines.cosmos.pipeline_cosmos2_5_transfer import DEFAULT_NEGATIVE_PROMPT

    out = Path(args.out_dir)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    raw = _load_frames(out / "raw")
    if args.max_frames:
        raw = raw[: args.max_frames]
    if args.control == "blur":
        ctrl = _blur_control(raw, args.blur_strength)
    else:
        ctrl = _depth_control(out / "depth")[: len(raw)]
    tag = f"cosmos_{args.control}" + (f"_{args.blur_strength}" if args.control == "blur" else "") + f"_w{args.control_scale:g}"
    if args.diag_name:  # diagnostic runs live outside the cosmos_* glob used by `eval`
        tag = f"diag_{args.diag_name}"
    _write_mp4(ctrl, out / f"control_{tag}.mp4", meta["fps"])
    controls = [Image.fromarray((f * 255).astype(np.uint8)) for f in ctrl]

    t0 = time.perf_counter()
    controlnet = AutoModel.from_pretrained(
        COSMOS_ID, revision=f"diffusers/controlnet/general/{args.control}", torch_dtype=torch.bfloat16
    )
    pipe = Cosmos2_5_TransferPipeline.from_pretrained(
        COSMOS_ID, controlnet=controlnet, revision="diffusers/general", torch_dtype=torch.bfloat16
    )
    t_load = time.perf_counter() - t0

    # Encode prompts once. The 7B text encoder (~16.6 GB) does not fit next to other GPU work on a
    # 20 GB card, so it runs on CPU unless --text-encoder-gpu is set; embeddings are cached to disk.
    emb_path = out / "prompt_embeds.pt"
    do_cfg = args.guidance_scale > 1.0
    cached = torch.load(emb_path) if emb_path.is_file() else None
    # Cache key covers both prompts: an empty negative prompt (encode_prompt's default) produced heavy
    # brush-stroke artifacts, so the pipeline's official DEFAULT_NEGATIVE_PROMPT is always used.
    if cached is not None and cached.get("prompt") == args.prompt and cached.get("negative") == DEFAULT_NEGATIVE_PROMPT:
        pe, npe = cached["pe"], cached["npe"]
    else:
        te_dev = torch.device("cuda" if args.text_encoder_gpu else "cpu")
        pipe.text_encoder.to(te_dev)
        t1 = time.perf_counter()
        with torch.no_grad():
            pe, npe = pipe.encode_prompt(
                args.prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                do_classifier_free_guidance=True,
                device=te_dev,
                dtype=torch.bfloat16,
            )
        print(f"[cosmos] prompt encoded on {te_dev} in {time.perf_counter() - t1:.1f}s", flush=True)
        pipe.text_encoder.to("cpu")
        pe, npe = pe.cpu(), npe.cpu()
        torch.save({"pe": pe, "npe": npe, "prompt": args.prompt, "negative": DEFAULT_NEGATIVE_PROMPT}, emb_path)
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    # Full-clip 704x1280 VAE decode peaks above what a 20 GB card has left; decode in spatial tiles.
    pipe.vae.enable_tiling()
    # User-supplied embeddings are not moved by the pipeline; put them on the execution device.
    pe, npe = pe.to("cuda", torch.bfloat16), npe.to("cuda", torch.bfloat16)

    tag_latents = out / f"latents_{args.control}.pt"

    def _save_final_latents(_pipe, step: int, _t, kw: dict) -> dict:
        # Persist denoised latents so a decode failure does not cost the ~10 min denoise again.
        if step == args.steps - 1:
            torch.save(kw["latents"].detach().cpu(), tag_latents)
        return kw

    torch.cuda.reset_peak_memory_stats()
    t2 = time.perf_counter()
    result = pipe(
        callback_on_step_end=_save_final_latents,
        callback_on_step_end_tensor_inputs=["latents"],
        controls=controls,
        controls_conditioning_scale=args.control_scale,
        prompt_embeds=pe,
        negative_prompt_embeds=npe if do_cfg else None,
        height=meta["height"],
        width=meta["width"],
        num_frames=len(controls),
        num_frames_per_chunk=len(controls),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        output_type="np",
    )
    t_gen = time.perf_counter() - t2
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    video = np.asarray(result.frames[0], dtype=np.float32)  # [T, H, W, 3] in [0, 1]
    if video.max() > 1.5:
        video = video / 255.0
    for i, f in enumerate(video):
        _save_png(f, out / tag / f"{i:04d}.png")

    meta.setdefault("timing_s", {})[tag] = {"load": t_load, "generate": t_gen}
    meta.setdefault("cosmos_runs", {})[tag] = {
        "control": args.control,
        "blur_strength": args.blur_strength if args.control == "blur" else None,
        "control_scale": args.control_scale,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "peak_vram_gb": peak_gb,
        "prompt": args.prompt,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[cosmos] {tag}: {t_gen:.1f}s for {len(video)} frames, peak VRAM {peak_gb:.1f} GB", flush=True)


# ------------------------------------------------------------------------------------------- eval
def _luma(x: torch.Tensor) -> torch.Tensor:  # [T, 3, H, W] -> [T, 1, H, W]
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = int(3 * sigma)
    k = torch.exp(-((torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)) ** 2) / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def _metrics(video: np.ndarray, raw: np.ndarray) -> dict[str, float]:
    v = torch.from_numpy(video).permute(0, 3, 1, 2).cuda()
    r = torch.from_numpy(raw).permute(0, 3, 1, 2).cuda()
    y = _luma(v)
    flicker = (y[1:-1] - 0.5 * (y[:-2] + y[2:])).abs().mean().item() * 255.0
    # Low-passed flicker separates large-area instability from fine texture changing under motion.
    lp = {}
    for sigma in (2.0, 6.0):
        ys = _gauss_blur(y, sigma)
        lp[sigma] = (ys[1:-1] - 0.5 * (ys[:-2] + ys[2:])).abs().mean().item() * 255.0
    lv, lr = _gauss_blur(v, 4.0), _gauss_blur(r, 4.0)
    mse = ((lv - lr) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    fidelity = (10 * torch.log10(1.0 / mse)).mean().item()
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    sharp = F.conv2d(y * 255.0, lap).var(dim=(1, 2, 3)).mean().item()
    return {
        "flicker": flicker,
        "flicker_lowpass_s2": lp[2.0],
        "flicker_lowpass_s6": lp[6.0],
        "fidelity_psnr_lowpass": fidelity,
        "sharpness_lap_var": sharp,
    }


def stage_eval(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    raw = _load_frames(out / "raw")
    refiner_prefixes = ("cosmos_", "artifixer", "harmonizer", "fixer")
    methods = ["raw", "difix"] + sorted(p.name for p in out.iterdir() if p.is_dir() and p.name.startswith(refiner_prefixes))
    videos = {m: _load_frames(out / m) for m in methods}

    rows = {m: _metrics(videos[m], raw) for m in methods}
    rows["raw"]["fidelity_psnr_lowpass"] = float("inf")
    (out / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n{'method':<30}{'flicker ↓':>11}{'σ2 ↓':>8}{'σ6 ↓':>8}{'fidelity dB ↑':>15}{'sharpness':>11}")
    for m, r in rows.items():
        print(
            f"{m:<30}{r['flicker']:>11.3f}{r['flicker_lowpass_s2']:>8.3f}{r['flicker_lowpass_s6']:>8.3f}"
            f"{r['fidelity_psnr_lowpass']:>15.2f}{r['sharpness_lap_var']:>11.1f}"
        )

    # Full-resolution per-method videos (used by the visualizer's synchronized A/B player)
    for m in methods:
        _write_mp4(videos[m], out / f"video_{m}.mp4", meta["fps"])

    # Side-by-side comparison: one row per method, columns = first / middle / last frame
    t_idx = [0, len(raw) // 2, len(raw) - 1]
    thumbs_w = 640
    rows_img = []
    for m in methods:
        tiles = []
        for t in t_idx:
            im = Image.fromarray((videos[m][t] * 255).astype(np.uint8))
            tiles.append(np.asarray(im.resize((thumbs_w, int(im.height * thumbs_w / im.width)))))
        rows_img.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows_img, axis=0)).save(out / "comparison_grid.jpg", quality=92)

    # Stacked comparison video (methods top-to-bottom, half resolution)
    def _half(f: np.ndarray) -> np.ndarray:
        im = Image.fromarray((f * 255).astype(np.uint8))
        return np.asarray(im.resize((thumbs_w, int(im.height * thumbs_w / im.width))))

    stack = np.concatenate([np.stack([_half(f) for f in videos[m]]) for m in methods], axis=1)
    stack = stack.astype(np.float32) / 255.0
    _write_mp4(stack, out / "comparison.mp4", meta["fps"])
    print(f"\nmethods (top→bottom): {methods}\nwrote {out / 'comparison_grid.jpg'} and {out / 'comparison.mp4'}")


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["render", "cosmos", "eval"])
    p.add_argument("--scene", choices=sorted(SCENES), default="bb5389f1")
    p.add_argument("--out-dir", default=None, help="default: outputs/exp_cosmos_transfer/<scene>")
    # render
    p.add_argument("--num-frames", type=int, default=29, help="1 + 4k so the Cosmos VAE aligns")
    p.add_argument("--fps", type=float, default=16.0)
    p.add_argument("--start-sec", type=float, default=0.85)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    # cosmos
    p.add_argument("--control", choices=["blur", "depth"], default="blur")
    p.add_argument("--blur-strength", choices=["none", "low", "medium", "high"], default="low")
    p.add_argument("--control-scale", type=float, default=0.5, help="official default for blur/depth is 0.5")
    p.add_argument("--steps", type=int, default=36)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default=None, help="default: per-scene prompt in SCENE_PROMPTS")
    p.add_argument("--text-encoder-gpu", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="cosmos: use only the first N frames (1 + 4k)")
    p.add_argument("--diag-name", default="", help="cosmos: write to diag_<name>/ instead of cosmos_*/")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = f"outputs/exp_cosmos_transfer/{args.scene}"
    if args.prompt is None:
        args.prompt = SCENE_PROMPTS.get(args.scene, DEFAULT_PROMPT)
    if args.stage == "render" and (args.num_frames - 1) % 4 != 0:
        p.error("--num-frames must be 1 + 4k")
    {"render": stage_render, "cosmos": stage_cosmos, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    main()
