#!/usr/bin/env python3
"""Intranet interactive visualizer v5 (Tesla-Aligned Native 720p/1080p HD).

Features:
  1. Native 720p (1280x720) & live 1080p (1920x1080) rendering with 4:4:4 JPEG (zero downscale blur)
  2. Background-first Difix3D+ (native 720p) -> razor-sharp 4x MSAA 213k PBR GLB ego car composited AFTER VAE
  3. Tesla FSD v15 drone follow-cam framing (~28% sky/horizon, full uncropped ego car & grounded shadow)
  4. Tesla-style borderless glowing electric-blue road trajectory carpet (depth-tested)
  5. Causal multi-frame 3D Cache (~31% -> ~5% blind spots) + live 360-deg RTX 3080 orbit controller
"""

from __future__ import annotations

import argparse
import io
import json
import math
import threading
import time
import urllib.parse
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from open_reality_view.data.nurec_renderer import NuRecSceneRenderer
from open_reality_view.geometry import (
    MultiViewDepthProjector,
    Temporal3DCacheProjector,
    build_orbit_chase_pose,
    composite_trajectory_ribbon,
)
from open_reality_view.rendering import Difix3DRefiner

DEFAULT_CA_USDZ = (
    "data/nurec_samples/sample_set/26.10_release/"
    "bb5389f1-9514-4cfa-8322-dae897ed3786/bb5389f1-9514-4cfa-8322-dae897ed3786.usdz"
)
ALT_HIGHWAY_USDZ = (
    "data/nurec_samples/sample_set/26.10_release/"
    "3582a449-6e1a-4dca-8c53-efc49a623b88/3582a449-6e1a-4dca-8c53-efc49a623b88.usdz"
)

# Tesla FSD v15 drone follow-cam defaults: ~28% sky/horizon visible + full uncropped ego car at bottom-center
TESLA_CHASE_RADIUS = 13.8
TESLA_CHASE_ELEV_DEG = 11.5
TESLA_CHASE_FOV_X_DEG = 74.0
TESLA_LOOK_AHEAD_X = 4.5
TESLA_LOOK_AT_Z = 1.25


def _to_pil(rgb_hwc: torch.Tensor | np.ndarray) -> Image.Image:
    arr = rgb_hwc.detach().cpu().numpy() if isinstance(rgb_hwc, torch.Tensor) else rgb_hwc
    u8 = np.clip(arr * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(u8)


def _save_hd_jpeg(img: Image.Image, path: Path, quality: int = 95) -> None:
    img.save(path, format="JPEG", quality=quality, subsampling=0)


def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _make_surround_mosaic(surround_rgb: dict[str, torch.Tensor], cam_names: tuple[str, ...]) -> Image.Image:
    tiles: list[Image.Image] = []
    short_labels = {
        "camera_front_wide_120fov": "FRONT WIDE 120°",
        "camera_cross_left_120fov": "CROSS LEFT 120°",
        "camera_cross_right_120fov": "CROSS RIGHT 120°",
        "camera_rear_tele_30fov": "REAR TELE 30°",
        "camera_rear_left_70fov": "REAR LEFT 70°",
        "camera_rear_right_70fov": "REAR RIGHT 70°",
    }
    tw, th = 426, 240
    for cam in cam_names[:6]:
        im = _to_pil(surround_rgb[cam]).resize((tw, th), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(im)
        label = short_labels.get(cam, cam)
        draw.rectangle([6, 6, 6 + len(label) * 7 + 10, 24], fill=(10, 14, 22))
        draw.text((10, 9), label, fill=(225, 240, 255))
        tiles.append(im)

    mosaic = Image.new("RGB", (tw * 3, th * 2), (10, 12, 18))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, 3)
        mosaic.paste(tile, (c * tw, r * th))
    return mosaic


def _make_mask_overlay(coarse_chw: torch.Tensor, hole_mask_1hw: torch.Tensor) -> Image.Image:
    rgb = coarse_chw.detach().cpu().float().permute(1, 2, 0).numpy()
    mask = hole_mask_1hw[0].detach().cpu().float().numpy()
    tint = np.zeros_like(rgb)
    tint[..., 0] = 0.92
    tint[..., 1] = 0.16
    tint[..., 2] = 0.24
    blended = np.where(mask[..., None] > 0.5, rgb * 0.35 + tint * 0.65, rgb)
    return _to_pil(blended)


def _pinhole_k(hw: tuple[int, int], fov_x_deg: float, device: torch.device) -> torch.Tensor:
    h, w = hw
    fx = (w * 0.5) / math.tan(math.radians(fov_x_deg * 0.5))
    return torch.tensor(
        [[fx, 0.0, w * 0.5], [0.0, fx, h * 0.5], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Open Tesla Reality View — 内网高清可视化控制台 v5 (原生 720p/1080p · 对齐 FSD v15)</title>
<style>
  :root {
    --bg: #070a10;
    --panel: #101522;
    --border: #212b40;
    --text: #e8eef9;
    --muted: #8a98b6;
    --accent: #2563eb;
    --cyan: #38bdf8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: radial-gradient(circle at top, #0f182c 0%, var(--bg) 55%);
    color: var(--text);
    line-height: 1.5;
    padding: 18px;
  }
  .wrap { max-width: 1480px; margin: 0 auto; }
  header {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 14px; padding: 16px 22px; background: rgba(16,21,34,0.94);
    border: 1px solid var(--border); border-radius: 14px; margin-bottom: 16px;
  }
  .title h1 { font-size: 21px; font-weight: 700; letter-spacing: 0.2px; }
  .title p { font-size: 13px; color: var(--muted); margin-top: 3px; }
  .pills { display: flex; flex-wrap: wrap; gap: 8px; }
  .pill {
    font-size: 12px; padding: 5px 10px; border-radius: 999px;
    background: #172035; border: 1px solid #2b3856; color: #c7d6f5;
  }
  .pill.ok { border-color: #14532d; background: #0b2919; color: #6ee7b7; }
  .pill.purple { border-color: #5b21b6; background: #1e1238; color: #c4b5fd; }
  .pill.cyan { border-color: #0369a1; background: #082f49; color: #7dd3fc; }
  .tabs { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }
  .tab {
    padding: 10px 18px; border-radius: 10px; cursor: pointer;
    background: var(--panel); border: 1px solid var(--border); color: var(--muted);
    font-size: 14px; font-weight: 600; transition: all 0.15s;
  }
  .tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  .section { display: none; }
  .section.active { display: block; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px; margin-bottom: 16px;
  }
  .card h2 { font-size: 16px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  .sub { font-size: 12px; color: var(--muted); font-weight: 400; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 12px 14px; background: #0b0f1a; border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 14px;
  }
  button {
    background: #1c2740; color: var(--text); border: 1px solid #304166;
    padding: 7px 14px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600;
  }
  button:hover { background: #263657; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.active-toggle { background: #059669; border-color: #10b981; color: #fff; }
  button.cyan-toggle { background: #0284c7; border-color: #38bdf8; color: #fff; }
  input[type=range] { accent-color: var(--accent); cursor: pointer; }
  .grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
  @media (max-width: 980px) { .grid2 { grid-template-columns: 1fr; } }
  .imgbox {
    background: #06090f; border: 1px solid var(--border); border-radius: 10px;
    overflow: hidden; position: relative;
  }
  .imgbox img { width: 100%; display: block; object-fit: contain; }
  .imglabel {
    position: absolute; top: 8px; left: 8px; font-size: 12px; font-weight: 600;
    padding: 4px 10px; border-radius: 6px; background: rgba(8,11,18,0.84);
    border: 1px solid rgba(255,255,255,0.14);
  }
  .badge-red { color: #fca5a5; }
  .badge-purple { color: #d8b4fe; }
  .badge-cyan { color: #7dd3fc; }
  .orbit-layout { display: grid; grid-template-columns: 1.8fr 1fr; gap: 16px; }
  @media (max-width: 1120px) { .orbit-layout { grid-template-columns: 1fr; } }
  .slider-row { margin-bottom: 12px; }
  .slider-row label { display: flex; justify-content: space-between; font-size: 13px; color: #c7d6f5; margin-bottom: 4px; }
  .slider-row input[type=range] { width: 100%; }
  .presets { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .compare-wrap {
    position: relative; width: 100%; max-width: 1280px; margin: 0 auto;
    border-radius: 12px; overflow: hidden; border: 1px solid var(--border); user-select: none;
  }
  .compare-wrap img { width: 100%; display: block; }
  .compare-top {
    position: absolute; inset: 0; overflow: hidden; width: 50%;
    border-right: 2px solid #fff;
  }
  .compare-top img { width: 100%; height: 100%; object-fit: cover; max-width: none; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
  .metric { background: #0c101b; border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
  .metric .k { font-size: 12px; color: var(--muted); }
  .metric .v { font-size: 22px; font-weight: 700; margin-top: 4px; color: #93c5fd; }
  .metric .d { font-size: 12px; color: #6ee7b7; margin-top: 2px; }
  .drag-hint {
    position: absolute; bottom: 10px; right: 10px; font-size: 12px;
    background: rgba(8,11,18,0.82); padding: 4px 10px; border-radius: 6px; color: #cbd5e1;
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="title">
      <h1>Open Tesla Reality View — 原生高清可视化控制台 v5（对齐 FSD v15）</h1>
      <p id="sceneMeta">加载场景信息中…</p>
    </div>
    <div class="pills">
      <span class="pill cyan">📺 原生 1280×720 HD / 1080p 实时</span>
      <span class="pill ok">🚗 自车后合成（零 VAE 模糊）</span>
      <span class="pill purple">✨ Difix3D+ 720p 原生直出</span>
      <span class="pill">🧭 特斯拉同款平视机位 + 柔光电光蓝光带</span>
    </div>
  </header>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('compare_difix', this)">✨ 1. 原生 720p 高清对比（对齐特斯拉构图）</div>
    <div class="tab" onclick="switchTab('instant_nurec', this)">⚡ 2. Instant-NuRec 实测对比（零离线地图）</div>
    <div class="tab" onclick="switchTab('timeline', this)">🔥 3. 同步时间轴回放（4面板 · 多帧 3D Cache）</div>
    <div class="tab" onclick="switchTab('orbit', this)">4. 实时 360° 自由视角（支持 720p / 1080p）</div>
    <div class="tab" onclick="switchTab('compare_front', this)">5. 前视原图 vs 自研3DGS</div>
    <div class="tab" onclick="switchTab('benchmarks', this)">6. v1 显存与实测指标</div>
  </div>

  <!-- SECTION 0 (DEFAULT): 720p HD TESLA-ALIGNED CURTAIN -->
  <div id="sec-compare_difix" class="section active">
    <div class="card">
      <h2>
        <span>原生 1280×720 HD：裸 3DGS（左） vs Difix3D+ 720p 街景修复 + 无损 4×MSAA PBR 自车 + 电光蓝光带（右）</span>
        <span class="sub" id="difixFrameLabel">拖动滑块直观对比清晰度与细节</span>
      </h2>
      <div class="controls">
        <button id="btnDiffChase" class="primary" onclick="setDifixMode('chase')">✨ 特斯拉平视追尾（720p · 10 帧）</button>
        <button id="btnDiffOblique" onclick="setDifixMode('oblique')">30° 侧后斜视特写（720p）</button>
        <button id="btnDiffCache" onclick="setDifixMode('cache')">🔥 单帧投影 vs 多帧 3D Cache</button>
        <span style="font-size:13px; margin-left:6px">卷帘位置：</span>
        <input type="range" id="splitDifix" min="0" max="100" value="35" style="flex:1; min-width:140px" oninput="setSplit('difix', this.value)" />
        <span style="font-size:13px; margin-left:8px" id="difixFrameWrap">切换帧：
          <input type="range" id="difixFrameSlider" min="0" max="9" value="3" style="width:140px; vertical-align:middle" oninput="setDifixCompareFrame(parseInt(this.value))" />
        </span>
      </div>
      <div class="compare-wrap" id="wrap-difix">
        <img id="difixRight" src="/assets/chase_difix_0003.jpg" alt="Right comparison" />
        <div class="compare-top" id="top-difix" style="width:35%">
          <img id="difixLeft" src="/assets/chase_raw_0003.jpg" alt="Left comparison" />
        </div>
        <div class="imglabel" id="lblCurtainLeft" style="top:10px; left:10px">裸 NuRec 3DGS（720p 原图）</div>
        <div class="imglabel badge-purple" id="lblCurtainRight" style="top:10px; left:auto; right:10px">✨ Difix3D+ 720p + 无损 4×MSAA PBR 自车 + 柔光蓝带</div>
      </div>
    </div>
  </div>

  <!-- SECTION: INSTANT-NUREC ZERO-OFFLINE-MAP BENCHMARK -->
  <div id="sec-instant_nurec" class="section">
    <div class="card">
      <h2>
        <span>⚡ NVIDIA Instant-NuRec (0.68GB) 零离线地图 / 零训练实测对比（Frame 9 · t=4.05s · 720p）</span>
        <span class="sub" id="inurecSub">仅输入 5 路因果环视相机画面（504×280）— 6帧 867ms / 3.86GB 显存 · 12帧 1.73s / 6.81GB 显存</span>
      </h2>
      <div class="controls">
        <button id="btnInurecVsGt" class="primary" onclick="setInurecMode('vs_gt')">⚡ 零训练成品（Instant-NuRec+Difix） vs 离线 NuRec GT</button>
        <button id="btnInurecVsDa3" onclick="setInurecMode('vs_da3')">🔥 旧 Branch A（DA3 点云） vs Instant-NuRec 原始 3DGS</button>
        <span style="font-size:13px; margin-left:6px">卷帘位置：</span>
        <input type="range" id="splitInurec" min="0" max="100" value="50" style="flex:1; min-width:140px" oninput="setSplit('inurec', this.value)" />
      </div>
      <div class="compare-wrap" id="wrap-inurec">
        <img id="inurecRight" src="/assets/chase_difix_0008.jpg" alt="Instant-NuRec right" />
        <div class="compare-top" id="top-inurec" style="width:50%">
          <img id="inurecLeft" src="/assets/inurec_difix_720p.jpg" alt="Instant-NuRec left" />
        </div>
        <div class="imglabel badge-cyan" id="lblInurecLeft" style="top:10px; left:10px">⚡ Instant-NuRec + Difix3D+ + PBR 自车（零离线地图 / 零训练）</div>
        <div class="imglabel badge-purple" id="lblInurecRight" style="top:10px; left:auto; right:10px">🎯 离线 NuRec 26.10 GT + Difix3D+（70 分钟离线精建）</div>
      </div>
    </div>

    <div class="card">
      <h2>
        <span>四图同屏高清对比（点击任意图片可在新标签页看 1280×720 原图）</span>
        <span class="sub">覆盖率：DA3 点云 ~91% → Instant-NuRec 前馈 3DGS 96.1%</span>
      </h2>
      <div class="grid2">
        <div class="imgbox">
          <div class="imglabel">① 旧 Branch A：DA3 多帧点云缓存（锯齿与黑洞明显）</div>
          <a href="/assets/coarse_cache_0008.jpg" target="_blank"><img src="/assets/coarse_cache_0008.jpg" alt="DA3 multi-frame cache" /></a>
        </div>
        <div class="imgbox">
          <div class="imglabel badge-cyan">② Instant-NuRec 原始 3DGS（0.68GB · 1.73s · 覆盖率 96.1%）</div>
          <a href="/assets/inurec_raw_720p.jpg" target="_blank"><img src="/assets/inurec_raw_720p.jpg" alt="Instant-NuRec raw" /></a>
        </div>
        <div class="imgbox">
          <div class="imglabel badge-purple">③ Instant-NuRec + Difix3D+ + PBR 自车（零离线地图 / 零训练）</div>
          <a href="/assets/inurec_difix_720p.jpg" target="_blank"><img src="/assets/inurec_difix_720p.jpg" alt="Instant-NuRec + Difix3D+" /></a>
        </div>
        <div class="imgbox">
          <div class="imglabel" style="color:#6ee7b7">④ 离线 NuRec 26.10 GT + Difix3D+（70 分钟离线扫描真值）</div>
          <a href="/assets/chase_difix_0008.jpg" target="_blank"><img src="/assets/chase_difix_0008.jpg" alt="Offline NuRec GT" /></a>
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 1: TIMELINE -->
  <div id="sec-timeline" class="section">
    <div class="card">
      <h2>
        <span>真实晴天高分场景同步回放（原生 720p · 多帧 3D Cache + PBR 自车）</span>
        <span class="sub" id="frameInfo">Frame 1 / 10</span>
      </h2>
      <div class="controls">
        <button class="primary" id="playBtn" onclick="togglePlay()">▶ 播放</button>
        <button onclick="stepFrame(-1)">◀ 上一帧</button>
        <button onclick="stepFrame(1)">下一帧 ▶</button>
        <input type="range" id="frameSlider" min="0" max="9" value="0" style="flex:1; min-width:140px" oninput="setFrame(parseInt(this.value))" />
        <button id="btnCacheToggle" class="cyan-toggle" onclick="toggleTimelineCache()">🔥 Branch A：多帧 3D Cache</button>
        <button id="btnDifixToggle" class="active-toggle" onclick="toggleTimelineDifix()">✨ 第四格：Difix3D+ 720p</button>
        <select id="fpsSelect" style="background:#182136;color:#fff;border:1px solid #304166;border-radius:6px;padding:6px 10px" onchange="restartTimerIfPlaying()">
          <option value="2">2 FPS</option>
          <option value="4" selected>4 FPS</option>
          <option value="8">8 FPS</option>
        </select>
      </div>
      <div class="grid2">
        <div class="imgbox">
          <div class="imglabel">① 输入：6 路车外环视相机（送入 Alpamayo2-Edge）</div>
          <img id="imgSurround" src="" alt="6-cam surround" />
        </div>
        <div class="imgbox">
          <div class="imglabel badge-purple" id="lblChase">④ 目标真值：720p Difix3D+ 街景 + 无损 PBR 自车 + 电光蓝带</div>
          <img id="imgChase" src="" alt="chase gt" />
        </div>
        <div class="imgbox">
          <div class="imglabel badge-cyan" id="lblCoarse">② Branch A：多帧因果 3D Cache 投影</div>
          <img id="imgCoarse" src="" alt="coarse 3d projection" />
        </div>
        <div class="imgbox">
          <div class="imglabel badge-red" id="lblMask">③ 盲区 Mask 叠层（红色 = 交由 Wan2.1-VACE 补全）</div>
          <img id="imgMask" src="" alt="blind spot mask" />
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 2: LIVE ORBIT -->
  <div id="sec-orbit" class="section">
    <div class="card">
      <h2>
        <span>交互式第三人称自由视角（RTX 3080 实时 720p / 1080p · PBR 自车零糊后合成）</span>
        <span class="sub" id="orbitStatus">就绪 — 支持按住鼠标左键直接拖拽旋转</span>
      </h2>
      <div class="orbit-layout">
        <div class="imgbox" id="orbitCanvas" style="cursor:grab; min-height:420px">
          <div class="imglabel" id="orbitLabel">Azimuth 0° | Elevation 11.5° | Radius 13.8m | t=2.0s</div>
          <img id="imgOrbit" src="/assets/chase_difix_0003.jpg" alt="live orbit render" draggable="false" />
          <div class="drag-hint">按住左键拖拽旋转视角</div>
        </div>
        <div style="background:#0c101b; border:1px solid var(--border); border-radius:12px; padding:16px">
          <div style="font-size:13px; color:var(--muted); margin-bottom:8px">快速预设机位（对齐特斯拉 FSD v15）</div>
          <div class="presets">
            <button onclick="applyPreset(0, 11.5, 13.8)">✨ Tesla 标配平视</button>
            <button onclick="applyPreset(-24, 14, 13.5)">左后 -24°</button>
            <button onclick="applyPreset(28, 15, 13.5)">右后 +28°</button>
            <button onclick="applyPreset(0, 38, 16.0)">高空俯视 38°</button>
            <button onclick="applyPreset(142, 12, 11.5)">右前侧看自车</button>
          </div>
          <div class="slider-row">
            <label><span>环绕方位角 (Azimuth)</span><span id="valAz">0°</span></label>
            <input type="range" id="slAz" min="-180" max="180" step="5" value="0" oninput="updateSliderLabels()" onchange="requestLiveRender()" />
          </div>
          <div class="slider-row">
            <label><span>俯仰仰角 (Elevation)</span><span id="valEl">11.5°</span></label>
            <input type="range" id="slEl" min="6" max="60" step="1.5" value="11.5" oninput="updateSliderLabels()" onchange="requestLiveRender()" />
          </div>
          <div class="slider-row">
            <label><span>视角距离 (Radius)</span><span id="valRad">13.8 m</span></label>
            <input type="range" id="slRad" min="6.0" max="22.0" step="0.4" value="13.8" oninput="updateSliderLabels()" onchange="requestLiveRender()" />
          </div>
          <div class="slider-row">
            <label><span>行车时间点 (Timestamp)</span><span id="valTime">2.0 s</span></label>
            <input type="range" id="slTime" min="0.2" max="19.0" step="0.3" value="2.0" oninput="updateSliderLabels()" onchange="requestLiveRender()" />
          </div>
          <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:14px 0; font-size:13px">
            <label style="cursor:pointer; color:#d8b4fe; font-weight:600"><input type="checkbox" id="chkDifix" checked onchange="requestLiveRender()" /> ✨ Difix3D+</label>
            <label style="cursor:pointer; color:#7dd3fc; font-weight:600"><input type="checkbox" id="chkTraj" checked onchange="requestLiveRender()" /> 🧭 电光蓝带</label>
            <label style="cursor:pointer"><input type="checkbox" id="chkEgo" checked onchange="requestLiveRender()" /> 🚗 PBR 自车</label>
            <select id="selRes" style="background:#182136;color:#7dd3fc;border:1px solid #0369a1;border-radius:6px;padding:4px 8px;font-weight:600" onchange="requestLiveRender()">
              <option value="720p" selected>720p HD (1280×720)</option>
              <option value="1080p">1080p Full HD (1920×1080)</option>
            </select>
            <select id="selMode" style="background:#182136;color:#fff;border:1px solid #304166;border-radius:6px;padding:4px 8px" onchange="requestLiveRender()">
              <option value="gt">NuRec 3DGS 真值</option>
              <option value="coarse_cache">6 路环视 · 多帧 3D Cache</option>
              <option value="coarse">6 路环视 · 单帧投影</option>
            </select>
          </div>
          <button class="primary" style="width:100%; padding:10px" onclick="requestLiveRender()">⚡ 立即实时渲染</button>
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 3: FRONT COMPARE -->
  <div id="sec-compare_front" class="section">
    <div class="card">
      <h2>
        <span>原始车端前视相机 JPEG（左） vs 自研纯 PyTorch 3DGS 渲染（右）</span>
        <span class="sub">验证天空盒、车道线与动态车辆 3D 几何对齐</span>
      </h2>
      <div class="controls">
        <span style="font-size:13px">卷帘位置：</span>
        <input type="range" id="splitFront" min="0" max="100" value="50" style="flex:1" oninput="setSplit('front', this.value)" />
      </div>
      <div class="compare-wrap" id="wrap-front">
        <img id="frontRight" src="/assets/front_render_ref.jpg" alt="rendered front" />
        <div class="compare-top" id="top-front">
          <img id="frontLeft" src="/assets/ref_front_wide.jpg" alt="original front" />
        </div>
        <div class="imglabel" style="top:10px; left:10px">原始前视真图 (ftheta 120°)</div>
        <div class="imglabel" style="top:10px; left:auto; right:10px">自研 3DGS 渲染 (720p pinhole 去畸变)</div>
      </div>
    </div>
  </div>

  <!-- SECTION 4: BENCHMARKS -->
  <div id="sec-benchmarks" class="section">
    <div class="card">
      <h2><span>v1 真权重本地实测指标（NVIDIA GeForce RTX 3080 20GB）</span></h2>
      <div class="metrics-grid">
        <div class="metric"><div class="k">Alpamayo2-Edge 视觉塔</div><div class="v">0.78 GB</div><div class="d">437 SigLIP2 权重 strict=True</div></div>
        <div class="metric"><div class="k">v1 全栈常驻显存</div><div class="v">5.21 GB</div><div class="d">Alpamayo2 + Wan2.1-VACE + VAE + Adapter</div></div>
        <div class="metric"><div class="k">原生 720p Stage 0 渲染</div><div class="v">1280×720</div><div class="d">零回缩直出，4:4:4 JPEG，自车后合成零糊</div></div>
        <div class="metric"><div class="k">213K面 PBR 自车 (4x MSAA)</div><div class="v">&lt; 2 ms</div><div class="d">RTX 3080 硬件 EGL + NuRec 天空盒反射</div></div>
        <div class="metric"><div class="k">Stage 1A 训练 (480×832, 17f)</div><div class="v">9.04 GB</div><div class="d">91.5M 参数，含 15% Branch A Dropout</div></div>
        <div class="metric"><div class="k">Stage 1B LoRA (256×448, 9f)</div><div class="v">12.23 GB</div><div class="d">897.3M 参数 (LoRA r=64 + 15层VACE)</div></div>
      </div>
    </div>
  </div>
</div>

<script>
let meta = null;
let curFrame = 3;
let playing = false;
let playTimer = null;
let liveInFlight = false;
let useTimelineDifix = true;
let useTimelineCache = true;
let difixCompareMode = 'chase';

async function init() {
  const res = await fetch('/api/meta');
  meta = await res.json();
  const avgSingle = (meta.hole_ratios_single.reduce((a, b) => a + b, 0) / meta.hole_ratios_single.length).toFixed(1);
  const avgCache = (meta.hole_ratios_cache.reduce((a, b) => a + b, 0) / meta.hole_ratios_cache.length).toFixed(1);
  document.getElementById('sceneMeta').textContent =
    `场景: ${meta.sequence_id} | 原生输出: ${meta.tgt_hw[1]}×${meta.tgt_hw[0]} HD | PBR 自车: ${meta.using_glb ? '213K面 4x MSAA 后合成(零糊)' : '程序化'} | 盲区均值: 单帧 ${avgSingle}% → 3D Cache ${avgCache}%`;
  document.getElementById('frameSlider').max = meta.num_frames - 1;
  document.getElementById('difixFrameSlider').max = meta.num_frames - 1;
  setFrame(Math.min(3, meta.num_frames - 1));
  setDifixMode('chase');
  updateSliderLabels();
  syncCompareWidths();
  window.addEventListener('resize', syncCompareWidths);
}

function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('sec-' + name).classList.add('active');
  syncCompareWidths();
}

function pad4(n) { return String(n).padStart(4, '0'); }

function setDifixMode(mode) {
  difixCompareMode = mode;
  document.getElementById('btnDiffOblique').classList.toggle('primary', mode === 'oblique');
  document.getElementById('btnDiffChase').classList.toggle('primary', mode === 'chase');
  document.getElementById('btnDiffCache').classList.toggle('primary', mode === 'cache');
  document.getElementById('difixFrameWrap').style.display = mode === 'oblique' ? 'none' : 'inline';

  if (mode === 'oblique') {
    document.getElementById('difixLeft').src = '/assets/oblique_raw.jpg';
    document.getElementById('difixRight').src = '/assets/oblique_difix.jpg';
    document.getElementById('lblCurtainLeft').textContent = '裸 NuRec 3DGS + PBR 自车（720p 修复前）';
    document.getElementById('lblCurtainRight').textContent = '✨ Difix3D+ 720p 街景 + 无损 PBR 自车 + 电光蓝带';
    document.getElementById('difixFrameLabel').textContent =
      '30° 侧后斜视特写（1280×720 原生 HD）— 拖动卷帘对比周边车体修复与无损 4×MSAA 车漆锐度';
    syncCompareWidths();
  } else {
    setDifixCompareFrame(parseInt(document.getElementById('difixFrameSlider').value));
  }
}

function setDifixCompareFrame(idx) {
  const k = pad4(idx);
  if (difixCompareMode === 'cache') {
    document.getElementById('difixLeft').src = `/assets/coarse_single_${k}.jpg`;
    document.getElementById('difixRight').src = `/assets/coarse_cache_${k}.jpg`;
    const hs = meta ? meta.hole_ratios_single[idx].toFixed(1) : '?';
    const hc = meta ? meta.hole_ratios_cache[idx].toFixed(1) : '?';
    document.getElementById('lblCurtainLeft').textContent = `单帧 6 路环视反投影（盲区 ${hs}%）`;
    document.getElementById('lblCurtainRight').textContent = `🔥 多帧因果 3D Cache 融合（盲区 ${hc}%）`;
    document.getElementById('difixFrameLabel').textContent =
      `Frame ${idx + 1}/${meta ? meta.num_frames : 10} — 对比单帧大黑洞 vs 多帧 3D Cache 融合`;
  } else {
    document.getElementById('difixLeft').src = `/assets/chase_raw_${k}.jpg`;
    document.getElementById('difixRight').src = `/assets/chase_difix_${k}.jpg`;
    document.getElementById('lblCurtainLeft').textContent = '裸 NuRec 3DGS + PBR 自车（720p 修复前）';
    document.getElementById('lblCurtainRight').textContent = '✨ Difix3D+ 720p 街景 + 无损 PBR 自车 + 柔光蓝带';
    if (meta) {
      document.getElementById('difixFrameLabel').textContent =
        `特斯拉同款平视追尾 Frame ${idx + 1}/${meta.num_frames} (t=${meta.timestamps_sec[idx].toFixed(2)}s, 1280×720 HD)`;
    }
  }
  syncCompareWidths();
}

function toggleTimelineCache() {
  useTimelineCache = !useTimelineCache;
  const btn = document.getElementById('btnCacheToggle');
  btn.textContent = useTimelineCache ? '🔥 Branch A：多帧 3D Cache' : 'Branch A：单帧反投影';
  btn.classList.toggle('cyan-toggle', useTimelineCache);
  document.getElementById('lblCoarse').textContent = useTimelineCache
    ? '② Branch A：多帧因果 3D Cache 投影'
    : '② Branch A：单帧 6 路环视反投影粗图';
  setFrame(curFrame);
}

function toggleTimelineDifix() {
  useTimelineDifix = !useTimelineDifix;
  const btn = document.getElementById('btnDifixToggle');
  btn.textContent = useTimelineDifix ? '✨ 第四格：Difix3D+ 720p' : '第四格：裸 3DGS（720p）';
  btn.classList.toggle('active-toggle', useTimelineDifix);
  document.getElementById('lblChase').textContent = useTimelineDifix
    ? '④ 目标真值：720p Difix3D+ 街景 + 无损 PBR 自车 + 电光蓝带'
    : '④ 原始渲染：720p 裸 NuRec 3DGS + PBR 自车';
  setFrame(curFrame);
}

function setFrame(idx) {
  if (!meta) return;
  curFrame = (idx + meta.num_frames) % meta.num_frames;
  document.getElementById('frameSlider').value = curFrame;
  const tSec = meta.timestamps_sec[curFrame].toFixed(2);
  const hs = meta.hole_ratios_single[curFrame].toFixed(1);
  const hc = meta.hole_ratios_cache[curFrame].toFixed(1);
  document.getElementById('frameInfo').textContent =
    `Frame ${curFrame + 1} / ${meta.num_frames}  |  t = ${tSec}s  |  盲区: 单帧 ${hs}% → 3D Cache ${hc}%`;
  const k = pad4(curFrame);
  document.getElementById('imgSurround').src = `/assets/surround_${k}.jpg`;
  document.getElementById('imgCoarse').src = useTimelineCache ? `/assets/coarse_cache_${k}.jpg` : `/assets/coarse_single_${k}.jpg`;
  document.getElementById('imgMask').src = useTimelineCache ? `/assets/mask_cache_${k}.jpg` : `/assets/mask_single_${k}.jpg`;
  document.getElementById('imgChase').src = useTimelineDifix ? `/assets/chase_difix_${k}.jpg` : `/assets/chase_raw_${k}.jpg`;
}

function stepFrame(delta) { setFrame(curFrame + delta); }

function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? '⏸ 暂停' : '▶ 播放';
  if (playing) restartTimerIfPlaying();
  else if (playTimer) { clearInterval(playTimer); playTimer = null; }
}

function restartTimerIfPlaying() {
  if (playTimer) clearInterval(playTimer);
  if (!playing) return;
  const fps = parseFloat(document.getElementById('fpsSelect').value);
  playTimer = setInterval(() => stepFrame(1), 1000 / fps);
}

function updateSliderLabels() {
  const az = document.getElementById('slAz').value;
  const el = document.getElementById('slEl').value;
  const rad = document.getElementById('slRad').value;
  const t = document.getElementById('slTime').value;
  document.getElementById('valAz').textContent = `${az}°`;
  document.getElementById('valEl').textContent = `${el}°`;
  document.getElementById('valRad').textContent = `${rad} m`;
  document.getElementById('valTime').textContent = `${t} s`;
  document.getElementById('orbitLabel').textContent =
    `Azimuth ${az}° | Elevation ${el}° | Radius ${rad}m | t=${t}s`;
}

function applyPreset(az, el, rad) {
  document.getElementById('slAz').value = az;
  document.getElementById('slEl').value = el;
  document.getElementById('slRad').value = rad;
  updateSliderLabels();
  requestLiveRender();
}

async function requestLiveRender() {
  if (liveInFlight) return;
  liveInFlight = true;
  const az = document.getElementById('slAz').value;
  const el = document.getElementById('slEl').value;
  const rad = document.getElementById('slRad').value;
  const t = document.getElementById('slTime').value;
  const ego = document.getElementById('chkEgo').checked ? 1 : 0;
  const difix = document.getElementById('chkDifix').checked ? 1 : 0;
  const traj = document.getElementById('chkTraj').checked ? 1 : 0;
  const res = document.getElementById('selRes').value;
  const mode = document.getElementById('selMode').value;
  const status = document.getElementById('orbitStatus');
  status.textContent = `⏳ RTX 3080 原生 ${res} 渲染${difix ? ' + Difix3D+ 扩散修复' : ''}中…`;
  try {
    const url = `/api/render?az=${az}&el=${el}&radius=${rad}&t=${t}&ego=${ego}&difix=${difix}&traj=${traj}&res=${res}&mode=${mode}&_=${Date.now()}`;
    const t0 = performance.now();
    const resp = await fetch(url);
    const ms = resp.headers.get('X-Render-Ms') || Math.round(performance.now() - t0);
    const blob = await resp.blob();
    const objUrl = URL.createObjectURL(blob);
    const img = document.getElementById('imgOrbit');
    if (img.dataset.prevUrl) URL.revokeObjectURL(img.dataset.prevUrl);
    img.src = objUrl;
    img.dataset.prevUrl = objUrl;
    status.textContent = `✅ ${res} 渲染完成 (${ms} ms${difix ? ', 含 Difix3D+' : ''}${traj ? ', 含电光蓝带' : ''})`;
  } catch (err) {
    status.textContent = '❌ 渲染出错: ' + err;
  } finally {
    liveInFlight = false;
  }
}

(function setupDragOrbit() {
  let dragging = false, sx = 0, sy = 0, startAz = 0, startEl = 0;
  const box = document.getElementById('orbitCanvas');
  box.addEventListener('pointerdown', e => {
    dragging = true; sx = e.clientX; sy = e.clientY;
    startAz = parseFloat(document.getElementById('slAz').value);
    startEl = parseFloat(document.getElementById('slEl').value);
    box.style.cursor = 'grabbing';
  });
  window.addEventListener('pointermove', e => {
    if (!dragging) return;
    let naz = Math.round((startAz - (e.clientX - sx) * 0.45) / 5) * 5;
    if (naz > 180) naz -= 360;
    if (naz < -180) naz += 360;
    const nel = Math.max(6, Math.min(60, Math.round((startEl + (e.clientY - sy) * 0.25) / 1.5) * 1.5));
    document.getElementById('slAz').value = naz;
    document.getElementById('slEl').value = nel;
    updateSliderLabels();
  });
  window.addEventListener('pointerup', () => {
    if (!dragging) return;
    dragging = false;
    box.style.cursor = 'grab';
    requestLiveRender();
  });
})();

function setInurecMode(mode) {
  document.getElementById('btnInurecVsGt').classList.toggle('primary', mode === 'vs_gt');
  document.getElementById('btnInurecVsDa3').classList.toggle('primary', mode === 'vs_da3');
  if (mode === 'vs_da3') {
    document.getElementById('inurecLeft').src = '/assets/coarse_cache_0008.jpg';
    document.getElementById('inurecRight').src = '/assets/inurec_raw_720p.jpg';
    document.getElementById('lblInurecLeft').textContent = '① 旧 Branch A：DA3 多帧点云缓存（大量黑洞锯齿）';
    document.getElementById('lblInurecRight').textContent = '② Instant-NuRec 前馈 3DGS（0.68GB · 覆盖率 96.1%）';
  } else {
    document.getElementById('inurecLeft').src = '/assets/inurec_difix_720p.jpg';
    document.getElementById('inurecRight').src = '/assets/chase_difix_0008.jpg';
    document.getElementById('lblInurecLeft').textContent = '③ Instant-NuRec + Difix3D+（零离线地图 / 零训练）';
    document.getElementById('lblInurecRight').textContent = '④ 离线 NuRec 26.10 GT + Difix3D+（70 分钟离线精建）';
  }
  syncCompareWidths();
}

function syncCompareWidths() {
  for (const key of ['difix', 'front', 'inurec']) {
    const wrap = document.getElementById('wrap-' + key);
    const leftImg = document.getElementById(key + 'Left');
    if (wrap && leftImg && wrap.clientWidth > 0) {
      leftImg.style.width = wrap.clientWidth + 'px';
    }
  }
}
function setSplit(key, pct) {
  document.getElementById('top-' + key).style.width = pct + '%';
  syncCompareWidths();
}

init();
</script>
</body>
</html>
"""


def build_or_load_cache(
    renderer: NuRecSceneRenderer,
    refiner: Difix3DRefiner | None,
    usdz_path: Path,
    cache_dir: Path,
    *,
    num_frames: int = 10,
    dt_sec: float = 0.40,
    start_sec: float = 0.85,
    src_hw: tuple[int, int] = (288, 512),
    tgt_hw: tuple[int, int] = (720, 1280),
) -> dict[str, Any]:
    """Pre-render native 720p HD timeline frames with Tesla framing & post-Difix 4x MSAA PBR ego car."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    cache_version = "v5_hd720_tesla_framing"
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            existing.get("cache_version") == cache_version
            and existing.get("num_frames") == num_frames
            and existing.get("sequence_id") == renderer.sequence_id
            and existing.get("tgt_hw") == list(tgt_hw)
        ):
            print(f"Reusing v5 HD 720p visualizer cache in {cache_dir}")
            return existing

    single_projector = MultiViewDepthProjector(splat_radius_px=1).to(renderer.device)
    cache_projector = Temporal3DCacheProjector(
        splat_radius_px=1,
        ego_exclusion_lwh=(
            renderer.ego_dim_lwh[0] + 0.15,
            renderer.ego_dim_lwh[1] + 0.15,
            renderer.ego_dim_lwh[2] + 0.30,
        ),
        ego_centroid_xy=(renderer.ego_centroid[0], renderer.ego_centroid[1]),
    ).to(renderer.device)

    cam_names = NuRecSceneRenderer.DEFAULT_SURROUND_CAMERAS
    timestamps_sec: list[float] = []
    hole_ratios_single: list[float] = []
    hole_ratios_cache: list[float] = []

    ref_ts_us = 0.055 * 1e6
    with zipfile.ZipFile(usdz_path) as zf:
        for name in zf.namelist():
            if name.startswith("frames/camera_front_wide_120fov/"):
                (cache_dir / "ref_front_wide.jpg").write_bytes(zf.read(name))
                stem = Path(name).stem
                if stem.isdigit():
                    ref_ts_us = float(stem)
                break

    front_c2e = torch.from_numpy(renderer.cameras["camera_front_wide_120fov"].T_sensor_rig).to(renderer.device)
    front_hd_k = renderer.camera_intrinsics("camera_front_wide_120fov", tgt_hw[0], tgt_hw[1])
    front_ref = renderer.render_camera(ref_ts_us, front_c2e, front_hd_k, tgt_hw[0], tgt_hw[1])
    _save_hd_jpeg(_to_pil(front_ref.rgb), cache_dir / "front_render_ref.jpg")

    # Tesla FSD v15 drone follow-cam: shows ~28% sky/horizon + complete uncropped ego sedan & shadow
    chase_pose = build_orbit_chase_pose(
        radius=TESLA_CHASE_RADIUS,
        elevation_deg=TESLA_CHASE_ELEV_DEG,
        azimuth_deg=0.0,
        look_at_xyz=(renderer.ego_centroid[0] + TESLA_LOOK_AHEAD_X, 0.0, TESLA_LOOK_AT_Z),
        device=renderer.device,
    )
    k_tgt = _pinhole_k(tgt_hw, TESLA_CHASE_FOV_X_DEG, renderer.device)

    hist_rgb: list[torch.Tensor] = []
    hist_dep: list[torch.Tensor] = []
    hist_k: list[torch.Tensor] = []
    hist_c2e: list[torch.Tensor] = []
    hist_r2w: list[torch.Tensor] = []
    max_win = 6

    print("Seeding 2 warm-up surround frames for causal 3D cache ...")
    for warm_idx in (2, 1):
        w_ts = max(0.055, start_sec - warm_idx * dt_sec) * 1e6
        wg = renderer._assemble_gaussians_at(w_ts)
        s_rgbs, s_deps, s_ks, s_c2es = [], [], [], []
        for cam in cam_names:
            ck = renderer.camera_intrinsics(cam, src_hw[0], src_hw[1])
            cc = torch.from_numpy(renderer.cameras[cam].T_sensor_rig).to(renderer.device)
            res = renderer.render_camera(w_ts, cc, ck, src_hw[0], src_hw[1], _cached_gaussians=wg)
            s_rgbs.append(res.rgb.permute(2, 0, 1))
            s_deps.append(res.depth.unsqueeze(0))
            s_ks.append(ck)
            s_c2es.append(cc)
        hist_rgb.append(torch.stack(s_rgbs, dim=0))
        hist_dep.append(torch.stack(s_deps, dim=0))
        hist_k.append(torch.stack(s_ks, dim=0))
        hist_c2e.append(torch.stack(s_c2es, dim=0))
        hist_r2w.append(renderer.rig_to_nurec(w_ts))

    print(f"Pre-rendering {num_frames} native {tgt_hw[1]}x{tgt_hw[0]} HD frames (Difix3D+ 720p -> Post-VAE 4x MSAA PBR car) ...")
    t0 = time.perf_counter()
    for i in range(num_frames):
        t_sec = start_sec + i * dt_sec
        timestamps_sec.append(t_sec)
        ts_us = t_sec * 1e6
        gaussians = renderer._assemble_gaussians_at(ts_us)
        r2w_cur = renderer.rig_to_nurec(ts_us)

        # Render 6 surround views
        surr_rgb_dict: dict[str, torch.Tensor] = {}
        s_rgbs, s_deps, s_ks, s_c2es = [], [], [], []
        for cam in cam_names:
            ck = renderer.camera_intrinsics(cam, src_hw[0], src_hw[1])
            cc = torch.from_numpy(renderer.cameras[cam].T_sensor_rig).to(renderer.device)
            res = renderer.render_camera(ts_us, cc, ck, src_hw[0], src_hw[1], _cached_gaussians=gaussians)
            surr_rgb_dict[cam] = res.rgb
            s_rgbs.append(res.rgb.permute(2, 0, 1))
            s_deps.append(res.depth.unsqueeze(0))
            s_ks.append(ck)
            s_c2es.append(cc)

        _save_hd_jpeg(_make_surround_mosaic(surr_rgb_dict, cam_names), cache_dir / f"surround_{i:04d}.jpg", quality=92)

        cur_rgb = torch.stack(s_rgbs, dim=0)
        cur_dep = torch.stack(s_deps, dim=0)
        cur_k = torch.stack(s_ks, dim=0)
        cur_c2e = torch.stack(s_c2es, dim=0)
        hist_rgb.append(cur_rgb)
        hist_dep.append(cur_dep)
        hist_k.append(cur_k)
        hist_c2e.append(cur_c2e)
        hist_r2w.append(r2w_cur)
        if len(hist_rgb) > max_win:
            hist_rgb.pop(0)
            hist_dep.pop(0)
            hist_k.pop(0)
            hist_c2e.pop(0)
            hist_r2w.pop(0)

        # Single-frame vs Multi-frame 3D Cache projection at native HD
        p_single = single_projector(
            cur_rgb.unsqueeze(0),
            cur_dep.unsqueeze(0),
            cur_k.unsqueeze(0),
            cur_c2e.unsqueeze(0),
            k_tgt.unsqueeze(0),
            chase_pose.unsqueeze(0),
            target_hw=tgt_hw,
        )
        p_single = renderer.ego_compositor(
            p_single, k_tgt.unsqueeze(0), chase_pose.unsqueeze(0), skybox_textures=renderer.skybox, T_rig_world=r2w_cur
        )
        hs_pct = float(p_single.hole_mask.mean().item() * 100.0)
        hole_ratios_single.append(hs_pct)
        _save_hd_jpeg(_to_pil(p_single.coarse_rgb[0].permute(1, 2, 0)), cache_dir / f"coarse_single_{i:04d}.jpg", quality=92)
        _save_hd_jpeg(_make_mask_overlay(p_single.coarse_rgb[0], p_single.hole_mask[0]), cache_dir / f"mask_single_{i:04d}.jpg", quality=92)

        p_cache = cache_projector.project_temporal_window(
            torch.stack(hist_rgb, dim=0).unsqueeze(0),
            torch.stack(hist_dep, dim=0).unsqueeze(0),
            torch.stack(hist_k, dim=0).unsqueeze(0),
            torch.stack(hist_c2e, dim=0).unsqueeze(0),
            torch.stack(hist_r2w, dim=0).unsqueeze(0),
            k_tgt.unsqueeze(0),
            chase_pose.unsqueeze(0),
            target_hw=tgt_hw,
        )
        p_cache = renderer.ego_compositor(
            p_cache, k_tgt.unsqueeze(0), chase_pose.unsqueeze(0), skybox_textures=renderer.skybox, T_rig_world=r2w_cur
        )
        hc_pct = float(p_cache.hole_mask.mean().item() * 100.0)
        hole_ratios_cache.append(hc_pct)
        _save_hd_jpeg(_to_pil(p_cache.coarse_rgb[0].permute(1, 2, 0)), cache_dir / f"coarse_cache_{i:04d}.jpg", quality=92)
        _save_hd_jpeg(_make_mask_overlay(p_cache.coarse_rgb[0], p_cache.hole_mask[0]), cache_dir / f"mask_cache_{i:04d}.jpg", quality=92)

        # Render pure 720p NuRec background (no ego car yet!)
        bg_out = renderer.render_camera(
            ts_us, chase_pose, k_tgt, tgt_hw[0], tgt_hw[1], composite_ego=False, _cached_gaussians=gaussians
        )
        raw_comp = renderer.composite_ego_and_trajectory(
            bg_out.rgb, bg_out.depth, bg_out.alpha, ts_us, chase_pose, k_tgt,
            composite_ego=True, draw_trajectory=False, display_clarity=False,
        )
        _save_hd_jpeg(_to_pil(raw_comp.rgb), cache_dir / f"chase_raw_{i:04d}.jpg")

        # Refine pure 720p background via native 720p Difix3D+, THEN composite razor-sharp 4x MSAA PBR car + blue ribbon!
        if refiner is not None:
            ref_front_hd = renderer.render_camera(
                ts_us, front_c2e, front_hd_k, tgt_hw[0], tgt_hw[1], _cached_gaussians=gaussians
            ).rgb
            bg_refined = refiner.refine(bg_out.rgb, ref_front_hd, process_hw=None)
        else:
            bg_refined = bg_out.rgb

        final_comp = renderer.composite_ego_and_trajectory(
            bg_refined, bg_out.depth, bg_out.alpha, ts_us, chase_pose, k_tgt,
            composite_ego=True, draw_trajectory=True, display_clarity=True,
        )
        _save_hd_jpeg(_to_pil(final_comp.rgb), cache_dir / f"chase_difix_{i:04d}.jpg")

        print(
            f"  [{i + 1:2d}/{num_frames}] t={t_sec:.2f}s | blind_spot: single={hs_pct:.1f}% -> cache={hc_pct:.1f}% "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    # 30-deg oblique showcase at native 720p HD
    ob_t_sec = start_sec + min(3, num_frames - 1) * dt_sec
    ob_ts_us = ob_t_sec * 1e6
    ob_pose = build_orbit_chase_pose(
        radius=13.2,
        elevation_deg=15.0,
        azimuth_deg=28.0,
        look_at_xyz=(renderer.ego_centroid[0] + TESLA_LOOK_AHEAD_X * 0.85, 0.0, TESLA_LOOK_AT_Z),
        device=renderer.device,
    )
    ob_g = renderer._assemble_gaussians_at(ob_ts_us)
    ob_bg = renderer.render_camera(
        ob_ts_us, ob_pose, k_tgt, tgt_hw[0], tgt_hw[1], composite_ego=False, _cached_gaussians=ob_g
    )
    ob_raw = renderer.composite_ego_and_trajectory(
        ob_bg.rgb, ob_bg.depth, ob_bg.alpha, ob_ts_us, ob_pose, k_tgt,
        composite_ego=True, draw_trajectory=False, display_clarity=False,
    )
    _save_hd_jpeg(_to_pil(ob_raw.rgb), cache_dir / "oblique_raw.jpg")

    if refiner is not None:
        ob_ref = renderer.render_camera(
            ob_ts_us, front_c2e, front_hd_k, tgt_hw[0], tgt_hw[1], _cached_gaussians=ob_g
        ).rgb
        ob_bg_ref = refiner.refine(ob_bg.rgb, ob_ref, process_hw=None)
    else:
        ob_bg_ref = ob_bg.rgb
    ob_final = renderer.composite_ego_and_trajectory(
        ob_bg_ref, ob_bg.depth, ob_bg.alpha, ob_ts_us, ob_pose, k_tgt,
        composite_ego=True, draw_trajectory=True, display_clarity=True,
    )
    _save_hd_jpeg(_to_pil(ob_final.rgb), cache_dir / "oblique_difix.jpg")

    meta = {
        "cache_version": cache_version,
        "sequence_id": renderer.sequence_id,
        "has_difix": refiner is not None,
        "using_glb": renderer.ego_compositor.using_glb,
        "tgt_hw": list(tgt_hw),
        "num_frames": num_frames,
        "timestamps_sec": timestamps_sec,
        "hole_ratios_single": hole_ratios_single,
        "hole_ratios_cache": hole_ratios_cache,
        "ego_dim_lwh": list(renderer.ego_dim_lwh),
        "ego_centroid": list(renderer.ego_centroid),
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Start intranet Reality View HD visualizer v5.")
    default_usdz = DEFAULT_CA_USDZ if Path(DEFAULT_CA_USDZ).is_file() else ALT_HIGHWAY_USDZ
    parser.add_argument("--usdz", default=default_usdz)
    parser.add_argument("--cache-dir", default="outputs/viz_cache_v5_hd")
    parser.add_argument("--ego-glb", default="auto")
    parser.add_argument("--paint", default="pearl_white", choices=["pearl_white", "midnight_silver", "ultra_red", "stealth_grey"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--no-difix", action="store_true")
    args = parser.parse_args()

    usdz_path = Path(args.usdz)
    cache_dir = Path(args.cache_dir)

    print(f"Loading NuRec scene {usdz_path} onto RTX 3080 ...")
    renderer = NuRecSceneRenderer(
        usdz_path,
        device="cuda",
        ego_glb_path=args.ego_glb,
        ego_paint_theme=args.paint,
    )
    print(f"Ego vehicle renderer: {'213k PBR GLB (4x MSAA EGL)' if renderer.ego_compositor.using_glb else 'Procedural fallback'}")

    refiner: Difix3DRefiner | None = None
    if not args.no_difix:
        print("Loading NVIDIA Difix3D+ (nvidia/difix_ref, FP16) ...")
        refiner = Difix3DRefiner.from_pretrained("nvidia/difix_ref", device="cuda", dtype=torch.float16)
        print("Loaded Difix3D+ successfully!")

    single_projector = MultiViewDepthProjector(splat_radius_px=1).to(renderer.device)
    cache_projector = Temporal3DCacheProjector(
        splat_radius_px=1,
        ego_exclusion_lwh=(
            renderer.ego_dim_lwh[0] + 0.15,
            renderer.ego_dim_lwh[1] + 0.15,
            renderer.ego_dim_lwh[2] + 0.30,
        ),
        ego_centroid_xy=(renderer.ego_centroid[0], renderer.ego_centroid[1]),
    ).to(renderer.device)
    start_x_ribbon = renderer.ego_centroid[0] + renderer.ego_dim_lwh[0] * 0.5 + 0.35

    meta = build_or_load_cache(renderer, refiner, usdz_path, cache_dir, num_frames=args.num_frames)
    gpu_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *log_args: Any) -> None:
            return

        def _send_bytes(
            self,
            payload: bytes,
            content_type: str,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for hk, hv in extra_headers.items():
                    self.send_header(hk, hv)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                self._send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/meta":
                self._send_bytes(json.dumps(meta).encode("utf-8"), "application/json")
                return

            if path.startswith("/assets/"):
                fname = Path(path).name
                fpath = cache_dir / fname
                if fpath.is_file():
                    ctype = "image/jpeg" if fname.endswith((".jpg", ".jpeg")) else "image/png"
                    self._send_bytes(fpath.read_bytes(), ctype)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                return

            if path == "/api/render":
                qs = urllib.parse.parse_qs(parsed.query)
                az = float(qs.get("az", ["0"])[0])
                el = float(qs.get("el", [str(TESLA_CHASE_ELEV_DEG)])[0])
                radius = float(qs.get("radius", [str(TESLA_CHASE_RADIUS)])[0])
                t_sec = max(0.05, min(19.5, float(qs.get("t", ["2.0"])[0])))
                show_ego = qs.get("ego", ["1"])[0] != "0"
                use_difix = qs.get("difix", ["1"])[0] != "0"
                show_traj = qs.get("traj", ["1"])[0] != "0"
                res_key = qs.get("res", ["720p"])[0]
                mode = qs.get("mode", ["gt"])[0]

                tgt_hw = (1080, 1920) if res_key == "1080p" else (720, 1280)
                difix_hw = (720, 1280)  # Keep Difix3D+ crisp & fast within 20GB VRAM even when compositing 1080p!
                ts_us = t_sec * 1e6
                t0 = time.perf_counter()
                with gpu_lock:
                    chase_pose = build_orbit_chase_pose(
                        radius=radius,
                        elevation_deg=el,
                        azimuth_deg=az,
                        look_at_xyz=(renderer.ego_centroid[0] + TESLA_LOOK_AHEAD_X, 0.0, TESLA_LOOK_AT_Z),
                        device=renderer.device,
                    )
                    k_tgt = _pinhole_k(tgt_hw, TESLA_CHASE_FOV_X_DEG, renderer.device)

                    if mode in ("coarse", "coarse_cache"):
                        cam_names = NuRecSceneRenderer.DEFAULT_SURROUND_CAMERAS
                        n_hist = 4 if mode == "coarse_cache" else 1
                        w_imgs, w_deps, w_ks, w_c2es, w_r2ws = [], [], [], [], []
                        for h_i in range(n_hist):
                            ht_us = max(0.05, t_sec - (n_hist - 1 - h_i) * 0.30) * 1e6
                            hg = renderer._assemble_gaussians_at(ht_us)
                            sr, sd, sk, sc = [], [], [], []
                            for cam in cam_names:
                                ck = renderer.camera_intrinsics(cam, 288, 512)
                                cc = torch.from_numpy(renderer.cameras[cam].T_sensor_rig).to(renderer.device)
                                r_out = renderer.render_camera(ht_us, cc, ck, 288, 512, _cached_gaussians=hg)
                                sr.append(r_out.rgb.permute(2, 0, 1))
                                sd.append(r_out.depth.unsqueeze(0))
                                sk.append(ck)
                                sc.append(cc)
                            w_imgs.append(torch.stack(sr, dim=0))
                            w_deps.append(torch.stack(sd, dim=0))
                            w_ks.append(torch.stack(sk, dim=0))
                            w_c2es.append(torch.stack(sc, dim=0))
                            w_r2ws.append(renderer.rig_to_nurec(ht_us))

                        if mode == "coarse_cache" and n_hist > 1:
                            proj = cache_projector.project_temporal_window(
                                torch.stack(w_imgs, dim=0).unsqueeze(0),
                                torch.stack(w_deps, dim=0).unsqueeze(0),
                                torch.stack(w_ks, dim=0).unsqueeze(0),
                                torch.stack(w_c2es, dim=0).unsqueeze(0),
                                torch.stack(w_r2ws, dim=0).unsqueeze(0),
                                k_tgt.unsqueeze(0),
                                chase_pose.unsqueeze(0),
                                target_hw=tgt_hw,
                            )
                        else:
                            proj = single_projector(
                                w_imgs[-1].unsqueeze(0),
                                w_deps[-1].unsqueeze(0),
                                w_ks[-1].unsqueeze(0),
                                w_c2es[-1].unsqueeze(0),
                                k_tgt.unsqueeze(0),
                                chase_pose.unsqueeze(0),
                                target_hw=tgt_hw,
                            )
                        bg_rgb = proj.coarse_rgb[0].permute(1, 2, 0)
                        bg_depth = proj.target_depth[0, 0]
                        bg_alpha = (proj.hole_mask[0, 0] < 0.5).float()
                        if use_difix and refiner is not None:
                            fk = renderer.camera_intrinsics("camera_front_wide_120fov", difix_hw[0], difix_hw[1])
                            fc = torch.from_numpy(renderer.cameras["camera_front_wide_120fov"].T_sensor_rig).to(renderer.device)
                            ref_front = renderer.render_camera(ts_us, fc, fk, difix_hw[0], difix_hw[1]).rgb
                            bg_rgb = refiner.refine(bg_rgb, ref_front, process_hw=difix_hw)
                        final_out = renderer.composite_ego_and_trajectory(
                            bg_rgb, bg_depth, bg_alpha, ts_us, chase_pose, k_tgt,
                            composite_ego=show_ego, draw_trajectory=show_traj, display_clarity=use_difix,
                        )
                        rgb_hwc = final_out.rgb
                    else:
                        gaussians = renderer._assemble_gaussians_at(ts_us)
                        bg_out = renderer.render_camera(
                            ts_us, chase_pose, k_tgt, tgt_hw[0], tgt_hw[1], composite_ego=False, _cached_gaussians=gaussians
                        )
                        bg_rgb = bg_out.rgb
                        if use_difix and refiner is not None:
                            fk = renderer.camera_intrinsics("camera_front_wide_120fov", difix_hw[0], difix_hw[1])
                            fc = torch.from_numpy(renderer.cameras["camera_front_wide_120fov"].T_sensor_rig).to(renderer.device)
                            ref_front = renderer.render_camera(
                                ts_us, fc, fk, difix_hw[0], difix_hw[1], _cached_gaussians=gaussians
                            ).rgb
                            bg_rgb = refiner.refine(bg_rgb, ref_front, process_hw=difix_hw)
                        final_out = renderer.composite_ego_and_trajectory(
                            bg_rgb, bg_out.depth, bg_out.alpha, ts_us, chase_pose, k_tgt,
                            composite_ego=show_ego, draw_trajectory=show_traj, display_clarity=use_difix,
                        )
                        rgb_hwc = final_out.rgb

                    # Keep start_x_ribbon referenced cleanly even if draw_trajectory handled inside helper
                    _ = (start_x_ribbon, composite_trajectory_ribbon)
                    img = _to_pil(rgb_hwc)
                    torch.cuda.synchronize()
                ms = round((time.perf_counter() - t0) * 1000)
                self._send_bytes(_pil_to_jpeg_bytes(img, quality=95), "image/jpeg", {"X-Render-Ms": str(ms)})
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Visualizer v5 HD READY on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
