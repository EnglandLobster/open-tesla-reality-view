#!/usr/bin/env python3
"""Intranet interactive visualizer v2: NuRec 3DGS + Smooth IBL Ego Sedan + NVIDIA Difix3D+ Diffusion.

Serves a Tesla-style dark-mode web UI on ``0.0.0.0:8765`` featuring:
  1. Synchronized timeline playback (6-cam surround | 3D projection | blind-spot mask | Difix3D+ vs raw 3DGS toggle)
  2. Live RTX 3080 interactive 360-degree orbit camera controller (with live Difix3D+ diffusion toggle)
  3. Dual split-curtain sliders: (A) Raw 3DGS vs Difix3D+ Enhanced, (B) Real Front Camera vs 3DGS Render
  4. Verified v1 hardware benchmark metrics
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
from open_reality_view.geometry import MultiViewDepthProjector, build_orbit_chase_pose
from open_reality_view.rendering import Difix3DRefiner


def _to_pil(rgb_hwc: torch.Tensor | np.ndarray) -> Image.Image:
    arr = rgb_hwc.detach().cpu().numpy() if isinstance(rgb_hwc, torch.Tensor) else rgb_hwc
    u8 = np.clip(arr * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(u8)


def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
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
    tw, th = 276, 156
    for cam in cam_names[:6]:
        im = _to_pil(surround_rgb[cam]).resize((tw, th), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(im)
        label = short_labels.get(cam, cam)
        draw.rectangle([4, 4, 4 + len(label) * 7 + 8, 22], fill=(10, 14, 22))
        draw.text((8, 7), label, fill=(225, 240, 255))
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


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Open Tesla Reality View — 可视化控制台 v2 (Difix3D+ 扩散增强)</title>
<style>
  :root {
    --bg: #080b12;
    --panel: #111624;
    --border: #232c42;
    --text: #e6ecf8;
    --muted: #8a97b5;
    --accent: #3b82f6;
    --accent2: #10b981;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: radial-gradient(circle at top, #10192e 0%, var(--bg) 55%);
    color: var(--text);
    line-height: 1.5;
    padding: 20px;
  }
  .wrap { max-width: 1500px; margin: 0 auto; }
  header {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 14px; padding: 18px 22px; background: rgba(17,22,36,0.92);
    border: 1px solid var(--border); border-radius: 14px; margin-bottom: 18px;
  }
  .title h1 { font-size: 22px; font-weight: 700; letter-spacing: 0.3px; }
  .title p { font-size: 13px; color: var(--muted); margin-top: 3px; }
  .pills { display: flex; flex-wrap: wrap; gap: 8px; }
  .pill {
    font-size: 12px; padding: 5px 10px; border-radius: 999px;
    background: #182136; border: 1px solid #2b3856; color: #c7d6f5;
  }
  .pill.ok { border-color: #14532d; background: #0b2919; color: #6ee7b7; }
  .pill.purple { border-color: #5b21b6; background: #1e1238; color: #c4b5fd; }
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
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
    padding: 12px 14px; background: #0c101b; border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 14px;
  }
  button {
    background: #1d2942; color: var(--text); border: 1px solid #304166;
    padding: 7px 14px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600;
  }
  button:hover { background: #263657; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.active-toggle { background: #059669; border-color: #10b981; color: #fff; }
  input[type=range] { accent-color: var(--accent); cursor: pointer; }
  .grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
  @media (max-width: 980px) { .grid2 { grid-template-columns: 1fr; } }
  .imgbox {
    background: #070a10; border: 1px solid var(--border); border-radius: 10px;
    overflow: hidden; position: relative;
  }
  .imgbox img { width: 100%; display: block; object-fit: contain; }
  .imglabel {
    position: absolute; top: 8px; left: 8px; font-size: 12px; font-weight: 600;
    padding: 4px 10px; border-radius: 6px; background: rgba(8,11,18,0.84);
    border: 1px solid rgba(255,255,255,0.14);
  }
  .badge-red { color: #fca5a5; }
  .badge-green { color: #6ee7b7; }
  .badge-purple { color: #d8b4fe; }
  .orbit-layout { display: grid; grid-template-columns: 1.55fr 1fr; gap: 16px; }
  @media (max-width: 1040px) { .orbit-layout { grid-template-columns: 1fr; } }
  .slider-row { margin-bottom: 12px; }
  .slider-row label { display: flex; justify-content: space-between; font-size: 13px; color: #c7d6f5; margin-bottom: 4px; }
  .slider-row input[type=range] { width: 100%; }
  .presets { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .compare-wrap {
    position: relative; width: 100%; max-width: 1040px; margin: 0 auto;
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
      <h1>Open Tesla Reality View — 内网效果可视化控制台 v2</h1>
      <p id="sceneMeta">加载场景信息中…</p>
    </div>
    <div class="pills">
      <span class="pill ok">RTX 3080 20GB Live</span>
      <span class="pill purple">✨ Difix3D+ 单步扩散去鬼影</span>
      <span class="pill">平滑曲面 IBL 天空盒反射自车</span>
      <span class="pill">Alpamayo2-Edge + Wan2.1-VACE-1.3B</span>
    </div>
  </header>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('compare_difix', this)">✨ 1. 核心对比：裸 3DGS vs Difix3D+ 扩散修复</div>
    <div class="tab" onclick="switchTab('timeline', this)">2. 同步时间轴回放（4面板）</div>
    <div class="tab" onclick="switchTab('orbit', this)">3. 实时 360° 自由视角（含实时 Difix3D+）</div>
    <div class="tab" onclick="switchTab('compare_front', this)">4. 前视原图 vs 自研3DGS</div>
    <div class="tab" onclick="switchTab('benchmarks', this)">5. v1 显存与实测指标</div>
  </div>

  <!-- SECTION 0 (DEFAULT): DIFIX3D+ BEFORE/AFTER CURTAIN -->
  <div id="sec-compare_difix" class="section active">
    <div class="card">
      <h2>
        <span>裸 NuRec 3DGS（左） vs 经 Difix3D+ 单步扩散修复 + 真实天空盒反射自车（右）</span>
        <span class="sub" id="difixFrameLabel">拖动滑块直观对比扩散去重影、车体光影融合与路面锐化效果</span>
      </h2>
      <div class="controls">
        <button id="btnDiffChase" class="primary" onclick="setDifixMode('chase')">标准追尾视图（10 帧）</button>
        <button id="btnDiffOblique" onclick="setDifixMode('oblique')">✨ 30° 侧后斜视特写（重影/轮毂修复对比）</button>
        <span style="font-size:13px; margin-left:6px">卷帘位置：</span>
        <input type="range" id="splitDifix" min="0" max="100" value="50" style="flex:1; min-width:140px" oninput="setSplit('difix', this.value)" />
        <span style="font-size:13px; margin-left:8px" id="difixFrameWrap">切换帧：
          <input type="range" id="difixFrameSlider" min="0" max="9" value="5" style="width:140px; vertical-align:middle" oninput="setDifixCompareFrame(parseInt(this.value))" />
        </span>
      </div>
      <div class="compare-wrap" id="wrap-difix">
        <img id="difixRight" src="/assets/chase_difix_0000.jpg" alt="Difix3D+ enhanced" />
        <div class="compare-top" id="top-difix">
          <img id="difixLeft" src="/assets/chase_raw_0000.jpg" alt="Raw 3DGS" />
        </div>
        <div class="imglabel" style="top:10px; left:10px">裸 NuRec 3DGS + 平滑自车（修复前）</div>
        <div class="imglabel badge-purple" style="top:10px; left:auto; right:10px">✨ Difix3D+ 单步扩散修复后（Stage 0 目标真值）</div>
      </div>
    </div>
  </div>

  <!-- SECTION 1: TIMELINE -->
  <div id="sec-timeline" class="section">
    <div class="card">
      <h2>
        <span>真实行驶序列同步回放（Stage 0 数据流水线产出）</span>
        <span class="sub" id="frameInfo">Frame 1 / 10</span>
      </h2>
      <div class="controls">
        <button class="primary" id="playBtn" onclick="togglePlay()">▶ 播放</button>
        <button onclick="stepFrame(-1)">◀ 上一帧</button>
        <button onclick="stepFrame(1)">下一帧 ▶</button>
        <input type="range" id="frameSlider" min="0" max="9" value="0" style="flex:1; min-width:160px" oninput="setFrame(parseInt(this.value))" />
        <button id="btnDifixToggle" class="active-toggle" onclick="toggleTimelineDifix()">✨ 第四格：Difix3D+ 修复后</button>
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
          <div class="imglabel badge-purple" id="lblChase">④ 目标真值：Difix3D+ 扩散修复后第三人称 GT</div>
          <img id="imgChase" src="" alt="chase gt" />
        </div>
        <div class="imgbox">
          <div class="imglabel">② Branch A：纯 6 路环视深度反投影粗图（进入 VACE 通道）</div>
          <img id="imgCoarse" src="" alt="coarse 3d projection" />
        </div>
        <div class="imgbox">
          <div class="imglabel badge-red">③ 盲区 Mask 叠层（红色 = 环视盲区，交由 Wan2.1-VACE 补全）</div>
          <img id="imgMask" src="" alt="blind spot mask" />
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 2: LIVE ORBIT -->
  <div id="sec-orbit" class="section">
    <div class="card">
      <h2>
        <span>交互式第三人称自由视角（RTX 3080 实时 3DGS + 可选 Difix3D+ 单步扩散）</span>
        <span class="sub" id="orbitStatus">就绪 — 支持按住鼠标左键直接拖拽旋转</span>
      </h2>
      <div class="orbit-layout">
        <div class="imgbox" id="orbitCanvas" style="cursor:grab; min-height:360px">
          <div class="imglabel" id="orbitLabel">Azimuth 0° | Elevation 20° | Radius 7.2m | t=1.0s</div>
          <img id="imgOrbit" src="/assets/chase_difix_0000.jpg" alt="live orbit render" draggable="false" />
          <div class="drag-hint">按住左键拖拽旋转视角</div>
        </div>
        <div style="background:#0c101b; border:1px solid var(--border); border-radius:12px; padding:16px">
          <div style="font-size:13px; color:var(--muted); margin-bottom:8px">快速预设机位</div>
          <div class="presets">
            <button onclick="applyPreset(0, 20, 7.2)">Tesla 标准追尾</button>
            <button onclick="applyPreset(-22, 22, 7.8)">左后 -22°</button>
            <button onclick="applyPreset(22, 22, 7.8)">右后 +22°</button>
            <button onclick="applyPreset(0, 48, 12.0)">高空俯视 48°</button>
            <button onclick="applyPreset(145, 16, 8.2)">右前侧看自车</button>
          </div>
          <div class="slider-row">
            <label><span>环绕方位角 (Azimuth)</span><span id="valAz">0°</span></label>
            <input type="range" id="slAz" min="-180" max="180" step="5" value="0" oninput="updateSliderLabels()" onchange="requestLiveRender()" />
          </div>
          <div class="slider-row">
            <label><span>俯仰仰角 (Elevation)</span><span id="valEl">20°</span></label>
            <input type="range" id="slEl" min="8" max="68" step="2" value="20" oninput="updateSliderLabels()" onchange="requestLiveRender()" />
          </div>
          <div class="slider-row">
            <label><span>视角距离 (Radius)</span><span id="valRad">7.2 m</span></label>
            <input type="range" id="slRad" min="4.5" max="18.0" step="0.3" value="7.2" oninput="updateSliderLabels()" onchange="requestLiveRender()" />
          </div>
          <div class="slider-row">
            <label><span>行车时间点 (Timestamp)</span><span id="valTime">1.0 s</span></label>
            <input type="range" id="slTime" min="0.1" max="19.0" step="0.3" value="1.0" oninput="updateSliderLabels()" onchange="requestLiveRender()" />
          </div>
          <div style="display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin:14px 0; font-size:13px">
            <label style="cursor:pointer; color:#d8b4fe; font-weight:600"><input type="checkbox" id="chkDifix" checked onchange="requestLiveRender()" /> ✨ 开启 Difix3D+ 扩散修复</label>
            <label style="cursor:pointer"><input type="checkbox" id="chkEgo" checked onchange="requestLiveRender()" /> 合成 3D 自车</label>
            <select id="selMode" style="background:#182136;color:#fff;border:1px solid #304166;border-radius:6px;padding:4px 8px" onchange="requestLiveRender()">
              <option value="gt">NuRec 3DGS 真值</option>
              <option value="coarse">纯 6 路环视粗投影</option>
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
        <span class="sub">验证天空盒、电线、车道线与动态车辆 3D 几何对齐</span>
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
        <div class="imglabel" style="top:10px; left:auto; right:10px">自研 3DGS 渲染 (pinhole 去畸变)</div>
      </div>
    </div>
  </div>

  <!-- SECTION 4: BENCHMARKS -->
  <div id="sec-benchmarks" class="section">
    <div class="card">
      <h2><span>v1 真权重本地实测指标（NVIDIA GeForce RTX 3080 20GB, BF16）</span></h2>
      <div class="metrics-grid">
        <div class="metric"><div class="k">Alpamayo2-Edge 视觉塔</div><div class="v">0.78 GB</div><div class="d">437 SigLIP2 权重 strict=True</div></div>
        <div class="metric"><div class="k">v1 全栈常驻显存</div><div class="v">5.21 GB</div><div class="d">Alpamayo2 + Wan2.1-VACE + VAE + Adapter</div></div>
        <div class="metric"><div class="k">Difix3D+ 单步扩散修复</div><div class="v">~150 ms</div><div class="d">576×1024 原生分辨率，去 3DGS 重影</div></div>
        <div class="metric"><div class="k">Stage 1A 训练 (480×832, 17f)</div><div class="v">9.04 GB</div><div class="d">91.5M 参数，7.96 s/step</div></div>
        <div class="metric"><div class="k">Stage 1B LoRA (256×448, 9f)</div><div class="v">12.23 GB</div><div class="d">897.3M 参数 (LoRA r=64 + 15层VACE)</div></div>
        <div class="metric"><div class="k">Flow-Matching 推理 (480×832)</div><div class="v">10.10 GB</div><div class="d">17 帧 4 步实测 9.17s</div></div>
      </div>
    </div>
  </div>
</div>

<script>
let meta = null;
let curFrame = 0;
let playing = false;
let playTimer = null;
let liveInFlight = false;
let useTimelineDifix = true;
let difixCompareMode = 'oblique';

async function init() {
  const res = await fetch('/api/meta');
  meta = await res.json();
  document.getElementById('sceneMeta').textContent =
    `场景: ${meta.sequence_id} | Difix3D+: ${meta.has_difix ? '已启用 (nvidia/difix_ref, 576×1024 高清参考)' : '未启用'} | 预渲染 ${meta.num_frames} 帧`;
  document.getElementById('frameSlider').max = meta.num_frames - 1;
  document.getElementById('difixFrameSlider').max = meta.num_frames - 1;
  setFrame(0);
  setDifixMode('oblique');
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
  document.getElementById('btnDiffChase').classList.toggle('primary', mode === 'chase');
  document.getElementById('btnDiffOblique').classList.toggle('primary', mode === 'oblique');
  document.getElementById('difixFrameWrap').style.display = mode === 'chase' ? 'inline' : 'none';
  if (mode === 'oblique') {
    document.getElementById('difixLeft').src = '/assets/oblique_raw.jpg';
    document.getElementById('difixRight').src = '/assets/oblique_difix.jpg';
    document.getElementById('difixFrameLabel').textContent =
      '30° 侧后斜视大视角偏移 (t=2.31s) — 拖动卷帘重点观察左侧白车轮毂、地面车道线与自车光影融合！';
    syncCompareWidths();
  } else {
    setDifixCompareFrame(parseInt(document.getElementById('difixFrameSlider').value));
  }
}

function setDifixCompareFrame(idx) {
  const k = pad4(idx);
  document.getElementById('difixLeft').src = `/assets/chase_raw_${k}.jpg`;
  document.getElementById('difixRight').src = `/assets/chase_difix_${k}.jpg`;
  if (meta) {
    document.getElementById('difixFrameLabel').textContent =
      `标准追尾 Frame ${idx + 1}/${meta.num_frames} (t=${meta.timestamps_sec[idx].toFixed(2)}s) — 拖动卷帘对比修复前后`;
  }
  syncCompareWidths();
}

function toggleTimelineDifix() {
  useTimelineDifix = !useTimelineDifix;
  const btn = document.getElementById('btnDifixToggle');
  btn.textContent = useTimelineDifix ? '✨ 第四格：Difix3D+ 修复后' : '第四格：裸 3DGS（修复前）';
  btn.classList.toggle('active-toggle', useTimelineDifix);
  document.getElementById('lblChase').textContent = useTimelineDifix
    ? '④ 目标真值：Difix3D+ 扩散修复后第三人称 GT'
    : '④ 原始渲染：裸 NuRec 3DGS + 平滑自车（修复前）';
  setFrame(curFrame);
}

function setFrame(idx) {
  if (!meta) return;
  curFrame = (idx + meta.num_frames) % meta.num_frames;
  document.getElementById('frameSlider').value = curFrame;
  const tSec = meta.timestamps_sec[curFrame].toFixed(2);
  const holePct = meta.hole_ratios[curFrame].toFixed(1);
  document.getElementById('frameInfo').textContent =
    `Frame ${curFrame + 1} / ${meta.num_frames}  |  t = ${tSec}s  |  环视盲区占比: ${holePct}%`;
  const k = pad4(curFrame);
  document.getElementById('imgSurround').src = `/assets/surround_${k}.jpg`;
  document.getElementById('imgCoarse').src = `/assets/coarse_${k}.jpg`;
  document.getElementById('imgMask').src = `/assets/mask_${k}.jpg`;
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
  const mode = document.getElementById('selMode').value;
  const status = document.getElementById('orbitStatus');
  status.textContent = difix ? '⏳ RTX 3080 渲染 + Difix3D+ 扩散修复中…' : '⏳ RTX 3080 实时渲染中…';
  try {
    const url = `/api/render?az=${az}&el=${el}&radius=${rad}&t=${t}&ego=${ego}&difix=${difix}&mode=${mode}&_=${Date.now()}`;
    const t0 = performance.now();
    const res = await fetch(url);
    const ms = res.headers.get('X-Render-Ms') || Math.round(performance.now() - t0);
    const blob = await res.blob();
    const objUrl = URL.createObjectURL(blob);
    const img = document.getElementById('imgOrbit');
    if (img.dataset.prevUrl) URL.revokeObjectURL(img.dataset.prevUrl);
    img.src = objUrl;
    img.dataset.prevUrl = objUrl;
    status.textContent = `✅ 渲染完成 (${ms} ms${difix ? ', 含 Difix3D+ 扩散修复' : ''})`;
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
    const nel = Math.max(8, Math.min(68, Math.round((startEl + (e.clientY - sy) * 0.25) / 2) * 2));
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

function syncCompareWidths() {
  for (const key of ['difix', 'front']) {
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
    dt_sec: float = 0.45,
    start_sec: float = 0.055,
    src_hw: tuple[int, int] = (224, 384),
    tgt_hw: tuple[int, int] = (480, 832),
) -> dict[str, Any]:
    """Pre-render v2 timeline frames including both raw 3DGS and Difix3D+ enhanced chase GT."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    cache_version = "v3_difix_hiref_fixed"
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            existing.get("cache_version") == cache_version
            and existing.get("num_frames") == num_frames
            and existing.get("sequence_id") == renderer.sequence_id
        ):
            print(f"Reusing v2 Difix3D+ visualizer cache in {cache_dir}")
            return existing

    projector = MultiViewDepthProjector(splat_radius_px=1).to(renderer.device)
    cam_names = NuRecSceneRenderer.DEFAULT_SURROUND_CAMERAS
    timestamps_sec: list[float] = []
    hole_ratios: list[float] = []

    ref_ts_us = start_sec * 1e6
    with zipfile.ZipFile(usdz_path) as zf:
        for name in zf.namelist():
            if name.startswith("frames/camera_front_wide_120fov/"):
                (cache_dir / "ref_front_wide.jpg").write_bytes(zf.read(name))
                stem = Path(name).stem
                if stem.isdigit():
                    ref_ts_us = float(stem)
                break

    front_k = renderer.camera_intrinsics("camera_front_wide_120fov", 540, 960)
    front_c2e = torch.from_numpy(renderer.cameras["camera_front_wide_120fov"].T_sensor_rig).to(renderer.device)
    front_ref = renderer.render_camera(ref_ts_us, front_c2e, front_k, 540, 960)
    _to_pil(front_ref.rgb).save(cache_dir / "front_render_ref.jpg", quality=92)

    # Slightly closer, lower Tesla-like chase camera framing for dramatic immersion
    chase_pose = build_orbit_chase_pose(
        radius=7.2,
        elevation_deg=20.0,
        azimuth_deg=0.0,
        look_at_xyz=(renderer.ego_centroid[0] + 1.4, 0.0, 0.85),
        device=renderer.device,
    )

    front_hi_k = renderer.camera_intrinsics("camera_front_wide_120fov", 576, 1024)

    print(f"Pre-rendering {num_frames} v3 frames (3DGS + smooth IBL sedan + Difix3D+ hi-res ref) ...")
    t0 = time.perf_counter()
    for i in range(num_frames):
        t_sec = start_sec + i * dt_sec
        timestamps_sec.append(t_sec)
        ts_us = t_sec * 1e6
        bundle = renderer.render_frame_bundle(
            ts_us,
            surround_cameras=cam_names,
            src_hw=src_hw,
            tgt_hw=tgt_hw,
            chase_pose_rig=chase_pose,
        )

        mosaic = _make_surround_mosaic(bundle["surround_rgb"], cam_names)
        mosaic.save(cache_dir / f"surround_{i:04d}.jpg", quality=88)

        surr_rgb = torch.stack([bundle["surround_rgb"][c].permute(2, 0, 1) for c in cam_names], dim=0).unsqueeze(0)
        surr_dep = torch.stack([bundle["surround_depth"][c].unsqueeze(0) for c in cam_names], dim=0).unsqueeze(0)
        proj = projector(
            surr_rgb,
            surr_dep,
            bundle["src_intrinsics"].unsqueeze(0),
            bundle["src_c2e"].unsqueeze(0),
            bundle["tgt_intrinsics"].unsqueeze(0),
            bundle["tgt_c2e"].unsqueeze(0),
            target_hw=tgt_hw,
        )
        proj_ego = renderer.ego_compositor(
            proj,
            bundle["tgt_intrinsics"].unsqueeze(0),
            bundle["tgt_c2e"].unsqueeze(0),
            skybox_textures=renderer.skybox,
            T_rig_world=renderer.rig_to_nurec(ts_us),
        )
        hole_pct = float(proj_ego.hole_mask.mean().item() * 100.0)
        hole_ratios.append(hole_pct)

        _to_pil(proj_ego.coarse_rgb[0].permute(1, 2, 0)).save(cache_dir / f"coarse_{i:04d}.jpg", quality=88)
        _make_mask_overlay(proj_ego.coarse_rgb[0], proj_ego.hole_mask[0]).save(cache_dir / f"mask_{i:04d}.jpg", quality=88)

        raw_chase = bundle["chase_rgb"]
        _to_pil(raw_chase).save(cache_dir / f"chase_raw_{i:04d}.jpg", quality=92)

        if refiner is not None:
            ref_front_hi = renderer.render_camera(ts_us, front_c2e, front_hi_k, 576, 1024).rgb
            refined = refiner.refine(raw_chase, ref_front_hi)
            _to_pil(refined).save(cache_dir / f"chase_difix_{i:04d}.jpg", quality=93)
        else:
            _to_pil(raw_chase).save(cache_dir / f"chase_difix_{i:04d}.jpg", quality=92)

        print(f"  [{i + 1:2d}/{num_frames}] t={t_sec:.2f}s | blind_spot={hole_pct:.1f}% ({time.perf_counter() - t0:.1f}s)")

    # Showcase 30-deg oblique novel view where Difix3D+ dramatically reconstructs blurry 3DGS side vehicles
    ob_ts_us = (start_sec + min(5, num_frames - 1) * dt_sec) * 1e6
    ob_pose = build_orbit_chase_pose(
        radius=9.2,
        elevation_deg=26.0,
        azimuth_deg=30.0,
        look_at_xyz=(renderer.ego_centroid[0] + 1.4, 0.0, 0.85),
        device=renderer.device,
    )
    fx_ob = (tgt_hw[1] * 0.5) / math.tan(math.radians(72.0 * 0.5))
    k_ob = torch.tensor(
        [[fx_ob, 0.0, tgt_hw[1] * 0.5], [0.0, fx_ob, tgt_hw[0] * 0.5], [0.0, 0.0, 1.0]],
        device=renderer.device,
        dtype=torch.float32,
    )
    ob_g = renderer._assemble_gaussians_at(ob_ts_us)
    ob_raw = renderer.render_camera(
        ob_ts_us, ob_pose, k_ob, tgt_hw[0], tgt_hw[1], composite_ego=True, _cached_gaussians=ob_g
    ).rgb
    _to_pil(ob_raw).save(cache_dir / "oblique_raw.jpg", quality=92)
    if refiner is not None:
        ob_ref = renderer.render_camera(
            ob_ts_us, front_c2e, front_hi_k, 576, 1024, composite_ego=False, _cached_gaussians=ob_g
        ).rgb
        ob_difix = refiner.refine(ob_raw, ob_ref)
        _to_pil(ob_difix).save(cache_dir / "oblique_difix.jpg", quality=93)
    else:
        _to_pil(ob_raw).save(cache_dir / "oblique_difix.jpg", quality=92)

    meta = {
        "cache_version": cache_version,
        "sequence_id": renderer.sequence_id,
        "has_difix": refiner is not None,
        "num_frames": num_frames,
        "timestamps_sec": timestamps_sec,
        "hole_ratios": hole_ratios,
        "ego_dim_lwh": list(renderer.ego_dim_lwh),
        "ego_centroid": list(renderer.ego_centroid),
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Start intranet Reality View visualizer v2 (with Difix3D+).")
    parser.add_argument(
        "--usdz",
        default="data/nurec_samples/sample_set/26.10_release/bb5389f1-9514-4cfa-8322-dae897ed3786/bb5389f1-9514-4cfa-8322-dae897ed3786.usdz",
    )
    parser.add_argument("--cache-dir", default="outputs/viz_cache_v2")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--no-difix", action="store_true")
    args = parser.parse_args()

    usdz_path = Path(args.usdz)
    cache_dir = Path(args.cache_dir)

    print(f"Loading NuRec scene {usdz_path} onto RTX 3080 ...")
    renderer = NuRecSceneRenderer(usdz_path, device="cuda")
    refiner: Difix3DRefiner | None = None
    if not args.no_difix:
        print("Loading NVIDIA Difix3D+ (nvidia/difix_ref, FP16) ...")
        refiner = Difix3DRefiner.from_pretrained("nvidia/difix_ref", device="cuda", dtype=torch.float16)
        print("Loaded Difix3D+ successfully!")

    projector = MultiViewDepthProjector(splat_radius_px=1).to(renderer.device)
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
                el = float(qs.get("el", ["20"])[0])
                radius = float(qs.get("radius", ["7.2"])[0])
                t_sec = max(0.05, min(19.5, float(qs.get("t", ["1.0"])[0])))
                show_ego = qs.get("ego", ["1"])[0] != "0"
                use_difix = qs.get("difix", ["1"])[0] != "0"
                mode = qs.get("mode", ["gt"])[0]

                tgt_hw = (480, 832)
                t0 = time.perf_counter()
                with gpu_lock:
                    chase_pose = build_orbit_chase_pose(
                        radius=radius,
                        elevation_deg=el,
                        azimuth_deg=az,
                        look_at_xyz=(renderer.ego_centroid[0] + 1.4, 0.0, 0.85),
                        device=renderer.device,
                    )
                    fx = (tgt_hw[1] * 0.5) / math.tan(math.radians(72.0 * 0.5))
                    k_tgt = torch.tensor(
                        [[fx, 0.0, tgt_hw[1] * 0.5], [0.0, fx, tgt_hw[0] * 0.5], [0.0, 0.0, 1.0]],
                        device=renderer.device,
                        dtype=torch.float32,
                    )
                    ref_front: torch.Tensor | None = None
                    if mode == "coarse":
                        cam_names = NuRecSceneRenderer.DEFAULT_SURROUND_CAMERAS
                        b = renderer.render_frame_bundle(
                            t_sec * 1e6,
                            surround_cameras=cam_names,
                            src_hw=(224, 384),
                            tgt_hw=tgt_hw,
                            chase_pose_rig=chase_pose,
                        )
                        ref_front = b["surround_rgb"]["camera_front_wide_120fov"]
                        surr_rgb = torch.stack([b["surround_rgb"][c].permute(2, 0, 1) for c in cam_names], dim=0).unsqueeze(0)
                        surr_dep = torch.stack([b["surround_depth"][c].unsqueeze(0) for c in cam_names], dim=0).unsqueeze(0)
                        proj = projector(
                            surr_rgb,
                            surr_dep,
                            b["src_intrinsics"].unsqueeze(0),
                            b["src_c2e"].unsqueeze(0),
                            k_tgt.unsqueeze(0),
                            chase_pose.unsqueeze(0),
                            target_hw=tgt_hw,
                        )
                        if show_ego:
                            proj = renderer.ego_compositor(
                                proj,
                                k_tgt.unsqueeze(0),
                                chase_pose.unsqueeze(0),
                                skybox_textures=renderer.skybox,
                                T_rig_world=renderer.rig_to_nurec(t_sec * 1e6),
                            )
                        rgb_hwc = proj.coarse_rgb[0].permute(1, 2, 0)
                    else:
                        gaussians = renderer._assemble_gaussians_at(t_sec * 1e6)
                        out = renderer.render_camera(
                            t_sec * 1e6,
                            chase_pose,
                            k_tgt,
                            tgt_hw[0],
                            tgt_hw[1],
                            composite_ego=show_ego,
                            _cached_gaussians=gaussians,
                        )
                        rgb_hwc = out.rgb
                        if use_difix and refiner is not None:
                            fk = renderer.camera_intrinsics("camera_front_wide_120fov", 576, 1024)
                            fc = torch.from_numpy(
                                renderer.cameras["camera_front_wide_120fov"].T_sensor_rig
                            ).to(renderer.device)
                            ref_front = renderer.render_camera(
                                t_sec * 1e6, fc, fk, 576, 1024, _cached_gaussians=gaussians
                            ).rgb

                    if use_difix and refiner is not None:
                        rgb_hwc = refiner.refine(rgb_hwc, ref_front)

                    img = _to_pil(rgb_hwc)
                    torch.cuda.synchronize()
                ms = round((time.perf_counter() - t0) * 1000)
                self._send_bytes(_pil_to_jpeg_bytes(img, quality=92), "image/jpeg", {"X-Render-Ms": str(ms)})
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Visualizer v2 READY on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
