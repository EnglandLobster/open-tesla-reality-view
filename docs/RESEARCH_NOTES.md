# Research & Engineering Notes: Reality View 全链路、GT 方案与 2024–2026 学术调研

本文档系统记录 `open-tesla-reality-view` 在架构设计、离线 Ground Truth（GT）数据工厂评估、2024–2026 学术界前沿流派调研以及关键渲染工程排障中的核心结论，作为后续 Stage 0 改造、v1 训练与 v2 实时化演进的参考依据。

---

## 1. 全链路总览：离线“造题工厂” vs 线上“实时推理模型”

理解本项目的核心前提是区分两个完全独立的阶段：

1. **Stage 0 离线标准答案工厂（`NuRec 26.10 + Difix3D+`）**
   - 现实中的量产车车尾没有无人机，无法直接同步采集“6 路车身摄像头”与“车后 13 米高空俯视画面”的配对数据。
   - 因此借助 NVIDIA 离线重建好的 3D 高斯驾驶沙盘（`PhysicalAI-Autonomous-Vehicles-NuRec`，1,601 个 20 秒真实街景），在同一时间戳下既渲染 6 路车载输入，又渲染车后第三人称高清真值（经 `Difix3D+` 去伪影），批量生产 `(6路输入 -> 第三人称真值)` 训练对。
   - **定位**：离线 ETL 数据工厂，**不会也不可能部署到陌生道路的车端**。

2. **线上推理模型（`Branch A + Branch B + Wan2.1-VACE + Post-VAE UI`）**
   - 上线后彻底脱离 NuRec 沙盘：只要输入任意陌生道路的 6 路环视视频 + 车辆里程计，即可端到端生成车后第三人称 Reality View。

```mermaid
flowchart LR
    IN["线上输入：真车 6 路环视视频 + 里程计"] --> A
    IN --> B

    subgraph A["Branch A：3D 几何缓存 (不训练，确定性计算)"]
        A1["Depth Anything 3 预测深度"] --> A2["Temporal3DCache 多帧点云融合"] --> A3["目标视角带洞草图 + 盲区 Mask"]
    end

    subgraph B["Branch B：驾驶视觉大脑 (主干冻结，仅训适配层)"]
        B1["Alpamayo2-Edge Vision Tower（冻结）"] --> B2["⭐ 训练 #1：AlpamayoContextAdapter (1152→4096)"]
    end

    A3 --> G["⭐ 训练 #2：Wan2.1-VACE-1.3B 主干\n(盲区脑补 + 纹理精修 + 时序一致)"]
    B2 --> G

    GT["🎯 离线 Ground Truth\n(NuRec 3DGUT + Difix3D+ 纯背景真值)"] -. "Flow Matching Loss 监督" .-> G

    G --> POST["后处理 UI 渲染层 (不训练)\nOpenGL 4×MSAA PBR 自车 + 电光蓝轨迹毯"]
    POST --> OUT["输出：车后第三人称 Reality View"]
```

### 真正需要训练的模块
| 模块 | 代码路径 | 是否训练 | 作用 |
| :--- | :--- | :---: | :--- |
| `Temporal3DCacheProjector` + DA3 | `src/open_reality_view/geometry/projector.py` | 否（确定性/冻结） | 构造几何脚手架，将 ~90% 已知像素锚定在准确 3D 坐标，防幻觉 |
| `AlpamayoVisionEncoder` | `src/open_reality_view/models/alpamayo_encoder.py` | 否（冻结） | 复用开源智驾大模型视觉表征（光照、语义、动态物体） |
| **`AlpamayoContextAdapter`** | `src/open_reality_view/models/alpamayo_encoder.py` | **是（Stage 1A/1B）** | 融合 Plücker Ray 视角编码并做 `1152 -> 4096` 协议转换 |
| **`WanVACETransformer3DModel`** | `src/open_reality_view/models/vace_reality_view.py` | **是（Stage 1B）** | 核心视频生成/修复主干，学习补全盲区、去伪影与多帧连贯 |
| `EgoVehicleCompositor` + `trajectory_overlay` | `src/open_reality_view/geometry/ego_car.py` | 否（OpenGL 渲染） | 在生成背景最上层叠加 4× MSAA PBR 车模与深度遮挡轨迹带 |

---

## 2. 对照特斯拉 FSD v15 Reality View

| 维度 | 特斯拉 FSD v15 Reality View | `open-tesla-reality-view` 路线 | 对齐度与取舍 |
| :--- | :--- | :--- | :--- |
| **核心范式** | 共享 FSD 感知视觉主干（UVE），将“智驾眼中的 3D 世界”直接解码为写实第三人称画面 | 冻结开源智驾视觉主干 `Alpamayo2-Edge` + 神经视频生成解码器 | **核心思想完全对齐** |
| **3D 几何防幻觉** | 依赖海量数据在 UVE 内部学出的强隐式 4D 空间表征 | 外挂 **Branch A（DA3 + 多帧 3D Cache）** 显式投影脚手架 + 15% Dropout | **中小算力务实折中**：500 A100h 级即可稳收敛且零车体漂移 |
| **训练 GT 来源** | 百万车队同路段多车次数据 + 内部大规模 3DGS / Neural World Simulator 重建 | NVIDIA `PhysicalAI-AV-NuRec`（1,601 场景 3DGUT）+ `Difix3D+` | **方法论同源**，开源微缩版 |
| **自车与规划蓝毯** | 车端高精度 3D 自车模型 + 实时环境光反射 + FSD Planner 电光蓝路径毯 | 4× MSAA 213k 面 PBR `.glb` 车模 + NuRec 天空盒 IBL + 深度遮挡蓝毯（后置叠加） | **100% 对齐** |
| **推理与实时性** | AI4/HW4 车端 30+ FPS 流式因果解码（8-frame chunk Mean-Flow） | **v1** 先跑通多步高质量生成；**v2** 做 `Causal-Forcing` 1–2 步蒸馏 + `LongLive` KV-Cache + `Stream-VAE` + FP8 | **先质量、后实时化** |

---

## 3. Ground Truth 生成方式评估与待修工程清单

### 3.1 行业 4 种第三人称 GT 路线对比（2026）
| 路线 | 代表方式 | 优势 | 局限性 |
| :--- | :--- | :--- | :--- |
| **1. 纯游戏引擎仿真** | CARLA / UE5 | 零伪影、几何与深度 100% 精确、视角无限自由 | Sim-to-Real 域差大，真实视频泛化易出现“塑料游戏感” |
| **2. 车队多车重跑 / 实拍** | Tesla / Waymo 内部车队 | 真实光学、零重建伪影 | 开源界无法获取 |
| **3. 预重建 3DGS + 单步扩散精修**（**当前采用**） | `NuRec 26.10` + `Difix3D+` | **开源性价比最高**：真实驾驶光学 + 省去数千卡时单场景建图 | ① 极远处天空/电线偶发 3DGS 团状残影；② 逐帧 `Difix3D+` 有轻微帧间高频抖动；③ 源 6 路输入为二次渲染而非原生 RAW |
| **4. 3DGS + 视频扩散迭代反哺** | `ReconDreamer++` / `DriveDreamer4D` / `StreetCrafter` | 大偏角与帧间一致性达到离线天花板 | 算力贵 **10–30×**（单场景反复跑视频扩散 + 重训 3DGS），难铺千级场景 |

**结论**：主干采用 `NuRec 26.10 + Difix3D+` 仍是开源第一梯队选择，无需推倒重来。

### 3.2 开训前必须修复的 Stage 0 工程缺陷（代码审计结论）
对 `scripts/batch_export_nurec.py` 与 `scripts/precache_features.py` 审计发现以下 **4 项高优先级改进**，须在大规模导出前落地：

1. **GT 必须导出纯背景（严禁先贴自车再过 `Difix3D+`）**
   - 现状：`batch_export_nurec.py` 调用 `render_frame_bundle(..., composite_ego=True)` 后再调用 `refiner.refine()`，会把自车送进 VAE 糊化，且迫使生成模型浪费容量去学画自车。
   - 改造：改为渲染纯背景 → 原生分辨率 `Difix3D+` 精修 → 直接保存为 `chase_gt`；自车与蓝毯仅在推理/可视化后处理叠加。
2. **根治深度图的 Train-Test Skew（训练/推理分布漂移）**
   - 现状：`batch_export_nurec.py` 导出的是 NuRec 完美 3DGS 深度；但线上推理到陌生马路时，只能拿到 `Depth Anything 3` 的带噪估计深度。
   - 改造：在 `precache_features.py` 中优先使用 `DA3-BASE` 估计深度（或按比例混合 DA3 深度 + 随机尺度/边缘抖动）生成 Branch A 投影草图，保证训练分布与真实线上一致。
3. **缓解逐帧 `Difix3D+` 的时序闪烁（Temporal Flicker）**
   - 改造：筛选高 PSNR 场景（`>= 30 dB`）；探索为 `Difix3D+` 提供多参考视角（当前帧前视 + 上一帧已精修帧）或在导出阶段做跨帧一致性校验，过滤残缺严重片段。
4. **同步黄金构图与原生分辨率参考图**
   - 现状：`batch_export_nurec.py` 仍停留在旧低头视角（`offset=(-7.5,0,4.2), pitch=-18°`）与硬编码 `576×1024` 参考图。
   - 改造：同步至已验证的 Tesla 追尾构图（`radius=13.8m, elevation=11.5°, fov_x=74°, look_at=(ego_x+4.5, 0, 1.25)`）并加入随机视角扰动（`azimuth ∈ [-30°, +30°]`、`elevation ∈ [10°, 22°]`），且参考图分辨率与目标分辨率精确对齐，消除二次缩放模糊。

---

## 4. 2024–2026 学术界四大技术流派调研

| 流派 | 代表工作（2024–2026） | 核心机制 | 优势 | 瓶颈 |
| :--- | :--- | :--- | :--- | :--- |
| **流派一：前馈 3D/4D 高斯预测**<br>*(Feed-Forward 3DGS)* | **Instant NuRec** (NVIDIA, 2026)<br>**STORM** (NVIDIA, 2025)<br>**DrivingForward** (AAAI 2025)<br>**Omni-Scene** (CVPR 2025) | 用 ViT/DA3 风格主干单次前向（200ms–1.5s）直接从多视角视频预测分层动态 3DGS | **极快、360° 强一致**；可用普通 6 路视频**自监督训练**，不依赖第三人称 GT | 回归式预测在大偏角（车后高空）盲区易发糊、拉丝、出浮空噪点，视觉质感不及扩散模型 |
| **流派二：3D 缓存 + 可控视频扩散**<br>*(3D-Guided Video Diffusion)*<br>**← 我们当前 v1 路线** | **GEN3C** (NVIDIA, CVPR 2025)<br>**TrajectoryCrafter** (腾讯 ARC, ICCV 2025 Oral)<br>**StreetCrafter** (ZJU3DV, 2025)<br>**ReCamMaster** (ZJU+快手, 2025) | 解耦“确定性 3D 投影草图”与“视频扩散语义生成”，用 3D Cache 锁定结构、扩散模型补全盲区与材质 | **画质天花板最高**、盲区脑补逼真；基于开源视频 DiT 微调，**中小算力（500 A100h）即可落地** | 依赖高质量第三人称配对 GT；多步采样推理慢，需 v2 步数蒸馏 |
| **流派三：离线 3DGS + 视频扩散迭代反哺**<br>*(Iterative Generative Reconstruction)* | **ReconDreamer / ReconDreamer++** (2025)<br>**DriveDreamer4D** (2025)<br>**Difix3D+** (CVPR 2025)<br>**FreeSim** (2025) | 3DGS 渲染大偏角退化图 → 视频/图像扩散修复 → 伪真值倒灌回 3DGS 重优化 | 离线重建画质与帧间稳定性极佳，适合闭环仿真资产生产 | 单场景重优化算力开销大 10–30×，难低成本规模化 |
| **流派四：前馈 4DGS + 单步扩散精修**<br>*(Hybrid Real-Time Target)* | `STORM / Instant NuRec` + `Difix3D+ / 1-step VACE` | 前馈 4DGS（~100–200ms）快速建出完整 3D 高斯场景 → 2ms 任意视角光栅化 → 1 步轻量扩散精修（~30ms） | **兼顾流派一的实时/强 3D 一致性与流派二的照片级画质**；精修器学习难度大幅降低 | 需要强前馈 4DGS 预训练权重作为前端 |

### 演进式架构策略（v1 → v2）
我们的 Branch A 接口只约定输出 `(coarse_rgb, hole_mask, target_depth)`：
- **v1（当前务实落地）**：走 **流派二**（`DA3 + Temporal3DCache` + `Alpamayo2-Edge` + `Wan2.1-VACE`），低算力快速跑出高清效果。
- **v2（平滑切换至流派四）**：已于本地 RTX 3080 20GB 实测跑通 **`nvidia/instant-nurec`**（见第 5 节），可直接**热插拔替换 Branch A 的点云缓存**，并结合 `Causal-Forcing` 1–2 步蒸馏实现车端级实时流式渲染。

---

## 5. 开源世界模型与前馈 3DGS 实测验证（2026 年底 Hugging Face 核验）

| 模型（Hugging Face ID，均已实测 `gated=False`） | 参数量 / 权重体积 | 本地 RTX 3080 20GB 实测可行性 | 工程定位 |
| :--- | :---: | :--- | :--- |
| **`nvidia/instant-nurec`** (`pa-multiview`) 🔥 | `202M` / **`0.68 GB`** | **✅ 已实测跑通！**<br>• 5路×6帧 (30图)：**`867 ms` / 显存增量 `3.86 GB`**，覆盖率 **`95.5%`**<br>• 5路×12帧 (60图)：**`1.73 s` / 显存增量 `6.81 GB`**，覆盖率 **`96.1%`**<br>• 免费自带 5 类语义分割 + 观测天空盒 | **超级 Branch A 底座候选**（从 DA3 满屏黑洞跃升至 `96.1%` 完整 3DGS 半成品）；不可直接替代离线 GT |
| **`nvidia/GEN3C-Cosmos-7B`** | `7B` / `26.96 GB` | **❌ 3080 跑不动**（全卸载仍需 `~43 GB VRAM`，需 A100/H100 80GB） | 留到 Colab A100 80GB 阶段作为离线视频级精修 Teacher 对比 `Difix3D+` |
| **`depth-anything/DA3-GIANT-1.1`** / **`tencent/HY-World-2.0`** | ~1.2B | 可评估 | 通用静态 3D 重建强，动态车辆拆分不如 `Instant-NuRec` 专业 |
| **`KlingTeam/ReCamMaster-Wan2.1`** / **`TrajectoryCrafter`** | 1.3B–5B | 单目转视角 | 盲区易幻觉，不适合直接造自动驾驶 GT；`ReCamMaster-Wan2.1` 相机控制注入设计可参考 |

### `Instant-NuRec` 零离线地图 / 零训练实测核心结论（客观像素级复盘）
- **实测方法**：仅输入 5 路因果环视图像（`504×280`，全为当前与过去帧，零离线 3D 地图、零未来帧），经 `Instant-NuRec` (`pa-multiview`) 现场生成 3D 高斯 → `triton_3dgs.py` 渲染 720p → `Difix3D+` 单步精修 → 后置 4× MSAA PBR 自车与蓝毯。
- **客观差距（零训练 `Instant-NuRec + Difix3D+` ≈ 55–60 分 vs 离线 `NuRec 26.10 + Difix3D+` ≈ 85–90 分）**：
  1. **主车道大结构极稳**：自车周边主车道、双黄线、绿色自行车道、红木围墙下半段及紧邻前车 SUV 位置严丝合缝无漂移；
  2. **四大明显短板**：① 右下角路边停靠银车出现断裂变形（`504×280` 边缘畸变 + 过滤 `MOVABLE` 误伤静止车）；② 右上角大棕榈树几乎消失且天空边缘带冷灰锯齿；③ 左侧一排白色防撞小立柱全部糊没；④ 前车上方远景路口发雾；
  3. **三条铁律结论**：**❌ 绝不能用 `Instant-NuRec` 替代离线 `NuRec 26.10` 造 GT**；**❌ 绝不能幻想靠免训练拼装追上特斯拉**；**✅ 它真正赢麻的对象是旧的 DA3 稀疏点云（Branch A），必须后接经高质量 GT 微调过的 `Alpamayo + Wan2.1-VACE` 才能还原旁车/立柱/棕榈树/落日光影**。

---

## 6. 特斯拉 GT 来源分析与开源“混合多源 GT”配方

### 6.1 特斯拉 FSD v15 Reality View 的 GT 并非单一 3DGS
特斯拉未公开完整训练数据配方，但综合马斯克与 Ashok Elluswamy 公开披露的信息，其 GT 必然是 **4 种来源混合**：
1. **自研动态 3DGS / 离线神经场景重建引擎**（Ashok 公开演示过 ~220ms 级动态 3DGS 与离线场景重渲染）；
2. **云端超大“像素级世界模型”**（马斯克确认车端 ~1B Conditional Mean-Flow 很大程度蒸馏自云端超大世界模型）；
3. **海量互联网视频预训练**（共享 UVE 底座天然具备航拍、无人机跟拍与车外第三人称先验）；
4. **百万车队独有的免费真实照片监督**：同路段前后车互拍（后车前视即前车真实第三人称跟拍）+ 单车“时间错位视角”（`t = 0s` 的真实前视画面正好位于 `t = 1.5s` 时车身正后方 10–15 米处）。

### 6.2 开源界最优“混合多源 GT”策略与导出耗时说明
- **导出耗时澄清**：离线 70 分钟/场景的重建成本已由 **NVIDIA 官方集群提前买单**（1,601 个预重建 `.usdz` 直接存放在 Hugging Face）；我们本机 RTX 3080 仅需下载与渲染，**480×832 17 帧仅 ~30–45 秒/条**（720p ~6.7 秒/帧，A100 仅 ~10–15 秒/条），每个 20 秒场景可切出 3–5 条训练样本。
- **推荐混合配方（突破单一 3DGS 天花板）**：
  1. **~70% 主力 — 提纯版 `NuRec 26.10`（纯背景 GT）**：仅选 `PSNR >= 30 dB` 高分场景，纯背景精修、同步 Tesla 黄金构图与多角度采样；
  2. **~15% 真实光学锐度锚点 — 真实原片视角 + 时间错位前视真图**：从同标定母库 `nvidia/PhysicalAI-Autonomous-Vehicles`（1,700+ 小时真车原片）抽原生未二次渲染帧，强迫模型保持 100% 真实相机锐度与光影；
  3. **~15% 极端几何补充 — CARLA / UE5 仿真追尾（可叠加 `Cosmos-Transfer` 写实化）**：补足 3DGS 在高俯角（30°–50°）及远景细电线杆/路牌上的先天短板；
  4. **独立评测尺子**：留出未见路段，并可接入 2026 真实多车/无人机同步跟拍数据集（如 `MV2`）作为终极零重建伪影 Benchmark。

---

## 7. NVIDIA 2026 自动驾驶与 Physical AI 开源全家桶清单（HF 实测核验）

以下资产均通过 Hugging Face API 实测确认存在并核验权限与用途：

### 7.1 自动驾驶 VLA 感知与决策大脑（Alpamayo 家族）
| 仓库 ID | 状态 | 核心能力与对本项目价值 |
| :--- | :---: | :--- |
| **`nvidia/Alpamayo2-Edge`** | ✅ 已集成 | 2.65B 车端轻量 VLA，0.41B SigLIP2 视觉塔（0.78GB）作为我们的共享 UVE，自带 64 点规划头 |
| `nvidia/Alpamayo-1.5-10B` / `Alpamayo2-Super` / `Alpamayo-R1-10B` | 参考 | 10B+ 带 Chain-of-Causation 因果思维链推理的智驾教师大模型 |

### 7.2 3D 驾驶场景重建、资产与渲染（NuRec 家族）
| 仓库 ID / GitHub | 状态 | 核心能力与对本项目价值 |
| :--- | :---: | :--- |
| **`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`** | ✅ 已集成 | 1,601 个离线精建 20s 7 相机+LiDAR 3DGS 场景（`.usdz`），Stage 0 GT 主力底座 |
| **`nvidia/PhysicalAI-Autonomous-Vehicles`** | 🔥 高价值 | **1,700+ 小时真车原始视频 + LiDAR 母库**，与 NuRec/Alpamayo 完全同标定，做真实光学监督零适配成本 |
| **`nvidia/instant-nurec`** (`github.com/NVIDIA/instant-nurec`) | ✅ 已实测 | 202M（0.68GB）前馈 3DGS，RTX 3080 上 867ms 现场生成 95.5% 覆盖率 3D 沙盘 |
| **`nvidia/difix_ref`** / `nvidia/difix` | ✅ 已集成 | 单步 3DGS 去伪影修图器（720p ~850ms，6.8GB 显存） |
| `NVlabs/alpasim` & `NVIDIA/3DGRUT` | 参考 | NVIDIA 官方 gRPC 微服务闭环驾驶仿真框架 + 3DGUT 底层渲染重建代码 |
| `nvidia/PhysicalAI-NuRec-PPISP` / `NuRec-AV-Object-Benchmark` | 参考 | 物理相机曝光/暗角/白平衡 ISP 校正模块 & 动态物体插入评测集 |

### 7.3 自动驾驶世界模型与实时少步渲染器（Cosmos 2.5 / Cosmos 3 / OmniDreams）
| 仓库 ID / GitHub | 权限 / 体量 | 核心能力与对本项目价值 |
| :--- | :---: | :--- |
| **`nvidia/omni-dreams-models`** (`nv-tlabs/omni-dreams`) 🔥🔥 | `gated=auto` / 34.5 GB | **实时闭环自动驾驶世界模型**（基于 `Cosmos-Predict2.5-2B`），已放出 **`2b_res720p_30fps_i2v_hdmap_distilled.pt`**（2B 因果蒸馏至 **720p @ 30 FPS 实时吐帧**，20s 长时序），是 v2 实时化架构的最佳开源教科书 |
| **`nvidia/Cosmos3-Nano-Sim-Transfer`** 🔥🔥 | `gated=False` / 29.7 GB | **2026 Cosmos 3 因果蒸馏少步 Sim2Real 渲染器**，吃 `depth + segmentation + edge + blur` 几步出照片级连续视频；既可把 CARLA/UE5 追尾变成照片级 GT，也值得测试直接精修 `NuRec / Instant-NuRec` 模糊视频 |
| **`nvidia/Cosmos3-Nano-Transfer-Auto`** 🔥 | `gated=False` / 32.1 GB | 自动驾驶专用多视角控制生成，支持最多 **11 路同步鱼眼/标准相机 + LiDAR**，适合晴/雨/夜极端天气数据扩增 |
| **`nvidia/Cosmos-Transfer2.5-2B`** & **`Cosmos-Predict2.5-2B` / `14B`** | `gated=auto` | 2B 轻量版含 `auto/multiview` 驾驶权重及 `distilled` 少步版本，适合多视角 Sim2Real 与数据扩增 |
| `nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams` | `gated=False` | 公开合成多视角驾驶视频数据集 |

### 7.4 通用 3D 世界生成与相机控制
| 仓库 ID | 权限 / 体量 | 核心能力与对本项目价值 |
| :--- | :---: | :--- |
| **`nvidia/GEN3C-Cosmos-7B`** | `gated=False` / 27.0 GB | 7B 级 3D Cache 相机可控视频模型（43GB 显存），留作 Colab A100 80GB 离线视频精修 Teacher 备选 |
| `nvidia/Lyra-2.0` (`arXiv:2604.13036`) | `gated=False` / 91.6 GB | 单图长程可探索 3D 世界生成，其用逐帧 3D 几何做历史帧路由以消除“空间遗忘”的思想值得长期关注 |

---

## 8. 关键渲染与工程排障记录（已解决）

1. **NuRec 时序傅里叶 Albedo 发灰洗白**（commit `a3adc4b`）
   - 根因：NuRec 26.10 `features_albedo` 形状为 `[N, K, 3]`，索引 `0` 为常值 DC 基色，`1..K-1` 为零均值时序谐波（对静态/部分动态体为 0）；此前取 `mean(dim=1)` 导致颜色被衰减 5×–20×。
   - 修复：提取 `albedo[:, 0, :]`（`nurec_renderer.py::_extract_base_albedo`）。
2. **画质糊化与自车被 VAE 模糊**（commit `282307f`）
   - 根因：早期流程先贴自车、再 `480p -> 576p -> 480p` 双线性缩放过 `Difix3D+` 且参考图仅 `224×384`。
   - 修复：`Difix3DRefiner.refine()` 默认 `process_hw=None` 原生分辨率（`720×1280`）精修纯背景，喂同分辨率前视参考图；**在 VAE 解码之后**再叠加 4× MSAA OpenGL PBR `.glb` 自车与电光蓝轨迹毯。
3. **3DGS 路面矩形“画面撕裂”与地下薄片穿帮**（commit `33cda92`）
   - 根因（双重叠加）：
     1. `triton_3dgs.py` 原设 `max_radius_px = 96` 硬截断，将屏幕半径 `150–480+ px` 的大扁平高斯强行切在 `16×16` tile 矩形边界上；
     2. NuRec 26.10 由 NVIDIA `3dgut-gsplat`（径向距离 + 3D 射线交点）训练，路面下 `1.4m` 藏有 `4.5m × 2.0m × 5mm` 扁平背景高斯；原光栅化器按高斯中心 `z` 排序，在 `z_路面 == z_中心` 处令地下薄片突然穿透路面。
   - 修复：移除 `96px` 硬截断改为透明度精确紧致椭圆分桶 `extend = sqrt(2 * ln(255 * opacity))`；采用**混合径向距离 + 大高斯逐 Tile 3D 射线交点深度 `t_hit` 排序（3DGUT / StopThePop 思想）**。
   - 收益：彻底消除所有矩形/水平撕裂，同时连带提升环视深度纯净度，将多帧 3D Cache 盲区率由 `9.2%` 进一步降至 `5.7%–6.6%`。
