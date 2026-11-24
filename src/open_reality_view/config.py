"""Configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "v1_alpamayo2_edge_wan21_vace"
    seed: int = 42
    precision: str = "bfloat16"


@dataclass
class DataConfig:
    num_cameras: int = 6
    camera_names: list[str] = field(
        default_factory=lambda: [
            "cam_front",
            "cam_front_left",
            "cam_front_right",
            "cam_back",
            "cam_back_left",
            "cam_back_right",
        ]
    )
    target_height: int = 480
    target_width: int = 832
    clip_num_frames: int = 49
    fps: int = 16
    default_chase_offset_xyz: list[float] = field(default_factory=lambda: [-7.5, 0.0, 4.2])
    default_chase_pitch_deg: float = -18.0
    ego_box_lwh: list[float] = field(default_factory=lambda: [4.72, 1.85, 1.44])


@dataclass
class EncoderConfig:
    type: str = "alpamayo2_edge"
    model_id: str = "nvidia/Alpamayo2-Edge"
    freeze: bool = True
    vision_hidden_size: int = 1152
    patch_size: int = 16
    spatial_merge_size: int = 2
    enable_policy_branch: bool = False


@dataclass
class GeometryConfig:
    depth_model_id: str = "depth-anything/DA3-BASE"
    freeze: bool = True
    splat_radius_px: int = 1
    min_depth_m: float = 0.5
    max_depth_m: float = 120.0
    coarse_rgb_dropout_prob: float = 0.15


@dataclass
class ContextAdapterConfig:
    num_latents: int = 256
    adapter_dim: int = 1152
    output_dim: int = 4096
    num_layers: int = 4
    num_heads: int = 16
    plucker_freq_bands: int = 6
    dropout: float = 0.05


@dataclass
class GeneratorConfig:
    model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    vae_spatial_stride: int = 8
    vae_temporal_stride: int = 4
    latent_channels: int = 16
    vace_in_channels: int = 96
    vace_context_scale: float = 1.0
    flow_shift: float = 3.0
    v1_inference_steps: int = 25
    guidance_scale: float = 1.0
    lora_rank: int = 64
    lora_alpha: int = 64
    train_vace_blocks: bool = True


@dataclass
class TrainingConfig:
    stage1a_steps: int = 5000
    stage1b_steps: int = 25000
    learning_rate_adapter: float = 1.0e-4
    learning_rate_dit: float = 2.0e-5
    weight_decay: float = 1.0e-2
    grad_clip_norm: float = 1.0
    gradient_checkpointing: bool = True


@dataclass
class RealityViewConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    context_adapter: ContextAdapterConfig = field(default_factory=ContextAdapterConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _from_dict(cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return cls()
    return cls(**data)


def load_config(path: str | Path) -> RealityViewConfig:
    """Load a RealityViewConfig from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RealityViewConfig(
        experiment=_from_dict(ExperimentConfig, raw.get("experiment")),
        data=_from_dict(DataConfig, raw.get("data")),
        encoder=_from_dict(EncoderConfig, raw.get("encoder")),
        geometry=_from_dict(GeometryConfig, raw.get("geometry")),
        context_adapter=_from_dict(ContextAdapterConfig, raw.get("context_adapter")),
        generator=_from_dict(GeneratorConfig, raw.get("generator")),
        training=_from_dict(TrainingConfig, raw.get("training")),
    )
