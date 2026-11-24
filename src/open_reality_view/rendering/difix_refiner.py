"""Single-step 3DGS/novel-view artifact remover wrapping NVIDIA Difix3D+ (``nvidia/difix_ref``).

Loads ``nvidia/difix_ref`` strictly on modern ``diffusers >= 0.34`` without ``trust_remote_code``
by applying Difix3D+'s exact VAE encoder->decoder skip connections (LoRA r=4) and UNet 2-view
cross-view self-attention directly on standard Diffusers modules.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from einops import rearrange
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from safetensors.torch import load_file as load_safetensors
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer


def _difix_vae_encoder_fwd(self: Any, sample: torch.Tensor) -> torch.Tensor:
    sample = self.conv_in(sample)
    down_acts: list[torch.Tensor] = []
    for down_block in self.down_blocks:
        down_acts.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = down_acts
    return sample


def _difix_vae_decoder_fwd(
    self: Any,
    sample: torch.Tensor,
    latent_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx].to(upscale_dtype) * self.gamma)
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
    sample = self.conv_norm_out(sample) if latent_embeds is None else self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def _patch_unet_cross_view_attention(unet: UNet2DConditionModel, num_views_getter: Any) -> None:
    """Wrap every BasicTransformerBlock.attn1 so batch views [target, ref] attend jointly."""
    from diffusers.models.attention import BasicTransformerBlock

    for module in unet.modules():
        if not isinstance(module, BasicTransformerBlock):
            continue
        orig_attn1 = module.attn1

        def _make_wrapped_attn1(attn_mod: nn.Module) -> Any:
            orig_fwd = attn_mod.forward

            def _wrapped(
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                **kwargs: Any,
            ) -> torch.Tensor:
                v = num_views_getter()
                if v > 1 and encoder_hidden_states is None and hidden_states.shape[0] % v == 0:
                    h_joint = rearrange(hidden_states, "(b v) n d -> b (v n) d", v=v)
                    out = orig_fwd(h_joint, encoder_hidden_states=None, attention_mask=attention_mask, **kwargs)
                    return rearrange(out, "b (v n) d -> (b v) n d", v=v)
                return orig_fwd(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            return _wrapped

        orig_attn1.forward = _make_wrapped_attn1(orig_attn1)  # type: ignore[method-assign]


class Difix3DRefiner(nn.Module):
    """Single-step NVIDIA Difix3D+ refiner cleaning 3DGS novel-view & ego-car artifacts."""

    def __init__(
        self,
        *,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        prompt_embeds: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.register_buffer("prompt_embeds", prompt_embeds, persistent=False)
        self.device = device
        self.dtype = dtype
        self._active_num_views = 1
        _patch_unet_cross_view_attention(self.unet, lambda: self._active_num_views)
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "nvidia/difix_ref",
        *,
        prompt: str = "remove degradation",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_dir: str | None = None,
    ) -> Difix3DRefiner:
        """Load ``nvidia/difix_ref`` with strict VAE skip-LoRA & UNet weight verification."""
        dev = torch.device(device)

        # 1. Pre-encode fixed prompt with CLIP and immediately free text encoder VRAM (~1.3 GB saved!)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=cache_dir)
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).to(dev)
        text_encoder.eval()
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(dev)
        with torch.no_grad():
            prompt_embeds = text_encoder(tokens)[0].to(device=dev, dtype=dtype)
        del text_encoder, tokenizer

        # 2. Construct Difix skip-connection VAE with exact LoRA r=4 decoder adapter
        vae_cfg_path = hf_hub_download(model_id, "vae/config.json", cache_dir=cache_dir)
        vae_weights_path = hf_hub_download(
            model_id, "vae/diffusion_pytorch_model.safetensors", cache_dir=cache_dir
        )
        raw_vae_cfg = json.loads(Path(vae_cfg_path).read_text(encoding="utf-8"))
        gamma = float(raw_vae_cfg.pop("gamma", 1.0))
        ignore_skip = bool(raw_vae_cfg.pop("ignore_skip", False))
        lora_rank = int(raw_vae_cfg.pop("lora_rank", 4))
        for drop_k in ("_class_name", "_diffusers_version", "_name_or_path"):
            raw_vae_cfg.pop(drop_k, None)

        vae = AutoencoderKL(**raw_vae_cfg)
        vae.encoder.forward = types.MethodType(_difix_vae_encoder_fwd, vae.encoder)
        vae.decoder.forward = types.MethodType(_difix_vae_decoder_fwd, vae.decoder)
        vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        vae.decoder.ignore_skip = ignore_skip
        vae.decoder.gamma = gamma

        target_suffixes = (
            "conv1",
            "conv2",
            "conv_in",
            "conv_shortcut",
            "conv",
            "conv_out",
            "skip_conv_1",
            "skip_conv_2",
            "skip_conv_3",
            "skip_conv_4",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        )
        decoder_targets = [
            name
            for name, _mod in vae.named_modules()
            if "decoder" in name and any(name.endswith(sfx) for sfx in target_suffixes)
        ]
        vae.add_adapter(
            LoraConfig(r=lora_rank, init_lora_weights="gaussian", target_modules=decoder_targets),
            adapter_name="vae_skip",
        )
        vae_sd = load_safetensors(vae_weights_path)
        vae.load_state_dict(vae_sd, strict=True)
        vae = vae.to(device=dev, dtype=dtype)

        # 3. Load standard UNet2DConditionModel (686 keys, strict) + DDPMScheduler
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).to(dev)
        scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir)

        return cls(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            device=dev,
            dtype=dtype,
        )

    @torch.no_grad()
    def refine(
        self,
        novel_rgb_hwc: torch.Tensor,
        ref_rgb_hwc: torch.Tensor | None = None,
        *,
        timestep: int = 199,
        process_hw: tuple[int, int] | None = (576, 1024),
    ) -> torch.Tensor:
        """Run single-step Difix3D+ refinement on ``novel_rgb_hwc [H, W, 3]`` in ``[0, 1]``.

        Args:
            novel_rgb_hwc: Rendered third-person image ``[H, W, 3]`` in ``[0, 1]``.
            ref_rgb_hwc: Optional sharp original camera view ``[H_r, W_r, 3]`` in ``[0, 1]`` for
                cross-view texture conditioning (recommended for ``nvidia/difix_ref``).
            timestep: Single-step denoising noise level (official Difix3D+ uses ``199``).
            process_hw: Optional internal resolution (official Difix3D+ native is ``(576, 1024)``).

        Returns:
            Refined RGB tensor ``[H, W, 3]`` in ``[0, 1]`` at the original input resolution.
        """
        orig_h, orig_w = novel_rgb_hwc.shape[:2]
        img = novel_rgb_hwc.to(device=self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        work_h, work_w = process_hw if process_hw is not None else ((orig_h // 8) * 8, (orig_w // 8) * 8)

        if (orig_h, orig_w) != (work_h, work_w):
            img = F.interpolate(img, size=(work_h, work_w), mode="bilinear", align_corners=False)
        img_m11 = (img * 2.0 - 1.0).to(dtype=self.dtype)

        if ref_rgb_hwc is not None:
            ref = ref_rgb_hwc.to(device=self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            ref = F.interpolate(ref, size=(work_h, work_w), mode="bilinear", align_corners=False)
            ref_m11 = (ref * 2.0 - 1.0).to(dtype=self.dtype)
            model_img = torch.cat([img_m11, ref_m11], dim=0)  # [2, 3, H, W]
            self._active_num_views = 2
            prompt_emb = self.prompt_embeds.expand(2, -1, -1)
        else:
            model_img = img_m11
            self._active_num_views = 1
            prompt_emb = self.prompt_embeds

        # 1. Encode via skip-capturing VAE encoder
        latents = self.vae.encode(model_img).latent_dist.mode() * self.vae.config.scaling_factor
        # Capture skip activations for the target view (index 0) before decoder pass
        target_skips = [act[:1] for act in self.vae.encoder.current_down_blocks]

        # 2. Single-step UNet denoising at t=199
        self.scheduler.set_timesteps(timesteps=[timestep], device=self.device)
        t = self.scheduler.timesteps[0]
        latent_in = self.scheduler.scale_model_input(latents, t)
        noise_pred = self.unet(
            latent_in,
            t,
            encoder_hidden_states=prompt_emb,
            return_dict=False,
        )[0]
        denoised = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 3. Decode target view (index 0) with skip activations
        self.vae.decoder.incoming_skip_acts = target_skips
        decoded = self.vae.decode(denoised[:1] / self.vae.config.scaling_factor, return_dict=False)[0]
        out_01 = ((decoded.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        if (work_h, work_w) != (orig_h, orig_w):
            out_01 = F.interpolate(out_01, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return out_01[0].permute(1, 2, 0).contiguous()
