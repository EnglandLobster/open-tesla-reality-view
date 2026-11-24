#!/usr/bin/env python3
"""Verify local/Colab GPU capabilities and required library versions."""

from __future__ import annotations

import importlib
import platform
import sys


def main() -> None:
    print(f"Python:   {sys.version.split()[0]} ({platform.platform()})")
    packages = ["torch", "diffusers", "transformers", "accelerate", "peft", "einops", "yaml"]
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            print(f"{pkg:14s}: {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"{pkg:14s}: NOT INSTALLED")

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            print(f"\nCUDA Device: {props.name} ({vram_gb:.1f} GB VRAM, CC {props.major}.{props.minor})")
            print(f"BF16 Support: {torch.cuda.is_bf16_supported()}")
            print(f"SDPA Flash:   {torch.backends.cuda.flash_sdp_enabled()}")
        else:
            print("\nCUDA Device: None (CPU only)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
