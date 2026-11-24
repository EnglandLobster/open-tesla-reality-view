"""Zero-compiler Triton / PyTorch 3D Gaussian Splatting rasterizer for NuRec Stage 0 rendering."""

from open_reality_view.rendering.triton_3dgs import (
    GaussianRenderOutput,
    evaluate_sh_degree3,
    render_gaussians_triton,
    sample_cubemap_skybox,
)

__all__ = [
    "GaussianRenderOutput",
    "evaluate_sh_degree3",
    "render_gaussians_triton",
    "sample_cubemap_skybox",
]
