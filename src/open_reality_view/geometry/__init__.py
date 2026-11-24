"""3D geometry utilities: Plücker ray maps and multi-view depth projection."""

from open_reality_view.geometry.plucker import FourierPluckerEncoder, compute_plucker_rays
from open_reality_view.geometry.projector import (
    MultiViewDepthProjector,
    ProjectionResult,
    build_chase_camera_pose,
    build_orbit_chase_pose,
)

__all__ = [
    "FourierPluckerEncoder",
    "MultiViewDepthProjector",
    "ProjectionResult",
    "build_chase_camera_pose",
    "build_orbit_chase_pose",
    "compute_plucker_rays",
]
