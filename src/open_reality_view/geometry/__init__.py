"""3D geometry utilities: Plücker ray maps, multi-view depth projection, and ego-car compositing."""

from open_reality_view.geometry.ego_car import EgoVehicleCompositor
from open_reality_view.geometry.plucker import FourierPluckerEncoder, compute_plucker_rays
from open_reality_view.geometry.projector import (
    MultiViewDepthProjector,
    ProjectionResult,
    build_chase_camera_pose,
    build_orbit_chase_pose,
)

__all__ = [
    "EgoVehicleCompositor",
    "FourierPluckerEncoder",
    "MultiViewDepthProjector",
    "ProjectionResult",
    "build_chase_camera_pose",
    "build_orbit_chase_pose",
    "compute_plucker_rays",
]
