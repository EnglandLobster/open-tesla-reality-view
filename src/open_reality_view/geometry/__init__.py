"""3D geometry utilities: Plücker rays, multi-frame 3D cache projection, PBR ego car, trajectory ribbon."""

from open_reality_view.geometry.ego_car import EgoVehicleCompositor, load_glb_ego_mesh
from open_reality_view.geometry.plucker import FourierPluckerEncoder, compute_plucker_rays
from open_reality_view.geometry.projector import (
    MultiViewDepthProjector,
    ProjectionResult,
    Temporal3DCacheProjector,
    build_chase_camera_pose,
    build_orbit_chase_pose,
)
from open_reality_view.geometry.trajectory_overlay import (
    composite_trajectory_ribbon,
    extract_future_waypoints_ego,
)

__all__ = [
    "EgoVehicleCompositor",
    "FourierPluckerEncoder",
    "MultiViewDepthProjector",
    "ProjectionResult",
    "Temporal3DCacheProjector",
    "build_chase_camera_pose",
    "build_orbit_chase_pose",
    "composite_trajectory_ribbon",
    "compute_plucker_rays",
    "extract_future_waypoints_ego",
    "load_glb_ego_mesh",
]
