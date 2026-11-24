"""Pure-Python reader & multi-view / third-person renderer for NVIDIA PhysicalAI-AV-NuRec USDZ scenes."""

from __future__ import annotations

import io
import json
import math
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from open_reality_view.geometry.ego_car import EgoVehicleCompositor
from open_reality_view.geometry.projector import (
    ProjectionResult,
    build_chase_camera_pose,
    build_orbit_chase_pose,
)
from open_reality_view.rendering.triton_3dgs import (
    GaussianRenderOutput,
    quat_wxyz_to_rotmat,
    render_gaussians_triton,
)


class _DummyNuRecObj:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


class _NuRecUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith(("nurec", "omegaconf", "ncore")):
            return _DummyNuRecObj
        return super().find_class(module, name)


class _NuRecPickleModule:
    Unpickler = _NuRecUnpickler

    @staticmethod
    def load(file_obj: Any, **kwargs: Any) -> Any:
        return _NuRecUnpickler(file_obj).load()


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def _quat_wxyz_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of ``wxyz`` quaternions ``[..., 4]``."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _interp_se3(ts: np.ndarray, poses_4x4: np.ndarray, target_t: float) -> np.ndarray:
    """Interpolate rigid 4x4 trajectory at ``target_t`` (linear translation + SVD-projected rotation)."""
    if target_t <= ts[0]:
        return poses_4x4[0].astype(np.float32)
    if target_t >= ts[-1]:
        return poses_4x4[-1].astype(np.float32)
    idx = int(np.searchsorted(ts, target_t))
    i0, i1 = idx - 1, idx
    alpha = float((target_t - ts[i0]) / max(1e-6, ts[i1] - ts[i0]))
    t_interp = (1.0 - alpha) * poses_4x4[i0, :3, 3] + alpha * poses_4x4[i1, :3, 3]
    r_blend = (1.0 - alpha) * poses_4x4[i0, :3, :3] + alpha * poses_4x4[i1, :3, :3]
    u, _, vt = np.linalg.svd(r_blend)
    r_interp = u @ vt
    if np.linalg.det(r_interp) < 0:
        u[:, -1] *= -1
        r_interp = u @ vt
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r_interp
    out[:3, 3] = t_interp
    return out


def _interp_track_pose_wxyz(
    ts: np.ndarray,
    poses_7d: np.ndarray,
    target_t: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(xyz [3], quat_wxyz [4])`` if track is active at ``target_t``, else ``None``."""
    if len(ts) == 0 or target_t < ts[0] - 5e4 or target_t > ts[-1] + 5e4:
        return None
    if target_t <= ts[0]:
        return poses_7d[0, :3].astype(np.float32), _quat_xyzw_to_wxyz(poses_7d[0, 3:7])
    if target_t >= ts[-1]:
        return poses_7d[-1, :3].astype(np.float32), _quat_xyzw_to_wxyz(poses_7d[-1, 3:7])
    idx = int(np.searchsorted(ts, target_t))
    i0, i1 = idx - 1, idx
    alpha = float((target_t - ts[i0]) / max(1e-6, ts[i1] - ts[i0]))
    xyz = (1.0 - alpha) * poses_7d[i0, :3] + alpha * poses_7d[i1, :3]
    q0 = _quat_xyzw_to_wxyz(poses_7d[i0, 3:7])
    q1 = _quat_xyzw_to_wxyz(poses_7d[i1, 3:7])
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    q = q / max(1e-6, float(np.linalg.norm(q)))
    return xyz.astype(np.float32), q.astype(np.float32)


def _interp_temporal_albedo(albedo: torch.Tensor, t_norm: float) -> torch.Tensor:
    """Interpolate time-varying albedo ``[N, K_knots, 3]`` at normalized clip time ``t_norm in [0, 1]``."""
    if albedo.ndim == 2:
        return albedo
    num_knots = albedo.shape[1]
    if num_knots == 1:
        return albedo[:, 0, :]
    pos = max(0.0, min(1.0, t_norm)) * (num_knots - 1)
    k0 = math.floor(pos)
    k1 = min(num_knots - 1, k0 + 1)
    alpha = pos - k0
    return (1.0 - alpha) * albedo[:, k0, :] + alpha * albedo[:, k1, :]


@dataclass
class NuRecCameraSpec:
    name: str
    T_sensor_rig: np.ndarray  # [4, 4] OpenCV cam-to-rig
    resolution_wh: tuple[int, int]
    focal_px: float
    principal_xy: tuple[float, float]


class NuRecSceneRenderer:
    """Loads a PhysicalAI-AV-NuRec 26.10 ``.usdz`` scene and renders arbitrary novel views on GPU."""

    DEFAULT_SURROUND_CAMERAS = (
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_rear_tele_30fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
    )

    def __init__(self, usdz_path: str | Path, device: str | torch.device = "cuda") -> None:
        self.usdz_path = Path(usdz_path)
        self.device = torch.device(device)

        with zipfile.ZipFile(self.usdz_path) as zf:
            self.rig_data = json.loads(zf.read("rig_trajectories.json"))
            self.tracks_raw = json.loads(zf.read("sequence_tracks.json"))
            self.data_info = json.loads(zf.read("data_info.json"))
            with zf.open("checkpoint.ckpt") as f:
                ckpt = torch.load(
                    io.BytesIO(f.read()),
                    map_location="cpu",
                    weights_only=False,
                    pickle_module=_NuRecPickleModule,
                )

        sd: dict[str, Any] = ckpt["state_dict"]
        self.world_to_nurec = np.array(self.rig_data["world_to_nurec"]["matrix"], dtype=np.float32)
        rt0 = self.rig_data["rig_trajectories"][0]
        self.sequence_id: str = rt0["sequence_id"]
        self.rig_ts_us = np.array(rt0["T_rig_world_timestamps_us"], dtype=np.float64)
        self.T_rig_worlds = np.array(rt0["T_rig_worlds"], dtype=np.float32)
        self.t_start_us = float(self.data_info["sequence_timestamp_interval_us"]["start"])
        self.t_stop_us = float(self.data_info["sequence_timestamp_interval_us"]["stop"])

        # Exact ego vehicle bounding box (rear-axle rig frame)
        bbox = rt0["rig_bbox"]
        self.ego_centroid = tuple(float(v) for v in bbox["centroid"])
        self.ego_dim_lwh = tuple(float(v) for v in bbox["dim"])

        # Parse camera calibrations
        self.cameras: dict[str, NuRecCameraSpec] = {}
        for calib in self.rig_data["camera_calibrations"].values():
            short_name = calib["logical_sensor_name"]
            params = calib["camera_model"]["parameters"]
            res_w, res_h = int(params["resolution"][0]), int(params["resolution"][1])
            poly = params.get("angle_to_pixeldist_poly", [0.0, float(res_w) * 0.5])
            focal = float(poly[1]) if len(poly) > 1 else float(res_w) * 0.5
            pp = params.get("principal_point", [res_w * 0.5, res_h * 0.5])
            self.cameras[short_name] = NuRecCameraSpec(
                name=short_name,
                T_sensor_rig=np.array(calib["T_sensor_rig"], dtype=np.float32),
                resolution_wh=(res_w, res_h),
                focal_px=focal,
                principal_xy=(float(pp[0]), float(pp[1])),
            )

        # Parse skybox cubemap
        self.skybox = (
            sd["model.background.textures"].to(device=self.device, dtype=torch.float32)
            if "model.background.textures" in sd
            else None
        )

        # Load static layers (background + road) onto GPU
        self.static_layers: list[dict[str, torch.Tensor]] = []
        for lname in ("background", "road"):
            prefix = f"model.gaussians_nodes.{lname}."
            if f"{prefix}positions" not in sd:
                continue
            self.static_layers.append(
                {
                    "positions": sd[f"{prefix}positions"].to(self.device, dtype=torch.float32),
                    "rotations": F.normalize(sd[f"{prefix}rotations"].to(self.device, dtype=torch.float32), dim=-1),
                    "scales": torch.exp(sd[f"{prefix}scales"].to(self.device, dtype=torch.float32)),
                    "opacities": torch.sigmoid(sd[f"{prefix}densities"].squeeze(-1).to(self.device, dtype=torch.float32)),
                    "albedo": sd[f"{prefix}features_albedo"].to(self.device, dtype=torch.float32),
                    "specular": sd[f"{prefix}features_specular"].to(self.device, dtype=torch.float32),
                }
            )

        # Load dynamic rigid vehicle Gaussians + track lookup
        dyn_prefix = "model.gaussians_nodes.dynamic_rigids."
        self.has_dynamic = f"{dyn_prefix}positions" in sd and sd[f"{dyn_prefix}positions"].numel() > 0
        if self.has_dynamic:
            self.dyn_positions = sd[f"{dyn_prefix}positions"].to(self.device, dtype=torch.float32)
            self.dyn_rotations = F.normalize(sd[f"{dyn_prefix}rotations"].to(self.device, dtype=torch.float32), dim=-1)
            self.dyn_scales = torch.exp(sd[f"{dyn_prefix}scales"].to(self.device, dtype=torch.float32))
            self.dyn_opacities = torch.sigmoid(sd[f"{dyn_prefix}densities"].squeeze(-1).to(self.device, dtype=torch.float32))
            self.dyn_albedo = sd[f"{dyn_prefix}features_albedo"].to(self.device, dtype=torch.float32)
            self.dyn_specular = sd[f"{dyn_prefix}features_specular"].to(self.device, dtype=torch.float32)
            self.dyn_cuboid_ids = sd[f"{dyn_prefix}gaussian_cuboid_ids"].to(self.device, dtype=torch.long)

            extra = sd.get("model._extra_state", {})
            self.dyn_track_ids: list[str] = list(extra.get("obj_track_ids", {}).get("dynamic_rigids", []))
            td = next(iter(self.tracks_raw.values()))["tracks_data"]
            self.tracks_by_id: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for tid, poses, tss in zip(
                td["tracks_id"], td["tracks_poses"], td["tracks_timestamps_us"], strict=True
            ):
                self.tracks_by_id[str(tid)] = (
                    np.asarray(tss, dtype=np.float64),
                    np.asarray(poses, dtype=np.float32),
                )

        # Ego vehicle compositor shifted to match NuRec rear-axle rig_bbox centroid!
        self.ego_compositor = EgoVehicleCompositor(
            ego_box_lwh=self.ego_dim_lwh,
            points_per_meter=24,
            splat_radius_px=1,
        ).to(self.device)
        # Align procedural ego center (0, 0, H/2) to NuRec rig_bbox centroid (e.g., x=+1.31m ahead of rear axle)
        shift = torch.tensor(
            [self.ego_centroid[0], self.ego_centroid[1], 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        self.ego_compositor.ego_points = self.ego_compositor.ego_points + shift[None, :]

    def rig_to_nurec(self, timestamp_us: float) -> torch.Tensor:
        T_rw = _interp_se3(self.rig_ts_us, self.T_rig_worlds, timestamp_us)
        return torch.from_numpy(self.world_to_nurec @ T_rw).to(self.device, dtype=torch.float32)

    def _assemble_gaussians_at(
        self,
        timestamp_us: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_norm = (timestamp_us - self.t_start_us) / max(1.0, self.t_stop_us - self.t_start_us)

        means_list: list[torch.Tensor] = []
        quats_list: list[torch.Tensor] = []
        scales_list: list[torch.Tensor] = []
        opacs_list: list[torch.Tensor] = []
        dc_list: list[torch.Tensor] = []
        spec_list: list[torch.Tensor] = []

        for layer in self.static_layers:
            means_list.append(layer["positions"])
            quats_list.append(layer["rotations"])
            scales_list.append(layer["scales"])
            opacs_list.append(layer["opacities"])
            dc_list.append(_interp_temporal_albedo(layer["albedo"], t_norm))
            spec_list.append(layer["specular"])

        if self.has_dynamic and self.dyn_track_ids:
            w2n_t = torch.from_numpy(self.world_to_nurec[:3, 3]).to(self.device)
            w2n_r = torch.from_numpy(self.world_to_nurec[:3, :3]).to(self.device)

            # Build per-cuboid rigid transform into NuRec frame
            num_tracks = len(self.dyn_track_ids)
            active = torch.zeros(num_tracks, device=self.device, dtype=torch.bool)
            track_t = torch.zeros((num_tracks, 3), device=self.device, dtype=torch.float32)
            track_q = torch.zeros((num_tracks, 4), device=self.device, dtype=torch.float32)
            track_q[:, 0] = 1.0

            for idx, tid in enumerate(self.dyn_track_ids):
                entry = self.tracks_by_id.get(tid)
                if entry is None:
                    continue
                interp = _interp_track_pose_wxyz(entry[0], entry[1], timestamp_us)
                if interp is None:
                    continue
                xyz_w, q_wxyz = interp
                active[idx] = True
                # Apply world_to_nurec (pure translation for NuRec 26.10, handle general rotation too)
                xyz_n = w2n_r @ torch.from_numpy(xyz_w).to(self.device) + w2n_t
                track_t[idx] = xyz_n
                track_q[idx] = torch.from_numpy(q_wxyz).to(self.device)

            valid_g = (self.dyn_cuboid_ids < num_tracks) & active[self.dyn_cuboid_ids.clamp(max=num_tracks - 1)]
            if valid_g.any():
                cids = self.dyn_cuboid_ids[valid_g]
                q_obj = track_q[cids]  # [M, 4] wxyz
                t_obj = track_t[cids]  # [M, 3]
                r_obj = quat_wxyz_to_rotmat(q_obj)  # [M, 3, 3]

                pos_local = self.dyn_positions[valid_g]
                pos_nurec = torch.bmm(r_obj, pos_local.unsqueeze(-1)).squeeze(-1) + t_obj
                rot_nurec = F.normalize(_quat_wxyz_mul(q_obj, self.dyn_rotations[valid_g]), dim=-1)

                means_list.append(pos_nurec)
                quats_list.append(rot_nurec)
                scales_list.append(self.dyn_scales[valid_g])
                opacs_list.append(self.dyn_opacities[valid_g])
                dc_list.append(_interp_temporal_albedo(self.dyn_albedo[valid_g], t_norm))
                spec_list.append(self.dyn_specular[valid_g])

        return (
            torch.cat(means_list, dim=0),
            torch.cat(quats_list, dim=0),
            torch.cat(scales_list, dim=0),
            torch.cat(opacs_list, dim=0),
            torch.cat(dc_list, dim=0),
            torch.cat(spec_list, dim=0),
        )

    def camera_intrinsics(self, cam_name: str, height: int, width: int) -> torch.Tensor:
        spec = self.cameras[cam_name]
        sx = width / float(spec.resolution_wh[0])
        sy = height / float(spec.resolution_wh[1])
        return torch.tensor(
            [
                [spec.focal_px * sx, 0.0, spec.principal_xy[0] * sx],
                [0.0, spec.focal_px * sy, spec.principal_xy[1] * sy],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def render_camera(
        self,
        timestamp_us: float,
        T_cam_rig: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        *,
        composite_ego: bool = False,
        _cached_gaussians: tuple[torch.Tensor, ...] | None = None,
    ) -> GaussianRenderOutput:
        """Render any rig-relative camera (surround OR third-person chase/orbit) at ``timestamp_us``."""
        gaussians = _cached_gaussians if _cached_gaussians is not None else self._assemble_gaussians_at(timestamp_us)
        means, quats, scales, opacs, sh_dc, sh_spec = gaussians

        T_rig_nurec = self.rig_to_nurec(timestamp_us)
        T_cam_nurec = T_rig_nurec @ T_cam_rig.to(device=self.device, dtype=torch.float32)

        out = render_gaussians_triton(
            means=means,
            quats_wxyz=quats,
            scales=scales,
            opacities=opacs,
            sh_dc=sh_dc,
            sh_rest=sh_spec,
            c2w=T_cam_nurec,
            intrinsics=intrinsics,
            height=height,
            width=width,
            skybox_textures=self.skybox,
        )

        if composite_ego:
            proj = ProjectionResult(
                coarse_rgb=out.rgb.permute(2, 0, 1).unsqueeze(0),
                hole_mask=(out.alpha < 0.05).float().unsqueeze(0).unsqueeze(0),
                target_depth=out.depth.unsqueeze(0).unsqueeze(0),
            )
            comp = self.ego_compositor(
                proj,
                intrinsics.unsqueeze(0),
                T_cam_rig.to(device=self.device, dtype=torch.float32).unsqueeze(0),
            )
            return GaussianRenderOutput(
                rgb=comp.coarse_rgb[0].permute(1, 2, 0),
                depth=comp.target_depth[0, 0],
                alpha=(comp.hole_mask[0, 0] < 0.5).float(),
            )
        return out

    @torch.no_grad()
    def render_frame_bundle(
        self,
        timestamp_us: float,
        *,
        surround_cameras: tuple[str, ...] = DEFAULT_SURROUND_CAMERAS,
        src_hw: tuple[int, int] = (224, 384),
        tgt_hw: tuple[int, int] = (480, 832),
        chase_pose_rig: torch.Tensor | None = None,
        orbit_azimuth_deg: float = 0.0,
        chase_fov_x_deg: float = 72.0,
    ) -> dict[str, Any]:
        """Render all surround views + depths AND one third-person chase view (with ego car) at ``timestamp_us``."""
        gaussians = self._assemble_gaussians_at(timestamp_us)

        surround_rgb: dict[str, torch.Tensor] = {}
        surround_depth: dict[str, torch.Tensor] = {}
        src_intrinsics: list[torch.Tensor] = []
        src_c2e: list[torch.Tensor] = []

        for cam in surround_cameras:
            spec = self.cameras[cam]
            k = self.camera_intrinsics(cam, src_hw[0], src_hw[1])
            c2e = torch.from_numpy(spec.T_sensor_rig).to(self.device)
            res = self.render_camera(
                timestamp_us,
                c2e,
                k,
                src_hw[0],
                src_hw[1],
                composite_ego=False,
                _cached_gaussians=gaussians,
            )
            surround_rgb[cam] = res.rgb
            surround_depth[cam] = res.depth
            src_intrinsics.append(k)
            src_c2e.append(c2e)

        if chase_pose_rig is None:
            if orbit_azimuth_deg != 0.0:
                chase_pose_rig = build_orbit_chase_pose(
                    radius=8.5,
                    elevation_deg=24.0,
                    azimuth_deg=orbit_azimuth_deg,
                    look_at_xyz=(self.ego_centroid[0] + 1.5, 0.0, 0.9),
                    device=self.device,
                )
            else:
                chase_pose_rig = build_chase_camera_pose(
                    offset_xyz=(self.ego_centroid[0] - 7.5, 0.0, 4.2),
                    pitch_deg=-18.0,
                    device=self.device,
                )

        fx_t = (tgt_hw[1] * 0.5) / math.tan(math.radians(chase_fov_x_deg * 0.5))
        k_tgt = torch.tensor(
            [[fx_t, 0.0, tgt_hw[1] * 0.5], [0.0, fx_t, tgt_hw[0] * 0.5], [0.0, 0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        chase_res = self.render_camera(
            timestamp_us,
            chase_pose_rig,
            k_tgt,
            tgt_hw[0],
            tgt_hw[1],
            composite_ego=True,
            _cached_gaussians=gaussians,
        )

        return {
            "surround_rgb": surround_rgb,
            "surround_depth": surround_depth,
            "src_intrinsics": (
                torch.stack(src_intrinsics, dim=0)
                if src_intrinsics
                else torch.empty((0, 3, 3), device=self.device)
            ),
            "src_c2e": (
                torch.stack(src_c2e, dim=0)
                if src_c2e
                else torch.empty((0, 4, 4), device=self.device)
            ),
            "chase_rgb": chase_res.rgb,
            "chase_depth": chase_res.depth,
            "tgt_intrinsics": k_tgt,
            "tgt_c2e": chase_pose_rig,
        }
