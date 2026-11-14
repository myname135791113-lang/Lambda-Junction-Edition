"""Single-grasp relocation tracker: track a scanned object from pick-up to flat-table placement.

Math summary:
  1. At `LOCK_GRASP` (`t0`):
     Given the scanned object's initial pose on the table `T_ws_obj(t0)` and the gloved right
     wrist pose `T_ws_hand(t0)`, lock the rigid in-hand offset:
         `T_hand_obj = inv(T_ws_hand(t0)) @ T_ws_obj(t0)`
  2. During carry (`t in [t0, t1]`):
     Propagate `T_ws_obj(t) = T_ws_hand(t) @ T_hand_obj` so the 3D viewer renders the matched
     object moving smoothly in the wearer's hand.
  3. At `PLACED` (`t1`, triggered right before fingers release on the flat table):
     - Compute raw propagated pose `T_raw = T_ws_hand(t1) @ T_hand_obj`.
     - Detect whether the object is still upright or laid onto its side (`|R_raw[2,2]| < cos(55 deg)`).
     - Snap pitch/roll flush to the table plane (`z = z_table`) -> eliminates vertical & tilt drift!
     - Run damped 2D `(x, y, yaw)` finger-cage refinement against the 5 palmar contact points at
       `t1`, regularized toward `T_raw` so slight in-hand slip is corrected without sliding along
       open sides.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from glove_twin.objects import RefObject

from .cloud_hull import _sdf_table_primitive


def make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ t)
    return inv


def upright_se3(xy: np.ndarray, z_center: float, yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return make_se3(R, np.array([float(xy[0]), float(xy[1]), float(z_center)]))


@dataclass
class ObjectPoseEstimate:
    phase: str                       # "scanning" | "carrying" | "placed"
    position_m: np.ndarray           # (3,) centre of object in workspace frame
    rotation: np.ndarray             # (3, 3) orientation in workspace frame
    yaw_rad: float
    resting_height_m: float
    upright: bool = True
    cage_rms_mm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "x_m": round(float(self.position_m[0]), 4),
            "y_m": round(float(self.position_m[1]), 4),
            "z_center_m": round(float(self.position_m[2]), 4),
            "z_bottom_m": round(float(self.position_m[2] - 0.5 * self.resting_height_m), 4),
            "yaw_rad": round(float(self.yaw_rad), 4),
            "yaw_deg": round(float(math.degrees(self.yaw_rad)), 1),
            "upright": self.upright,
            "cage_rms_mm": round(float(self.cage_rms_mm), 2),
            "rotation": np.round(self.rotation, 5).tolist(),
        }


class RelocationTracker:
    """Stateful tracker for SCANNING -> CARRYING -> PLACED object pose."""

    def __init__(self, z_table: float = 0.0):
        self.z_table = z_table
        self.phase: str = "scanning"
        self._T_hand_obj: np.ndarray | None = None
        self._initial_pose: ObjectPoseEstimate | None = None
        self._current_pose: ObjectPoseEstimate | None = None

    def reset(self, z_table: float | None = None) -> None:
        if z_table is not None:
            self.z_table = z_table
        self.phase = "scanning"
        self._T_hand_obj = None
        self._initial_pose = None
        self._current_pose = None

    def set_scan_pose(self, center_xy: np.ndarray, height_m: float, yaw_rad: float) -> ObjectPoseEstimate:
        """Update initial resting pose while the object is still sitting on the scan spot."""
        if self.phase != "scanning":
            assert self._current_pose is not None
            return self._current_pose
        z_c = self.z_table + 0.5 * max(height_m, 0.01)
        T = upright_se3(center_xy, z_c, yaw_rad)
        est = ObjectPoseEstimate(
            phase="scanning",
            position_m=T[:3, 3].copy(),
            rotation=T[:3, :3].copy(),
            yaw_rad=float(yaw_rad),
            resting_height_m=max(height_m, 0.01),
            upright=True,
        )
        self._initial_pose = est
        self._current_pose = est
        return est

    def lock_grasp(
        self,
        T_ws_hand: np.ndarray,
        *,
        fallback_xy: np.ndarray | None = None,
        height_m: float = 0.10,
        yaw_rad: float = 0.0,
    ) -> ObjectPoseEstimate:
        """Freeze `T_hand_obj` at the moment the user grasps the object (from scan OR prior placement)."""
        base_pose = self._current_pose if self._current_pose is not None else self._initial_pose
        if base_pose is None:
            xy = np.zeros(2) if fallback_xy is None else fallback_xy
            base_pose = self.set_scan_pose(xy, height_m, yaw_rad)
        self._initial_pose = base_pose

        T_ws_obj0 = make_se3(base_pose.rotation, base_pose.position_m)
        self._T_hand_obj = invert_se3(T_ws_hand) @ T_ws_obj0
        self.phase = "carrying"
        self._current_pose = ObjectPoseEstimate(
            phase="carrying",
            position_m=base_pose.position_m.copy(),
            rotation=base_pose.rotation.copy(),
            yaw_rad=base_pose.yaw_rad,
            resting_height_m=base_pose.resting_height_m,
            upright=base_pose.upright,
        )
        return self._current_pose

    def refresh_placed_from_scan(
        self,
        center_xy: np.ndarray,
        height_m: float,
        yaw_rad: float,
        *,
        n_voxels: int,
        ref_obj: RefObject | None = None,
    ) -> ObjectPoseEstimate:
        """Update resting pose in `placed` phase when additional tactile points refine the object."""
        prev = self._current_pose
        upright = True if prev is None else prev.upright
        if ref_obj is not None:
            ext = ref_obj.extents()
            h_rest = float(ext[2]) if (upright or ref_obj.shape == "sphere") else float(min(ext[:2]))
        else:
            h_rest = max(height_m, 0.01 if prev is None else prev.resting_height_m)

        if prev is None:
            xy_use = np.asarray(center_xy, dtype=float)[:2]
            yaw_use = float(yaw_rad)
        else:
            xy_use = prev.position_m[:2].copy()
            yaw_use = prev.yaw_rad

        z_c = self.z_table + 0.5 * h_rest
        T_fin = upright_se3(xy_use, z_c, yaw_use)
        est = ObjectPoseEstimate(
            phase="placed",
            position_m=T_fin[:3, 3].copy(),
            rotation=T_fin[:3, :3].copy(),
            yaw_rad=yaw_use,
            resting_height_m=h_rest,
            upright=upright,
            cage_rms_mm=0.0 if prev is None else prev.cage_rms_mm,
        )
        self._current_pose = est
        return est

    def update_carry(self, T_ws_hand: np.ndarray) -> ObjectPoseEstimate | None:
        """Stream live 6-DoF carried object pose while in mid-air."""
        if self.phase != "carrying" or self._T_hand_obj is None or self._current_pose is None:
            return self._current_pose
        T_now = T_ws_hand @ self._T_hand_obj
        R_now = T_now[:3, :3]
        yaw = float(math.atan2(R_now[1, 0], R_now[0, 0]))
        self._current_pose = ObjectPoseEstimate(
            phase="carrying",
            position_m=T_now[:3, 3].copy(),
            rotation=R_now.copy(),
            yaw_rad=yaw,
            resting_height_m=self._current_pose.resting_height_m,
            upright=bool(abs(R_now[2, 2]) >= math.cos(math.radians(55.0))),
        )
        return self._current_pose

    def place_on_table(
        self,
        T_ws_hand: np.ndarray,
        finger_pads_ws: np.ndarray | None = None,
        ref_obj: RefObject | None = None,
    ) -> ObjectPoseEstimate:
        """Compute final table-snapped `(x, y, z, yaw)` right before the user opens their hand."""
        if self._T_hand_obj is None:
            # If user jumped straight to PLACED without LOCK_GRASP, centre inside current finger pads
            pads = np.zeros((1, 3)) if (finger_pads_ws is None or len(finger_pads_ws) == 0) else finger_pads_ws
            xy_raw = pads[:, :2].mean(axis=0)
            yaw_raw = 0.0
            upright = True
        else:
            T_raw = T_ws_hand @ self._T_hand_obj
            R_raw = T_raw[:3, :3]
            xy_raw = T_raw[:2, 3].copy()
            yaw_raw = float(math.atan2(R_raw[1, 0], R_raw[0, 0]))
            upright = bool(abs(R_raw[2, 2]) >= math.cos(math.radians(55.0)))

        if ref_obj is not None:
            ext = ref_obj.extents()  # [x, y, z] upright
            H_rest = float(ext[2]) if (upright or ref_obj.shape == "sphere") else float(min(ext[:2]))
        elif self._current_pose is not None:
            H_rest = self._current_pose.resting_height_m
        else:
            H_rest = 0.10

        xy_fin = xy_raw.copy()
        yaw_fin = yaw_raw
        cage_rms_mm = 0.0

        # Refine (x, y, yaw) against active finger-pad contacts at release time (corrects minor slip!)
        if (
            finger_pads_ws is not None
            and len(finger_pads_ws) >= 3
            and ref_obj is not None
            and upright
        ):
            xy_fin, yaw_fin, cage_rms_mm = _refine_pose_with_finger_cage(
                xy_raw, yaw_raw, finger_pads_ws, ref_obj, self.z_table
            )

        z_center = self.z_table + 0.5 * H_rest
        T_snapped = upright_se3(xy_fin, z_center, yaw_fin)
        self.phase = "placed"
        self._current_pose = ObjectPoseEstimate(
            phase="placed",
            position_m=T_snapped[:3, 3].copy(),
            rotation=T_snapped[:3, :3].copy(),
            yaw_rad=yaw_fin,
            resting_height_m=H_rest,
            upright=upright,
            cage_rms_mm=cage_rms_mm,
        )
        return self._current_pose


def _refine_pose_with_finger_cage(
    xy_prior: np.ndarray,
    yaw_prior: float,
    pads_ws: np.ndarray,
    ref_obj: RefObject,
    z_table: float,
) -> tuple[np.ndarray, float, float]:
    """Regularized Gauss-Newton fit of `(x, y, yaw)` to final finger-pad contacts on `ref_obj`."""
    # Only refine when >= 3 fingers are genuinely enclosing the placed object on opposing sides.
    # If the user already opened their hand or held it from one side only, trust rigid wrist carry!
    dists_xy = np.linalg.norm(pads_ws[:, :2] - xy_prior, axis=1)
    max_r = 0.5 * float(max(ref_obj.extents()[:2])) + 0.025
    inliers = pads_ws[dists_xy <= max_r]
    if len(inliers) < 3:
        return xy_prior.copy(), float(yaw_prior), 0.0

    rel = inliers[:, :2] - xy_prior
    angs = np.sort(np.arctan2(rel[:, 1], rel[:, 0]))
    gaps = np.diff(np.append(angs, angs[0] + 2.0 * math.pi))
    covered_arc = 2.0 * math.pi - float(np.max(gaps))
    if covered_arc < math.radians(100.0):
        return xy_prior.copy(), float(yaw_prior), 0.0

    x_state = np.array([float(xy_prior[0]), float(xy_prior[1]), float(yaw_prior)], dtype=float)
    prior = x_state.copy()
    # Prior regularization keeps carry-tracked yaw/along-face position stable while snapping radial slip
    reg_sigma = np.array([0.025, 0.025, 0.35])  # 25 mm XY prior, ~20 deg yaw prior
    surf_sigma = 0.004

    def residuals(st: np.ndarray) -> np.ndarray:
        cx, cy, yw = st
        if ref_obj.shape == "sphere":
            r_surf = _sdf_table_primitive("sphere", np.array([cx, cy, ref_obj.dims["radius"]]), inliers, z_table)
        elif ref_obj.shape == "cylinder":
            r_surf = _sdf_table_primitive(
                "cylinder",
                np.array([cx, cy, ref_obj.dims["radius"], ref_obj.dims["height"]]),
                inliers,
                z_table,
            )
        else:
            r_surf = _sdf_table_primitive(
                "box",
                np.array([cx, cy, yw, ref_obj.dims["x"], ref_obj.dims["y"], ref_obj.dims["z"]]),
                inliers,
                z_table,
            )
        r_prior = (st - prior) / reg_sigma
        # Wrap yaw difference cleanly
        r_prior[2] = math.atan2(math.sin(st[2] - prior[2]), math.cos(st[2] - prior[2])) / reg_sigma[2]
        return np.concatenate([r_surf / surf_sigma, r_prior])

    eps = 1e-5
    for _ in range(12):
        r0 = residuals(x_state)
        J = np.empty((len(r0), 3), dtype=float)
        for j in range(3):
            xp = x_state.copy()
            xp[j] += eps
            J[:, j] = (residuals(xp) - r0) / eps
        try:
            dx = -np.linalg.solve(J.T @ J + 1e-2 * np.eye(3), J.T @ r0)
        except np.linalg.LinAlgError:
            break
        x_state += dx
        if float(np.linalg.norm(dx)) < 1e-5:
            break

    # Compute pure surface RMS on the refined pose
    n_in = len(inliers)
    surf_res_m = residuals(x_state)[:n_in] * surf_sigma
    rms_mm = float(np.sqrt(np.mean(surf_res_m ** 2))) * 1000.0
    yaw_wrapped = math.atan2(math.sin(x_state[2]), math.cos(x_state[2]))
    return x_state[:2].copy(), yaw_wrapped, rms_mm
