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


def _orthonormalize_so3(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(R, dtype=float))
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def _minimal_rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return exact SO(3) matrix rotating unit vector `a` onto unit vector `b` along the shortest arc."""
    ua = np.asarray(a, dtype=float)
    ub = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(ua))
    nb = float(np.linalg.norm(ub))
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3, dtype=float)
    ua = ua / na
    ub = ub / nb
    c = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    if c > 1.0 - 1e-9:
        return np.eye(3, dtype=float)
    if c < -1.0 + 1e-9:
        ortho = np.array([1.0, 0.0, 0.0]) if abs(ua[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(ua, ortho)
        axis /= float(np.linalg.norm(axis))
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=float)
    v = np.cross(ua, ub)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )
    return np.eye(3, dtype=float) + vx + (vx @ vx) / (1.0 + c)


def snap_rotation_to_table(
    R_raw: np.ndarray,
    ref_obj: RefObject | None,
    fallback_height_m: float = 0.10,
) -> tuple[np.ndarray, float, float, bool]:
    """Snap pitch/roll flush to `z = z_table` while preserving sideways/flipped 3D resting orientation."""
    R0 = _orthonormalize_so3(R_raw)

    if ref_obj is not None and ref_obj.shape == "sphere":
        h_rest = float(ref_obj.extents()[2])
        yaw = float(math.atan2(R0[1, 0], R0[0, 0]))
        return R0, h_rest, yaw, True

    if ref_obj is not None and ref_obj.shape == "cylinder":
        ext = ref_obj.extents()  # [2R, 2R, H]
        ez_w = R0[:, 2]
        if abs(float(ez_w[2])) >= math.cos(math.radians(55.0)):
            s_up = 1.0 if ez_w[2] >= 0.0 else -1.0
            R_snap = _orthonormalize_so3(_minimal_rot_a_to_b(ez_w, np.array([0.0, 0.0, s_up])) @ R0)
            yaw = float(math.atan2(R_snap[1, 0], R_snap[0, 0]))
            return R_snap, float(ext[2]), yaw, bool(s_up > 0.0)
        horiz = np.array([ez_w[0], ez_w[1], 0.0], dtype=float)
        hn = float(np.linalg.norm(horiz))
        axis_xy = horiz / hn if hn > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=float)
        R_snap = _orthonormalize_so3(_minimal_rot_a_to_b(ez_w, axis_xy) @ R0)
        yaw = float(math.atan2(axis_xy[1], axis_xy[0]))
        return R_snap, float(ext[0]), yaw, False

    # Box or unmatched 3D cloud: snap nearest local signed axis onto +Z_world
    k_up = int(np.argmax(np.abs(R0[2, :])))
    s_up = 1.0 if R0[2, k_up] >= 0.0 else -1.0
    R_snap = _orthonormalize_so3(_minimal_rot_a_to_b(s_up * R0[:, k_up], np.array([0.0, 0.0, 1.0])) @ R0)
    upright = bool(k_up == 2 and s_up > 0.0)

    if ref_obj is not None:
        ext = ref_obj.extents()
        h_rest = float(ext[k_up])
        horiz_axes = [j for j in range(3) if j != k_up]
        j_long = max(horiz_axes, key=lambda j: float(ext[j]))
        yaw = float(math.atan2(R_snap[1, j_long], R_snap[0, j_long]))
    else:
        h_rest = max(float(fallback_height_m), 0.01)
        j_h = 0 if k_up != 0 else 1
        yaw = float(math.atan2(R_snap[1, j_h], R_snap[0, j_h]))

    return R_snap, h_rest, yaw, upright


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
            "resting_height_m": round(float(self.resting_height_m), 4),
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

    def set_scan_pose(
        self,
        center_xy: np.ndarray,
        height_m: float,
        yaw_rad: float,
        *,
        rotation: np.ndarray | None = None,
        upright: bool = True,
    ) -> ObjectPoseEstimate:
        """Update initial resting pose while the object is still sitting on the scan spot."""
        if self.phase != "scanning":
            assert self._current_pose is not None
            return self._current_pose
        h_use = max(float(height_m), 0.01)
        z_c = self.z_table + 0.5 * h_use
        if rotation is not None:
            R_use = _orthonormalize_so3(rotation)
            pos = np.array([float(center_xy[0]), float(center_xy[1]), float(z_c)], dtype=float)
        else:
            T = upright_se3(center_xy, z_c, yaw_rad)
            R_use = T[:3, :3].copy()
            pos = T[:3, 3].copy()
        est = ObjectPoseEstimate(
            phase="scanning",
            position_m=pos,
            rotation=R_use,
            yaw_rad=float(yaw_rad),
            resting_height_m=h_use,
            upright=bool(upright),
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
        rotation: np.ndarray | None = None,
        upright: bool | None = None,
    ) -> ObjectPoseEstimate:
        """Update resting pose in `placed` phase when additional tactile points refine the object."""
        prev = self._current_pose
        fallback_h = max(float(height_m), 0.01 if prev is None else prev.resting_height_m)
        if rotation is not None:
            R_raw = rotation
        elif prev is not None:
            R_raw = prev.rotation
        else:
            R_raw = upright_se3(center_xy, 0.0, yaw_rad)[:3, :3]

        R_use, h_rest, yaw_snap, up_snap = snap_rotation_to_table(R_raw, ref_obj, fallback_h)
        if upright is not None:
            up_snap = bool(upright)

        if prev is None:
            xy_use = np.asarray(center_xy, dtype=float)[:2].copy()
            yaw_use = float(yaw_rad) if rotation is None else yaw_snap
        else:
            xy_use = prev.position_m[:2].copy()
            yaw_use = prev.yaw_rad if rotation is None else yaw_snap

        z_c = self.z_table + 0.5 * h_rest
        est = ObjectPoseEstimate(
            phase="placed",
            position_m=np.array([float(xy_use[0]), float(xy_use[1]), float(z_c)], dtype=float),
            rotation=R_use.copy(),
            yaw_rad=yaw_use,
            resting_height_m=h_rest,
            upright=up_snap,
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
            upright=bool(R_now[2, 2] >= math.cos(math.radians(55.0))),
        )
        return self._current_pose

    def place_on_table(
        self,
        T_ws_hand: np.ndarray,
        finger_pads_ws: np.ndarray | None = None,
        ref_obj: RefObject | None = None,
    ) -> ObjectPoseEstimate:
        """Compute final table-snapped 3D pose right before the user opens their hand."""
        fallback_h = 0.10 if self._current_pose is None else self._current_pose.resting_height_m
        if self._T_hand_obj is None:
            # If user jumped straight to PLACED without LOCK_GRASP, centre inside current finger pads
            pads = np.zeros((1, 3)) if (finger_pads_ws is None or len(finger_pads_ws) == 0) else finger_pads_ws
            xy_raw = pads[:, :2].mean(axis=0)
            R_raw = np.eye(3, dtype=float)
        else:
            T_raw = T_ws_hand @ self._T_hand_obj
            R_raw = T_raw[:3, :3]
            xy_raw = T_raw[:2, 3].copy()

        R_snap, H_rest, yaw_raw, upright = snap_rotation_to_table(R_raw, ref_obj, fallback_h)

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
            d_yaw = yaw_fin - yaw_raw
            cy, sy = math.cos(d_yaw), math.sin(d_yaw)
            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            R_snap = _orthonormalize_so3(Rz @ R_snap)

        z_center = self.z_table + 0.5 * H_rest
        self.phase = "placed"
        self._current_pose = ObjectPoseEstimate(
            phase="placed",
            position_m=np.array([float(xy_fin[0]), float(xy_fin[1]), float(z_center)], dtype=float),
            rotation=R_snap.copy(),
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
