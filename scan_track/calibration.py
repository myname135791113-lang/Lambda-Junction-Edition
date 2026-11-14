"""Workspace calibration: flat table plane fit + Coil Pro <-> Unitree G1 base Kabsch alignment.

Supports two fast calibration tiers:
  1. Table-Only Calibration (3 seconds, Smartgloves-only or Coil Pro):
     Place right palm/fingertips flat on the table -> fits table plane `(normal, z_table)` and
     levels any sensor tilt so `+Z` is strictly perpendicular to the table surface.
  2. Coil-to-Robot 4-Point Calibration (30 seconds, for physical Unitree G1 pick execution):
     Touch >= 3 (recommended 4-6) marked X spots on the table with both the robot hand TCP and
     the gloved Right Index finger -> solves exact rigid `SE(3)` transform `T_robot_from_coil`
     via Kabsch SVD, plus an optional 2D planar affine correction if mild magnetic distortion
     is detected near the robot motors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def kabsch_se3(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid SE(3) alignment mapping `src_pts` (N, 3) -> `dst_pts` (N, 3) with det(R) = +1.

    Returns `(R, t, rms_m)` such that `dst ~ (R @ src.T).T + t`.
    """
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("kabsch_se3 requires matching (N, 3) point arrays with N >= 3")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d_sign = float(np.linalg.det(Vt.T @ U.T))
    S = np.diag([1.0, 1.0, 1.0 if d_sign >= 0.0 else -1.0])
    R = Vt.T @ S @ U.T
    t = mu_d - R @ mu_s
    pred = (src @ R.T) + t
    rms_m = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return R, t, rms_m


def fit_table_plane(points_m: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit a flat table plane to `points_m` (N >= 3, 3).

    Returns `(unit_normal_up, z_height_along_normal, rms_thickness_m)`.
    """
    P = np.asarray(points_m, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 3:
        raise ValueError("fit_table_plane requires at least 3 3D points")
    centroid = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    residuals = (P - centroid) @ normal
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    height = float(centroid @ normal)
    return normal, height, rms


def _leveling_rotation_to_z_up(normal: np.ndarray) -> np.ndarray:
    """Minimal rotation `R` mapping `normal` -> `[0, 0, 1]`."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = float(np.linalg.norm(v))
    c = float(np.dot(n, z))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))


@dataclass
class WorkspaceCalibration:
    """Transforms raw sensor coordinates into a table-levelled frame + Unitree G1 base frame."""
    z_table: float = 0.0
    R_level: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_level: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Transform from levelled workspace frame -> Unitree G1 pelvis/waist base frame
    # Default places the table 0.45 m in front of G1 pelvis (`+X_g1` forward, `+Y_g1` left, `+Z_g1` up)
    # Note workspace uses `+X` right, `+Y` forward, `+Z` up -> G1 base uses `X_g1 = Y_ws`, `Y_g1 = -X_ws`, `Z_g1 = Z_ws`
    R_robot_from_ws: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    t_robot_from_ws: np.ndarray = field(default_factory=lambda: np.array([0.12, 0.0, -0.05]))
    # Optional 2D planar affine correction [2x3] in workspace XY for mild Coil Pro magnetic warp
    xy_warp_affine: np.ndarray | None = None
    alignment_rms_mm: float = 0.0
    calibrated_table: bool = False
    calibrated_robot: bool = False

    @classmethod
    def from_table_points(cls, raw_points: np.ndarray) -> "WorkspaceCalibration":
        normal, _, _ = fit_table_plane(raw_points)
        R_lvl = _leveling_rotation_to_z_up(normal)
        levelled = raw_points @ R_lvl.T
        z0 = float(np.median(levelled[:, 2]))
        cal = cls(
            z_table=0.0,
            R_level=R_lvl,
            t_level=np.array([0.0, 0.0, -z0]),
            calibrated_table=True,
        )
        return cal

    @classmethod
    def from_paired_landmarks(
        cls,
        coil_points: np.ndarray,
        robot_points: np.ndarray,
        *,
        fit_xy_warp: bool = False,
    ) -> "WorkspaceCalibration":
        """Full Coil Pro <-> Unitree G1 calibration from >= 3 touched table landmarks."""
        cal = cls.from_table_points(coil_points)
        ws_pts = cal.raw_to_workspace_points(coil_points)
        R_rw, t_rw, rms_m = kabsch_se3(ws_pts, robot_points)
        cal.R_robot_from_ws = R_rw
        cal.t_robot_from_ws = t_rw
        cal.alignment_rms_mm = round(rms_m * 1000.0, 2)
        cal.calibrated_robot = True

        if fit_xy_warp and len(coil_points) >= 4:
            # Map robot target back to ideal workspace XY and solve 2D affine correction
            ideal_ws = (robot_points - t_rw) @ R_rw
            A = np.c_[ws_pts[:, :2], np.ones(len(ws_pts))]
            aff, *_ = np.linalg.lstsq(A, ideal_ws[:, :2], rcond=None)  # (3, 2)
            cal.xy_warp_affine = aff.T  # (2, 3)
            warped_ws = cal.raw_to_workspace_points(coil_points)
            pred_robot = (warped_ws @ R_rw.T) + t_rw
            cal.alignment_rms_mm = round(
                float(np.sqrt(np.mean(np.sum((pred_robot - robot_points) ** 2, axis=1)))) * 1000.0, 2
            )
        return cal

    def raw_to_workspace_point(self, p: np.ndarray) -> np.ndarray:
        out = self.R_level @ p + self.t_level
        if self.xy_warp_affine is not None:
            xy1 = np.array([out[0], out[1], 1.0])
            out = np.array([*(self.xy_warp_affine @ xy1), out[2]])
        return out

    def raw_to_workspace_points(self, P: np.ndarray) -> np.ndarray:
        out = (P @ self.R_level.T) + self.t_level
        if self.xy_warp_affine is not None:
            A = np.c_[out[:, :2], np.ones(len(out))]
            out[:, :2] = A @ self.xy_warp_affine.T
        return out

    def raw_to_workspace_rot(self, R_raw: np.ndarray) -> np.ndarray:
        return self.R_level @ R_raw

    def workspace_to_robot_pose(self, pos_ws: np.ndarray, yaw_ws: float) -> dict[str, Any]:
        """Transform a table pose `(x, y, z, yaw)` in workspace coordinates to Unitree G1 base frame."""
        cy, sy = np.cos(yaw_ws), np.sin(yaw_ws)
        R_ws = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        pos_r = self.R_robot_from_ws @ pos_ws + self.t_robot_from_ws
        R_r = self.R_robot_from_ws @ R_ws
        yaw_r = float(np.arctan2(R_r[1, 0], R_r[0, 0]))
        return {
            "position_m": [round(float(v), 4) for v in pos_r],
            "yaw_rad": round(yaw_r, 4),
            "yaw_deg": round(float(np.degrees(yaw_r)), 1),
            "rotation_matrix": np.round(R_r, 5).tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "z_table": self.z_table,
            "R_level": self.R_level.tolist(),
            "t_level": self.t_level.tolist(),
            "R_robot_from_ws": self.R_robot_from_ws.tolist(),
            "t_robot_from_ws": self.t_robot_from_ws.tolist(),
            "xy_warp_affine": None if self.xy_warp_affine is None else self.xy_warp_affine.tolist(),
            "alignment_rms_mm": self.alignment_rms_mm,
            "calibrated_table": self.calibrated_table,
            "calibrated_robot": self.calibrated_robot,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkspaceCalibration":
        return cls(
            z_table=float(d.get("z_table", 0.0)),
            R_level=np.array(d["R_level"], dtype=float),
            t_level=np.array(d["t_level"], dtype=float),
            R_robot_from_ws=np.array(d["R_robot_from_ws"], dtype=float),
            t_robot_from_ws=np.array(d["t_robot_from_ws"], dtype=float),
            xy_warp_affine=None if d.get("xy_warp_affine") is None else np.array(d["xy_warp_affine"], dtype=float),
            alignment_rms_mm=float(d.get("alignment_rms_mm", 0.0)),
            calibrated_table=bool(d.get("calibrated_table", False)),
            calibrated_robot=bool(d.get("calibrated_robot", False)),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: str | Path) -> "WorkspaceCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
