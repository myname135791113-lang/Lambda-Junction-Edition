"""Grasp planning: candidate hand poses around the object, checked with inverse kinematics.

Hand frame (right Dex3-style hand, wrist-link frame): fingers point +x, the palm faces +y, the two
fingers are stacked along z (index +z, middle -z) and the thumb sits between them. Fingers curl toward
+y, the thumb comes from the -x side, so the grasp closes in the x-y plane about the point `CENTER`.

Two grasp families:
  side  wrist z is vertical: palm faces the object horizontally, fingers stacked vertically (tall objects)
  top   hand rolled so the palm faces down: fingers side by side, thumb opposing (small or flat objects)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .kin import solve_ik
from .scene import SceneInfo

CENTER = np.array([0.125, 0.060, 0.0])   # where the object axis sits in the wrist frame (m)
# Largest object width the hand can enclose, measured across the thumb-finger direction (m).
MAX_APERTURE = 0.095
MIN_APERTURE = 0.012
MAX_PAYLOAD_KG = 0.5                      # Dex3-1 rating

# hand joint order: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
OPEN = {"side": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "top": np.array([0.0, 0.0, -0.9, 0.0, 0.0, 0.0, 0.0])}
CLOSED = np.array([0.0, -0.5, -1.7, 1.5, 1.7, 1.5, 1.7])


@dataclass
class Grasp:
    kind: str
    psi_deg: float
    R: np.ndarray            # wrist rotation (3x3, world)
    wrist_pos: np.ndarray    # wrist target at the grasp
    q_grasp: np.ndarray      # arm joints at the grasp
    pre_pos: np.ndarray
    q_pre: np.ndarray
    ik_err_mm: float
    width_m: float           # object width the fingers must enclose

    def summary(self) -> dict:
        return {"kind": self.kind, "approach_yaw_deg": self.psi_deg,
                "wrist_pos_m": [round(float(x), 4) for x in self.wrist_pos],
                "object_width_mm": round(self.width_m * 1000, 1), "ik_error_mm": round(self.ik_err_mm, 2)}


def width_across(mesh, direction_xy: np.ndarray) -> float:
    """Extent of the mesh along a horizontal unit direction, at the grasp height band."""
    d = np.array([direction_xy[0], direction_xy[1], 0.0])
    p = mesh.vertices @ d
    return float(p.max() - p.min())


def width_in_plane(mesh, R: np.ndarray, kind: str) -> float:
    """Width the hand must enclose = extent along the closing direction (wrist x axis)."""
    d = R[:, 0]
    p = mesh.vertices @ d
    return float(p.max() - p.min())


def rotation(kind: str, psi_deg: float) -> np.ndarray:
    Rz = Rot.from_euler("z", psi_deg, degrees=True).as_matrix()
    return Rz if kind == "side" else Rz @ Rot.from_euler("x", -90, degrees=True).as_matrix()


def kinds_for(mesh) -> list[str]:
    """Preferred family order by object height: tall things are held from the side."""
    h = mesh.bounds[1][2] - mesh.bounds[0][2]
    return ["side", "top"] if h >= 0.075 else ["top", "side"]


SIDE_MIN_Z = 0.058
TOP_PALM_CLEAR = 0.038
CENTERS = {"side": CENTER.copy(), "top": np.array([0.14, 0.06, 0.0])}   # tuned in simulation (see README)


def plan_grasps(si: SceneInfo, mesh, obj_xy, kinds=None, psis=None, pre_dist=0.12, max_n=3, need_ok=True,
                centers=None):
    """Return feasible grasps (best first) by trying approach yaws and both families."""
    obj_xy = np.asarray(obj_xy, float)
    centers = centers or CENTERS
    lo, hi = mesh.bounds
    out, tried = [], 0
    psis = psis if psis is not None else [s * a for a in range(0, 120, 10) for s in ((1,) if a == 0 else (1, -1))]
    best_miss = None
    for kind in (kinds or kinds_for(mesh)):
        for psi in psis:
            R = rotation(kind, psi)
            # object axis point: horizontal centre, vertical middle
            cz = (lo[2] + hi[2]) / 2
            target_centre = np.array([obj_xy[0], obj_xy[1], cz])
            if kind == "side":
                # the lower finger sits 4 cm below the wrist axis: keep it clear of the table
                target_centre[2] = max(cz, SIDE_MIN_Z)
            else:
                # tall object from above: pinch its upper part so the palm clears the top (palm is 6 cm above the pinch)
                target_centre[2] = max(cz, hi[2] - TOP_PALM_CLEAR)
            wrist = target_centre - R @ np.asarray(centers[kind])
            pre = wrist - R @ np.array([0.0, 0.10, 0.0])
            if kind == "top":
                pre = wrist + np.array([0, 0, 0.10])
            q_g, ok_g, ep, er = solve_ik(si, wrist, R)
            q_p, ok_p, ep2, _ = solve_ik(si, pre, R, q0=q_g)
            tried += 1
            err = max(ep, ep2) * 1000
            if best_miss is None or err < best_miss:
                best_miss = err
            if not (ok_g and ok_p):
                continue
            w = width_in_plane(mesh, R, kind)
            out.append(Grasp(kind, psi, R, wrist, q_g, pre, q_p, err, w))
            if len(out) >= max_n:
                return out, best_miss
    return out, best_miss
