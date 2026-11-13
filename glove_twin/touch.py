"""From glove readings to touch contacts (points + normals in the hand's hub frame).

Input per grasp, exactly what real hardware gives without a Coil Pro:
  * joint angles from the Rokoko hand solver (83.33 Hz; one stable frame per grasp)
  * pressure-pad (FSR) readings on the finger segments and palm
No hand position in the room is used anywhere.

Contact model: a pad that reads above threshold touches the object at the palmar point of the pad
(pad centre + pad radius along the segment's palmar axis), and the object's outward normal there
is the segment's dorsal axis. Bone lengths come from the size-M defaults (`scale`); the remaining
offset is calibrated once on a known object (calibrate_pad_offset).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .hand import ALL_JOINTS, PAD_NAMES, hand_body, hand_defaults
from .shape_fit import Grasp, caliper_distances, fit_model

FSR_THRESHOLD_N = 0.2


@dataclass
class Frame:
    """One stable glove frame during a grasp."""
    grasp_id: int
    kind: str                      # "enclosure" | "top" | "caliper"
    joints: dict                   # joint name -> angle (rad)
    fsr: dict                      # pad name -> force (N)
    t: float = 0.0                 # seconds since the session started


def save_frames(path: str | Path, frames: list[Frame]) -> None:
    Path(path).write_text(json.dumps([asdict(f) for f in frames], indent=1), encoding="utf-8")


def load_frames(path: str | Path) -> list[Frame]:
    return [Frame(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


@lru_cache(maxsize=4)
def _hand_model(scale: float):
    xml = (f'<mujoco><compiler angle="radian"/><default>{hand_defaults()}</default>'
           f'<worldbody>{hand_body(pose={}, touch_pads=True, tap_pad=False, scale=scale)}</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    qadr = {j: model.joint(j).qposadr[0] for j in ALL_JOINTS}
    pads = {p: model.geom(p).id for p in PAD_NAMES}
    return model, qadr, pads


def pad_contacts(joints: dict, pads_on: list[str], *, scale: float = 1.0, pad_offset: float = 0.0):
    """Hub-frame contact points and outward normals for the given pads."""
    model, qadr, pad_ids = _hand_model(round(scale, 6))
    data = mujoco.MjData(model)
    for j, q in joints.items():
        if j in qadr:
            data.qpos[qadr[j]] = q
    mujoco.mj_kinematics(model, data)
    wrist = model.body("wrist").id
    Rw, pw = data.xmat[wrist].reshape(3, 3), data.xpos[wrist]
    pts, nrm = [], []
    for name in pads_on:
        g = pad_ids[name]
        z = data.geom_xmat[g].reshape(3, 3)[:, 2]          # dorsal axis of the segment / palm
        p = data.geom_xpos[g] - (model.geom_size[g][0] + pad_offset) * z
        pts.append(Rw.T @ (p - pw))
        nrm.append(Rw.T @ z)
    return np.array(pts).reshape(-1, 3), np.array(nrm).reshape(-1, 3)


def _offset(pad_offset, kind: str) -> float:
    if isinstance(pad_offset, dict):
        return float(pad_offset.get("caliper" if kind == "caliper" else "enclosure", 0.0))
    return float(pad_offset)


def reconstruct(frames: list[Frame], *, scale: float = 1.0, pad_offset: float | dict = 0.0,
                threshold: float = FSR_THRESHOLD_N) -> list[Grasp]:
    """One Grasp per grasp_id, using the frame with the most pads pressed (the stable hold).

    pad_offset: a number, or {"enclosure": m, "caliper": m} from calibrate_pad_offset().
    """
    best: dict[int, Frame] = {}
    for f in frames:
        n_on = sum(v > threshold for v in f.fsr.values())
        if f.grasp_id not in best or n_on > sum(v > threshold for v in best[f.grasp_id].fsr.values()):
            best[f.grasp_id] = f
    grasps = []
    for gid in sorted(best):
        f = best[gid]
        on = [p for p in PAD_NAMES if f.fsr.get(p, 0.0) > threshold]
        if not on:
            continue
        pts, nrm = pad_contacts(f.joints, on, scale=scale, pad_offset=_offset(pad_offset, f.kind))
        if f.kind == "caliper":
            # the measurement is index pad vs thumb pad: take the most opposed pair
            idx = [i for i, n in enumerate(on) if n.startswith("index")]
            thb = [i for i, n in enumerate(on) if n.startswith("thumb")]
            if not idx or not thb:
                continue
            i, j = min(((i, j) for i in idx for j in thb), key=lambda ij: nrm[ij[0]] @ nrm[ij[1]])
            pts, nrm, on = pts[[i, j]], nrm[[i, j]], [on[i], on[j]]
        grasps.append(Grasp(pts, nrm, f.kind, on))
    return grasps


def calibrate_pad_offset(frames: list[Frame], true_radius: float, *, scale: float = 1.0) -> dict:
    """One-time calibration on a known cylinder (e.g. the Red Bull can, r = 26.5 mm).

    Enclosures: moving every contact inward by `offset` shrinks the fitted radius by about the
    same amount, so offset = r_fit(0) - r_true.
    Calipers: the pulp-to-pulp distance across the diameter should be 2 r, so
    offset = (d_measured - 2 r) / 2.
    The two differ because enclosures press the sides of the pads, pinches press the pulp.
    """
    grasps = reconstruct(frames, scale=scale)
    encl = [g for g in grasps if g.kind != "caliper"]
    fit = fit_model("cylinder", encl)
    out = {"enclosure": float(fit.theta[0] - true_radius), "caliper": 0.0}
    widths = [d for d in caliper_distances(grasps) if d < 3 * true_radius]   # diameter pinches only
    if widths:
        out["caliper"] = float((np.median(widths) - 2 * true_radius) / 2)
    return out
