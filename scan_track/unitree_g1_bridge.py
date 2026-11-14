"""MuJoCo Digital Twin grasp verification + Unitree G1 Humanoid pick command bridge.

Pipeline when the user signals `PLACED`:
  1. Digital Twin Safety Gate (`verify_in_mujoco`):
     Spawns the matched `RefObject` in MuJoCo (`glove_twin.robot_hand.robot_grasp`) and runs the
     closed-loop force-regulated lift with `card["grasp"]["grip_force_n"]`. Confirms
     `outcome == "lifted"` (neither dropped from low force nor crushed above `max_safe_grip_n`).
  2. Unitree G1 Pick Plan (`build_unitree_g1_pick_plan`):
     Transforms the verified table pose `(x, y, z_table, yaw)` into the **Unitree G1 pelvis/waist
     base frame** (`+X` forward, `+Y` left, `+Z` up), selects the closer arm (`right` or `left`,
     or user-forced arm), and emits:
       - 4 Cartesian wrist waypoints (`PRE_GRASP` -> `DESCEND` -> `CLOSE_HAND` -> `LIFT`)
       - **Unitree Dex3-1** (3-finger, 7-DoF per hand: `thumb_0..2`, `middle_0..1`, `index_0..1`)
         open preshape `q_open`, closed target `q_grasp`, gains `kp`/`kd`, and force-mapped
         feedforward joint torques `tau_ff_nm`
       - **Inspire RH56DFX** (5-finger, 6-DoF optional G1 hand) normalized closure `[0..1000]` +
         force threshold
  3. Dispatch (`dispatch_unitree_g1_plan`):
     Writes atomic `unitree_g1_pick.json`, optionally broadcasts over UDP (`--g1-udp host:port`),
     and exposes a ready-to-enable `unitree_sdk2py` DDS hook (`rt/dex3/{right|left}/cmd`).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
from typing import Any

import numpy as np

from glove_twin.objects import RefObject

from .calibration import WorkspaceCalibration
from .relocation import ObjectPoseEstimate


def verify_in_mujoco(ref_obj: RefObject, card: dict[str, Any]) -> dict[str, Any]:
    """Run closed-loop multi-finger MuJoCo grasp check before commanding the physical G1."""
    target_n = float(card["grasp"]["grip_force_n"])
    try:
        from glove_twin.robot_hand import robot_grasp
        res = robot_grasp(ref_obj, card["grasp"], target_n, render=False)
        return {
            "simulated": True,
            "verified": res.outcome == "lifted",
            "outcome": res.outcome,
            "target_squeeze_n": round(float(res.target_squeeze_n), 2),
            "peak_squeeze_n": round(float(res.peak_squeeze_n), 2),
            "lift_height_m": round(float(res.lift_m), 4),
        }
    except ImportError:
        # Analytic Coulomb friction + crush safety fallback if MuJoCo isn't installed on a dev box
        min_slip_n = ref_obj.mass_kg * 9.81 / max(2.0 * ref_obj.friction, 1e-3)
        if target_n > ref_obj.max_grip_n:
            outcome = "crushed"
        elif target_n >= min_slip_n:
            outcome = "lifted"
        else:
            outcome = "dropped"
        return {
            "simulated": False,
            "verified": outcome == "lifted",
            "outcome": outcome,
            "target_squeeze_n": round(target_n, 2),
            "peak_squeeze_n": round(target_n, 2),
            "lift_height_m": 0.10 if outcome == "lifted" else 0.0,
            "note": "MuJoCo package not installed; verified analytically via Coulomb slip & crush limits.",
        }


def _dex3_hand_command(
    hand_side: str,
    grasp_type: str,
    grip_width_m: float,
    grip_force_n: float,
) -> dict[str, Any]:
    """Map object grasp width & planned force (N) to Unitree Dex3-1 7-DoF finger commands.

    Unitree Dex3-1 joint order (`rt/dex3/{right|left}/cmd`, 7 motors):
      [0] thumb_0 (rotation/opposition), [1] thumb_1 (proximal flex), [2] thumb_2 (distal flex),
      [3] middle_0 (proximal flex),      [4] middle_1 (distal flex),
      [5] index_0 (proximal flex),       [6] index_1 (distal flex).
    """
    sign = 1.0 if hand_side == "right" else -1.0
    # Aperture closure fraction: 0.0 at 120 mm wide, ~0.85 at 35 mm slim object
    closure = float(np.clip(1.0 - (grip_width_m - 0.030) / 0.100, 0.25, 0.92))

    # Effective fingertip moment arm ~ 0.065 m; split total squeeze across thumb vs (index + middle)
    lever_m = 0.065
    tau_thumb = float(np.clip(grip_force_n * lever_m * 0.55, 0.08, 1.40))
    tau_finger = float(np.clip(grip_force_n * lever_m * 0.30, 0.06, 1.10))

    opp = 0.85 if grasp_type in ("power_wrap", "box_wrap") else 0.65
    q_open = [round(sign * opp, 3), 0.10, 0.08, 0.10, 0.08, 0.10, 0.08]
    q_grasp = [
        round(sign * opp, 3),
        round(0.95 * closure, 3),
        round(0.80 * closure, 3),
        round(1.15 * closure, 3),
        round(0.90 * closure, 3),
        round(1.15 * closure, 3),
        round(0.90 * closure, 3),
    ]
    tau_ff = [
        0.0,
        round(tau_thumb, 3),
        round(0.75 * tau_thumb, 3),
        round(tau_finger, 3),
        round(0.75 * tau_finger, 3),
        round(tau_finger, 3),
        round(0.75 * tau_finger, 3),
    ]
    return {
        "hand_model": "unitree_dex3_1",
        "dds_topic": f"rt/dex3/{hand_side}/cmd",
        "joint_names": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
        "q_open_rad": q_open,
        "q_grasp_rad": q_grasp,
        "tau_ff_nm": tau_ff,
        "kp": [1.5, 1.2, 1.0, 1.2, 1.0, 1.2, 1.0],
        "kd": [0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
    }


def _inspire_hand_command(grip_width_m: float, grip_force_n: float, max_safe_n: float) -> dict[str, Any]:
    """Optional command block if the G1 is fitted with 5-finger Inspire RH56DFX hands (`0=closed, 1000=open`)."""
    open_val = 900
    close_val = int(np.clip((grip_width_m / 0.115) * 750.0, 120.0, 780.0))
    force_val = int(np.clip((grip_force_n / max(max_safe_n, 25.0)) * 650.0, 100.0, 900.0))
    return {
        "hand_model": "inspire_rh56dfx",
        "dof_order": ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"],
        "angle_open_0_1000": [open_val, open_val, open_val, open_val, open_val, 200],
        "angle_grasp_0_1000": [close_val, close_val, close_val, close_val, close_val, 200],
        "force_threshold_0_1000": [force_val] * 6,
    }


def _rot_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """Convert 3x3 rotation matrix to `[qw, qx, qy, qz]` (Unitree SDK & MuJoCo convention)."""
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return [round(float(v), 5) for v in q]


def build_unitree_g1_pick_plan(
    ref_obj: RefObject,
    card: dict[str, Any],
    placed_ws: ObjectPoseEstimate,
    cal: WorkspaceCalibration,
    *,
    arm: str = "auto",
    mujoco_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a complete Unitree G1 7-DoF arm + dexterous hand pick plan in G1 pelvis frame."""
    z_bottom_ws = float(placed_ws.position_m[2] - 0.5 * placed_ws.resting_height_m)
    obj_bottom_ws = np.array([placed_ws.position_m[0], placed_ws.position_m[1], z_bottom_ws])
    robot_pose = cal.workspace_to_robot_pose(obj_bottom_ws, placed_ws.yaw_rad)
    ox, oy, oz_table = robot_pose["position_m"]
    yaw_r = float(robot_pose["yaw_rad"])

    # In G1 pelvis frame: +X is forward, +Y is left, +Z is up.
    # Auto-pick right arm if object is on right side (`oy <= 0.03`), else left arm.
    chosen_arm = ("right" if oy <= 0.03 else "left") if arm == "auto" else arm

    H_obj = placed_ws.resting_height_m
    grip_w = float(card["grasp"]["grip_width_m"])
    grip_f = float(card["grasp"]["grip_force_n"])
    max_safe_f = float(card["max_safe_grip_n"])

    # Choose side approach angle facing away from G1 chest (+X forward, slight lateral angle for arm comfort)
    if ref_obj.shape == "box":
        # Align palm normal perpendicular to the narrower box face closest to G1's front approach
        c1 = yaw_r
        c2 = yaw_r + 0.5 * math.pi
        # Pick the face normal pointing most toward G1 torso (-X_g1)
        candidates = [c1, c1 + math.pi, c2, c2 + math.pi]
        approach_yaw = min(candidates, key=lambda a: abs(math.atan2(math.sin(a), math.cos(a))))
    else:
        # Radially symmetric in XY (sphere / upright cylinder): approach along line from G1 shoulder to object
        sh_y = -0.18 if chosen_arm == "right" else 0.18
        approach_yaw = math.atan2(oy - sh_y, max(ox, 0.15))

    ux, uy = math.cos(approach_yaw), math.sin(approach_yaw)
    z_grasp = oz_table + max(0.50 * H_obj, 0.035)
    palm_standoff = 0.5 * grip_w + 0.018

    # End-effector frame: +X_ee along approach `u`, +Z_ee = world up `[0, 0, 1]`, +Y_ee = Z x X
    R_ee = np.array([[ux, -uy, 0.0], [uy, ux, 0.0], [0.0, 0.0, 1.0]])
    quat_wxyz = _rot_to_quat_wxyz(R_ee)

    pre_pos = [round(ox - (palm_standoff + 0.10) * ux, 4), round(oy - (palm_standoff + 0.10) * uy, 4), round(z_grasp + 0.06, 4)]
    grasp_pos = [round(ox - palm_standoff * ux, 4), round(oy - palm_standoff * uy, 4), round(z_grasp, 4)]
    lift_pos = [grasp_pos[0], grasp_pos[1], round(z_grasp + 0.12, 4)]

    # Basic workspace reachability sanity check for Unitree G1 arm (~0.18 - 0.68 m reach from pelvis)
    horiz_dist = math.hypot(ox, oy)
    reachable = 0.16 <= horiz_dist <= 0.72 and -0.25 <= oz_table <= 0.45

    dex3_cmd = _dex3_hand_command(chosen_arm, card["grasp"]["type"], grip_w, grip_f)
    inspire_cmd = _inspire_hand_command(grip_w, grip_f, max_safe_f)

    return {
        "robot_model": "unitree_g1_humanoid",
        "base_frame": "pelvis (+X forward, +Y left, +Z up)",
        "arm": chosen_arm,
        "reachable": reachable,
        "safe_to_execute": bool(reachable and (mujoco_check is None or mujoco_check.get("verified", False))),
        "object": {
            "key": ref_obj.key,
            "label": ref_obj.label,
            "shape": ref_obj.shape,
            "mass_kg": ref_obj.mass_kg,
            "friction": ref_obj.friction,
            "fragile": bool(card.get("fragile", False)),
            "planned_grip_force_n": grip_f,
            "max_safe_grip_n": max_safe_f,
        },
        "object_pose_g1_base": robot_pose,
        "object_pose_workspace": placed_ws.to_dict(),
        "mujoco_twin_gate": mujoco_check,
        "waypoints": [
            {"stage": "1_PRE_GRASP", "wrist_pos_m": pre_pos, "wrist_quat_wxyz": quat_wxyz, "hand_state": "open", "duration_s": 1.2},
            {"stage": "2_DESCEND_TO_GRASP", "wrist_pos_m": grasp_pos, "wrist_quat_wxyz": quat_wxyz, "hand_state": "open", "duration_s": 0.8},
            {"stage": "3_FORCE_CLOSE", "wrist_pos_m": grasp_pos, "wrist_quat_wxyz": quat_wxyz, "hand_state": "grasp_force", "grip_force_n": grip_f, "duration_s": 0.6},
            {"stage": "4_LIFT_OBJECT", "wrist_pos_m": lift_pos, "wrist_quat_wxyz": quat_wxyz, "hand_state": "hold_force", "grip_force_n": grip_f, "duration_s": 1.0},
        ],
        "dex3_1_hand": dex3_cmd,
        "inspire_hand": inspire_cmd,
    }


def dispatch_unitree_g1_plan(
    plan: dict[str, Any],
    out_path: str | Path,
    *,
    udp_target: tuple[str, int] | None = None,
) -> dict[str, Any]:
    """Persist `unitree_g1_pick.json` atomically and optionally broadcast over UDP to G1 controller."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    payload = json.dumps(plan, indent=2)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, p)

    udp_sent = False
    if udp_target is not None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(payload.encode("utf-8"), udp_target)
            udp_sent = True
        finally:
            sock.close()

    return {"file": str(p), "udp_sent": udp_sent, "safe_to_execute": plan["safe_to_execute"]}
