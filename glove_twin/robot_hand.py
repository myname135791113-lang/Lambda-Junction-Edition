"""Simulated multi-finger robot hand that grasps an object from its object card.

The robot hand reuses the Rokoko-hand geometry (same proportions as the human who mapped the
object), mounted on three slide joints. Finger flexion joints are torque motors; DIP follows PIP
through an equality coupling (like tendon-driven robot hands); abduction joints hold the preshape.
A feedback loop regulates the squeeze (half the summed normal force on the object) to the card's
grip force. Outcome: lifted, dropped (squeeze too low for the real weight/friction) or crushed
(squeeze above what the real object takes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .explore_sim import object_geom_xml
from .hand import ALL_JOINTS, FINGERS, PRESHAPES, hand_body, hand_couplings, hand_defaults
from .objects import RefObject

FLEX_MOTORS = [f"{f}_{j}" for f in FINGERS for j in ("mcp_flex", "pip")] + ["thumb_cmc_flex", "thumb_mcp", "thumb_ip"]
HOLD_JOINTS = [f"{f}_mcp_abd" for f in FINGERS] + ["thumb_cmc_abd"]
LIFT_M = 0.10


def _hand_frame(obj: RefObject, approach_az: float):
    """Palm facing the object from the side, palm width along the object's long (z) axis."""
    u = np.array([np.cos(approach_az), np.sin(approach_az), 0.0])  # from object toward the hand
    z_h, y_h = u, np.array([0, 0, 1.0])
    x_h = np.cross(y_h, z_h)
    return np.column_stack([x_h, y_h, z_h]), u


def robot_scene_xml(obj: RefObject, grasp: dict, approach_az: float = 0.0) -> tuple[str, np.ndarray]:
    R, u = _hand_frame(obj, approach_az)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.ravel())
    pre = PRESHAPES.get(grasp.get("preshape", "open"), PRESHAPES["open"])
    zc = obj.extents()[2] / 2
    asset, geom = object_geom_xml(
        obj, pos="0 0 0",
        extra=(f'mass="{obj.mass_kg}" friction="{obj.friction} 0.01 0.001" contype="1" conaffinity="3" '
               f'condim="{6 if obj.shape == "sphere" else 4}"'))
    # slide axes are given in the (rotated) forearm frame, so express the world axes there
    ax = {k: R.T @ v for k, v in zip("xyz", np.eye(3))}
    slides = "".join(f'<joint name="slide_{k}" type="slide" axis="{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" damping="40"/>'
                     for k, v in ax.items())
    hand = hand_body(pose=pre, touch_pads=True, tap_pad=False, mocap=False, root_joints=slides,
                     root_quat=" ".join(f"{q:.6f}" for q in quat))
    motors = "\n".join(f'<motor name="m_{j}" joint="{j}" ctrlrange="0 1.2" ctrllimited="true"/>' for j in FLEX_MOTORS)
    holds = "\n".join(f'<position name="h_{j}" joint="{j}" kp="3" ctrlrange="-1 1"/>' for j in HOLD_JOINTS)
    xml = f"""
<mujoco model="robot_grasp_{obj.key}">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/></visual>
  <default>{hand_defaults(collide=True)}</default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.90 0.86 0.81" rgb2="1.0 0.98 0.95" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.96 0.93 0.89" rgb2="0.89 0.85 0.80" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
    {asset}
  </asset>
  <worldbody>
    <light pos="0.4 -0.4 1.0" dir="-0.4 0.4 -1.0" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.4 0.4 0.8" dir="0.4 -0.4 -0.8" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="grid" contype="1" conaffinity="1" friction="0.8"/>
    <body name="object" pos="0 0 {zc + 0.0005}">
      <freejoint/>
      {geom}
    </body>
    {hand}
  </worldbody>
  <equality>{hand_couplings()}</equality>
  <actuator>
    <position name="p_x" joint="slide_x" kp="3000" ctrlrange="-1 1"/>
    <position name="p_y" joint="slide_y" kp="3000" ctrlrange="-1 1"/>
    <position name="p_z" joint="slide_z" kp="3000" ctrlrange="-1 1"/>
    {motors}
    {holds}
  </actuator>
</mujoco>"""
    return xml, u


@dataclass
class RobotGraspResult:
    target_squeeze_n: float
    peak_squeeze_n: float
    lift_m: float
    outcome: str
    t: np.ndarray
    squeeze_n: np.ndarray
    height_m: np.ndarray
    frames: list = field(default_factory=list)


def robot_grasp(obj: RefObject, grasp: dict, target_squeeze_n: float, *, render: bool = False,
                size=(400, 300), approach_az: float = 0.0) -> RobotGraspResult:
    xml, u = robot_scene_xml(obj, grasp, approach_az)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    obj_body, obj_geom = m.body("object").id, m.geom("object").id
    hand_bodies = [b for b in range(m.nbody) if b not in (0, obj_body)]
    m.body_gravcomp[hand_bodies] = 1.0                       # robot links are gravity-compensated
    for j in ALL_JOINTS:                                     # robot joints: no human spring, light damping
        jid = m.joint(j).id
        m.jnt_stiffness[jid] = 0.0
        m.dof_damping[m.jnt_dofadr[jid]] = 0.01
        m.dof_armature[m.jnt_dofadr[jid]] = 2e-5
    hand_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in hand_bodies}

    # place the hand: palm pads just off the object's side, at mid-height
    mujoco.mj_forward(m, d)
    palm = np.mean([d.geom_xpos[m.geom(f"palm_pad_{i}").id] for i in range(4)], axis=0)
    zc = obj.extents()[2] / 2
    reach = obj.extents()[0] / 2 if obj.shape != "box" else (obj.dims["x"] / 2 if abs(u[0]) > abs(u[1]) else obj.dims["y"] / 2)
    target = np.array([0, 0, zc]) + u * (reach + 0.009)
    offset = target - palm
    act = {m.actuator(i).name: i for i in range(m.nu)}
    for i, ax in enumerate("xyz"):
        d.qpos[m.joint(f"slide_{ax}").qposadr[0]] = offset[i]
        d.ctrl[act[f"p_{ax}"]] = offset[i]
    pre = PRESHAPES.get(grasp.get("preshape", "open"), PRESHAPES["open"])
    for j in HOLD_JOINTS:
        d.ctrl[act[f"h_{j}"]] = pre.get(j, 0.0)
    for j, q in pre.items():
        d.qpos[m.joint(j).qposadr[0]] = q
    mujoco.mj_forward(m, d)

    renderer = cam = None
    frames: list = []
    if render:
        renderer = mujoco.Renderer(m, size[1], size[0])
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, zc + 0.04]
        cam.distance = 0.45 + 0.6 * max(obj.extents())
        cam.azimuth, cam.elevation = np.degrees(approach_az) + 115.0, -15.0

    tau = 0.02
    s_f = 0.0                                                # squeeze, low-passed over ~30 ms
    alpha = m.opt.timestep / 0.03
    f6 = np.zeros(6)
    ts, sq, hs = [], [], []
    crushed = False
    duration = 3.0
    for step in range(int(duration / m.opt.timestep)):
        t = step * m.opt.timestep
        s = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == obj_geom and c.geom2 in hand_geoms) or (c.geom2 == obj_geom and c.geom1 in hand_geoms):
                mujoco.mj_contactForce(m, d, i, f6)
                s += abs(f6[0])
        s *= 0.5                                             # squeeze = half the summed normal force
        s_f += alpha * (s - s_f)                             # crushing needs sustained force
        if t > 0.15:                                         # squeeze regulator (integral)
            tau = float(np.clip(tau + 0.002 * (target_squeeze_n - s_f), 0.0, 1.2))
        for j in FLEX_MOTORS:
            d.ctrl[act[f"m_{j}"]] = tau * (1.4 if j.startswith("thumb") else 1.0)
        if t > 1.0:                                          # lift
            d.ctrl[act["p_z"]] = offset[2] + LIFT_M * min(1.0, (t - 1.0) / 1.0)
        mujoco.mj_step(m, d)
        if s_f > obj.max_grip_n and not crushed:
            crushed = True
            m.geom_rgba[obj_geom] = [0.82, 0.23, 0.23, 1.0]
        if step % 5 == 0:
            ts.append(t)
            sq.append(s_f)
            hs.append(float(d.xpos[obj_body][2]) - (zc + 0.0005))
        if renderer is not None and step % 40 == 0:
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())
    if renderer is not None:
        renderer.close()
    lift = hs[-1]
    outcome = "crushed" if crushed else ("lifted" if lift > 0.6 * LIFT_M else "dropped")
    return RobotGraspResult(target_squeeze_n, float(np.max(sq)), lift, outcome,
                            np.array(ts), np.array(sq), np.array(hs), frames)
