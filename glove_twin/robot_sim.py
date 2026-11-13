"""MuJoCo simulations: the Rokoko-glove hand taps objects, then a robot gripper grasps them.

1. tap_object():   the hand (hand.py) swings its index finger onto the object. MuJoCo computes
                   the fingertip contact force at 16 kHz. The glove sensors (FSR, piezo, mic) are
                   synthesised from that force, and the Rokoko pose stream is sampled at 83.33 Hz.
2. grasp_object(): a parallel-jaw robot gripper squeezes and lifts the "real" object (ground-truth
                   mass, friction, fragility) with a grip force planned from a twin. Outcome:
                   lifted, dropped (force too low) or crushed (force above what the object takes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .hand import ROKOKO_RATE_HZ, TAP_POSE, hand_body, hand_defaults
from .materials import Material, get_material
from .simulate import MIC_GAIN, PIEZO_GAIN, synthesize_sensors
from .tap import TapEvent

G = 9.81
TAP_FS = 16_000
TAP_PRE = 64
TAP_N = 1024
CONTACT_SCALE = 1.0     # tuned so simulated contact times match the material table
CONTACT_DAMPING = 0.12
HAND_GAIN = 1.0 / 25.0   # glove amplifier gain for the hand sim (its taps are harder)
MATERIAL_RGBA = {
    "steel": "0.62 0.65 0.70 1", "aluminum": "0.80 0.82 0.85 1", "glass": "0.55 0.80 0.90 0.45",
    "ceramic": "0.95 0.94 0.90 1", "hard_plastic": "0.20 0.45 0.85 1", "wood": "0.66 0.47 0.28 1",
    "cardboard": "0.72 0.58 0.40 1", "foam": "0.95 0.85 0.35 1",
}


@dataclass(frozen=True)
class SceneObject:
    """Ground truth for a 'real' object. The robot never sees these numbers directly."""
    name: str
    material: str
    dims_m: tuple[float, float, float]   # full extents; x is the gripped width
    mass_kg: float
    friction: float
    max_grip_n: float
    label: str = ""

    @property
    def volume_cm3(self) -> float:
        return float(np.prod(self.dims_m)) * 1e6


def _contact_solref(m: Material) -> str:
    # Direct contact stiffness/damping (negative solref): a lightly damped spring whose half period
    # equals a fingernail tap on the real material, so the finger rebounds after that long.
    half_period = np.hypot(m.contact_ms, 0.3) / 1000.0 * CONTACT_SCALE
    k = (np.pi / half_period) ** 2
    b = 2.0 * CONTACT_DAMPING * np.sqrt(k)
    return f"{-k:.1f} {-b:.2f}"


# ---------------------------------------------------------------- tapping with the Rokoko hand

def tap_scene_xml(obj: SceneObject) -> str:
    m = get_material(obj.material)
    hx, hy, hz = (d / 2 for d in obj.dims_m)
    return f"""
<mujoco model="rokoko_tap_{obj.name}">
  <compiler angle="radian"/>
  <option timestep="{1.0 / TAP_FS:.8f}" gravity="0 0 0"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/></visual>
  <default>{hand_defaults()}</default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.90 0.86 0.81" rgb2="1.0 0.98 0.95" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.96 0.93 0.89" rgb2="0.89 0.85 0.80" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.3 0.3 0.6" dir="0.3 -0.3 -0.6" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="grid" contype="0" conaffinity="0"/>
    <geom name="object" type="box" pos="0 0 {hz}" size="{hx} {hy} {hz}" rgba="{MATERIAL_RGBA[obj.material]}"
          solref="{_contact_solref(m)}" condim="1" contype="1" conaffinity="1"/>
    {hand_body(active_joints={"index_mcp_flex"}, stiffness={"index_mcp_flex": 0.002})}
  </worldbody>
</mujoco>"""


@dataclass
class TapResult:
    tap: TapEvent
    force_true: np.ndarray            # MuJoCo contact force, N, at TAP_FS, aligned with tap window
    rokoko_t_ms: np.ndarray           # Rokoko frame times relative to contact onset
    rokoko_flex_deg: np.ndarray       # index flexion (MCP + PIP) as the Rokoko stream would see it
    flex_t_ms: np.ndarray             # the true, continuous flexion (for comparison)
    flex_deg: np.ndarray
    tip_speed_ms: float
    frames: list = field(default_factory=list)        # rendered RGB frames (optional), 1 ms apart
    frame_t_ms: list = field(default_factory=list)    # frame times relative to contact onset
    frame_force_n: list = field(default_factory=list)


def _place_hand(model: mujoco.MjModel, data: mujoco.MjData, obj: SceneObject, gap_m: float) -> None:
    """Put the mocap hand so the tap pad hovers gap_m above the object's top centre."""
    data.joint("index_mcp_flex").qpos[0] = TAP_POSE["index_mcp_flex"]
    pitch = 0.32  # fingers angled down toward the object
    data.mocap_quat[0] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
    data.mocap_pos[0] = [0, 0, 0.5]
    mujoco.mj_forward(model, data)
    pad = data.geom("tap_pad")
    radius = model.geom("tap_pad").size[0]
    target = np.array([0.0, 0.0, obj.dims_m[2] + radius + gap_m])
    data.mocap_pos[0] += target - pad.xpos
    mujoco.mj_forward(model, data)


def tap_object(obj: SceneObject, rng: np.random.Generator, *, with_mic: bool = True,
               render: bool = False, size: tuple[int, int] = (480, 360)) -> TapResult:
    model = mujoco.MjModel.from_xml_string(tap_scene_xml(obj))
    data = mujoco.MjData(model)
    _place_hand(model, data, obj, gap_m=rng.uniform(0.008, 0.014))

    # A light tap: fingertip speed 0.12-0.30 m/s, produced by swinging the index MCP joint
    mcp = model.joint("index_mcp_flex")
    lever = np.linalg.norm(data.geom("tap_pad").xpos - data.xanchor[mcp.id])
    tip_speed = rng.uniform(0.12, 0.30)
    data.qvel[mcp.dofadr[0]] = tip_speed / lever

    pad_id, obj_id = model.geom("tap_pad").id, model.geom("object").id
    mcp_q = model.joint("index_mcp_flex").qposadr[0]
    pip_deg = np.degrees(TAP_POSE["index_pip"])  # frozen during the tap
    f6 = np.zeros(6)

    renderer = cam = None
    frames: list = []
    if render:
        renderer = mujoco.Renderer(model, size[1], size[0])
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [-0.03, 0.0, obj.dims_m[2] + 0.025]
        cam.distance, cam.azimuth, cam.elevation = 0.30, 115.0, -14.0

    forces, flex, onset = [], [], None
    max_steps = int(0.25 * TAP_FS)
    for step in range(max_steps):
        mujoco.mj_step(model, data)
        f = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if {c.geom1, c.geom2} == {pad_id, obj_id}:
                mujoco.mj_contactForce(model, data, i, f6)
                f += abs(f6[0])
        forces.append(f)
        flex.append(np.degrees(data.qpos[mcp_q]) + pip_deg)
        if onset is None and f > 0:
            onset = step
        if render and step % 16 == 0 and (onset is None or step < onset + 0.030 * TAP_FS):
            renderer.update_scene(data, cam)
            frames.append((step, renderer.render().copy()))
        if onset is not None and step >= onset + (TAP_N - TAP_PRE) + 16:
            break
    if renderer is not None:
        renderer.close()
    if onset is None:
        raise RuntimeError(f"Finger never reached {obj.name}")

    forces = np.asarray(forces)
    flex = np.asarray(flex)
    window = forces[onset:onset + TAP_N - TAP_PRE]
    tap = synthesize_sensors(obj.material, window, TAP_FS, TAP_PRE, rng, with_mic=with_mic,
                             piezo_gain=PIEZO_GAIN * HAND_GAIN, mic_gain=MIC_GAIN * HAND_GAIN)

    # What the Rokoko glove itself reports: pose frames at 83.33 Hz with a random phase
    t_all_ms = (np.arange(len(flex)) - onset) / TAP_FS * 1000.0
    period = 1000.0 / ROKOKO_RATE_HZ
    t_frames = np.arange(t_all_ms[0] + rng.uniform(0, period), t_all_ms[-1], period)
    flex_frames = np.interp(t_frames, t_all_ms, flex)

    keep = [(st, img) for st, img in frames if st >= onset - int(0.045 * TAP_FS)]
    show = (t_all_ms >= -45) & (t_all_ms <= 60)
    return TapResult(tap=tap, force_true=np.concatenate([np.zeros(TAP_PRE), window]),
                     rokoko_t_ms=t_frames, rokoko_flex_deg=flex_frames,
                     flex_t_ms=t_all_ms[show], flex_deg=flex[show], tip_speed_ms=tip_speed,
                     frames=[img for _, img in keep],
                     frame_t_ms=[(st - onset) / TAP_FS * 1000.0 for st, _ in keep],
                     frame_force_n=[float(forces[st]) for st, _ in keep])


# ---------------------------------------------------------------- grasping with a robot gripper

def plan_grip_force(mass_kg: float, friction: float, max_grip_n: float | None = None,
                    safety: float = 1.5, gripper_max_n: float = 40.0) -> float:
    """Squeeze needed to hold the object between two pads, with a safety factor.

    Twins that know fragility cap the force at 80% of the object's safe limit.
    """
    force = safety * mass_kg * G / (2.0 * friction)
    if max_grip_n is not None:
        force = min(force, 0.8 * max_grip_n)
    return float(min(force, gripper_max_n))


def grasp_scene_xml(obj: SceneObject) -> str:
    hx, hy, hz = (d / 2 for d in obj.dims_m)
    pad_h = min(0.03, hz * 0.8)
    open_x = hx + 0.018
    return f"""
<mujoco model="grasp_{obj.name}">
  <compiler angle="radian"/>
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.90 0.86 0.81" rgb2="1.0 0.98 0.95" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.96 0.93 0.89" rgb2="0.89 0.85 0.80" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.4 1.0" dir="-0.4 0.4 -1.0" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.4 0.4 0.8" dir="0.4 -0.4 -0.8" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="grid" contype="1" conaffinity="1" friction="0.8"/>
    <body name="object" pos="0 0 {hz + 0.0005}">
      <freejoint/>
      <geom name="object" type="box" size="{hx} {hy} {hz}" mass="{obj.mass_kg}" friction="{obj.friction} 0.005 0.0001"
            rgba="{MATERIAL_RGBA[obj.material]}" contype="3" conaffinity="3"/>
    </body>
    <body name="gripper" pos="0 0 {hz}">
      <joint name="lift" type="slide" axis="0 0 1" damping="80"/>
      <geom type="box" pos="0 0 {pad_h + 0.06}" size="{open_x + 0.012} 0.022 0.012" rgba="0.25 0.27 0.30 1" mass="0.4" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 0 {pad_h + 0.12}" size="0.018 0.05" rgba="0.35 0.37 0.40 1" mass="0.1" contype="0" conaffinity="0"/>
      <body name="left_finger" pos="{-open_x} 0 0">
        <joint name="left" type="slide" axis="1 0 0" damping="6" range="0 {open_x}"/>
        <geom type="box" pos="-0.004 0 {pad_h / 2 + 0.03}" size="0.004 0.012 {pad_h / 2 + 0.03}" rgba="0.25 0.27 0.30 1" contype="0" conaffinity="0" mass="0.03"/>
        <geom name="left_pad" type="box" size="0.004 0.018 {pad_h}" friction="0.1 0.005 0.0001" rgba="0.92 0.41 0.20 1" contype="2" conaffinity="2" mass="0.02"/>
      </body>
      <body name="right_finger" pos="{open_x} 0 0">
        <joint name="right" type="slide" axis="-1 0 0" damping="6" range="0 {open_x}"/>
        <geom type="box" pos="0.004 0 {pad_h / 2 + 0.03}" size="0.004 0.012 {pad_h / 2 + 0.03}" rgba="0.25 0.27 0.30 1" contype="0" conaffinity="0" mass="0.03"/>
        <geom name="right_pad" type="box" size="0.004 0.018 {pad_h}" friction="0.1 0.005 0.0001" rgba="0.92 0.41 0.20 1" contype="2" conaffinity="2" mass="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="lift" joint="lift" kp="6000" ctrlrange="0 0.3"/>
    <motor name="left" joint="left" ctrlrange="0 60"/>
    <motor name="right" joint="right" ctrlrange="0 60"/>
  </actuator>
</mujoco>"""


@dataclass
class GraspResult:
    planned_force_n: float
    needed_force_n: float          # from the ground truth, for the report
    peak_squeeze_n: float
    lift_m: float
    outcome: str                   # "lifted" | "dropped" | "crushed"
    t: np.ndarray
    squeeze_n: np.ndarray
    height_m: np.ndarray
    frames: list = field(default_factory=list)


def grasp_object(obj: SceneObject, planned_force_n: float, *, render: bool = False,
                 size: tuple[int, int] = (400, 300), lift_target: float = 0.12) -> GraspResult:
    model = mujoco.MjModel.from_xml_string(grasp_scene_xml(obj))
    data = mujoco.MjData(model)
    obj_body = model.body("object").id
    obj_geom = model.geom("object").id
    pads = {model.geom("left_pad").id, model.geom("right_pad").id}
    f6 = np.zeros(6)

    renderer = cam = None
    frames: list = []
    if render:
        renderer = mujoco.Renderer(model, size[1], size[0])
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, obj.dims_m[2] / 2 + 0.06]
        cam.distance = 0.55 + 0.8 * max(obj.dims_m)
        cam.azimuth, cam.elevation = 90.0, -14.0

    duration, dt = 3.0, model.opt.timestep
    z0 = obj.dims_m[2] / 2 + 0.0005
    ts, squeeze, heights = [], [], []
    crushed = False
    for step in range(int(duration / dt)):
        t = step * dt
        grip = planned_force_n * min(1.0, t / 0.3)
        data.ctrl[1] = data.ctrl[2] = grip
        data.ctrl[0] = 0.0 if t < 0.7 else lift_target * min(1.0, (t - 0.7) / 1.2)
        mujoco.mj_step(model, data)

        per_pad = {p: 0.0 for p in pads}
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {c.geom1, c.geom2}
            if obj_geom in pair and pair & pads:
                mujoco.mj_contactForce(model, data, i, f6)
                per_pad[(pair & pads).pop()] += abs(f6[0])
        s = min(per_pad.values()) if all(per_pad.values()) else max(per_pad.values())
        if s > obj.max_grip_n and not crushed:
            crushed = True
            model.geom_rgba[obj_geom] = [0.82, 0.23, 0.23, 1.0]
        ts.append(t)
        squeeze.append(s)
        heights.append(float(data.xpos[obj_body][2]) - z0)
        if render and step % 20 == 0:
            renderer.update_scene(data, cam)
            frames.append(renderer.render().copy())
    if renderer is not None:
        renderer.close()

    lift = heights[-1]
    outcome = "crushed" if crushed else ("lifted" if lift > 0.6 * lift_target else "dropped")
    return GraspResult(planned_force_n=planned_force_n,
                       needed_force_n=obj.mass_kg * G / (2 * obj.friction),
                       peak_squeeze_n=float(np.max(squeeze)), lift_m=lift, outcome=outcome,
                       t=np.asarray(ts), squeeze_n=np.asarray(squeeze), height_m=np.asarray(heights),
                       frames=frames)


# ---------------------------------------------------------------- the test objects

CALIBRATION_OBJECTS = [  # solid reference blocks of known material, tapped to calibrate
    SceneObject(f"cal_{m}", m, (0.08, 0.08, 0.04), 0.0, 0.5, 100) for m in
    ("steel", "aluminum", "glass", "ceramic", "hard_plastic", "wood", "cardboard", "foam")
]

SCENARIOS = [
    SceneObject("steel_cube", "steel", (0.05, 0.05, 0.05), 0.981, 0.40, 200, "Solid steel cube, 5 cm"),
    SceneObject("glass_vase", "glass", (0.10, 0.10, 0.25), 0.450, 0.50, 15, "Thin glass vase, 25 cm tall"),
    SceneObject("cardboard_box", "cardboard", (0.12, 0.20, 0.10), 0.150, 0.55, 10, "Empty cardboard box"),
    SceneObject("wood_block", "wood", (0.08, 0.08, 0.08), 0.300, 0.45, 80, "Pine block, 8 cm"),
]
