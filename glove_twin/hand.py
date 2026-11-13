"""MuJoCo model of a hand wearing a Rokoko Smartglove (right hand).

Structure follows the Rokoko Device SDK (sdk.rokoko.com, hand-solving guide and Smartgloves
reference):
  * Skeleton forearm -> hand (wrist) -> five fingers. The thumb has metacarpal, proximal and distal
    joints; the other fingers add a medial joint. Bodies use the OpenXR XR_EXT_hand_tracking joint
    names that the SDK's hand solver emits with --emit-openxr (OpenXR "intermediate" = Rokoko
    "medial"), plus "forearm", which the solver has but OpenXR does not.
  * Non-thumb fingers bend on hinges, with backward hyper-extension clamped to a small limit.
  * Rokoko sensors (IMU + EMF) sit on the back of the middle segment of each finger and on the
    proximal segment of the thumb. The hub sits on the back of the hand and the arm sensor on the
    forearm near the wrist. Site names match the SDK sensor frames (index_finger_sensor, hub, ...).
  * The glove streams pose at 83.33 Hz and has NO force sensing, so this project adds a
    fingertip FSR + piezo (index tip) and a microphone on the back of the wrist.

The SDK pages give no bone lengths (they are sent at runtime in `static_data`), so the lengths
below are typical adult (size M) values. Replace them with the static_data rest pose for an exact
match to a specific glove.

Frame: x = fingers forward, z = dorsal (back of the hand), y = toward the thumb. Metres, radians
(scenes using this must set <compiler angle="radian"/>).
"""
from __future__ import annotations

import numpy as np

ROKOKO_RATE_HZ = 250.0 / 3.0  # Smartgloves expected_rate_hz (83.33)

# (y offset of the MCP joint, metacarpal, proximal, intermediate, distal lengths, radius)
FINGERS = {
    "index":  (0.026, 0.068, 0.043, 0.025, 0.020, 0.0085),
    "middle": (0.006, 0.066, 0.047, 0.029, 0.021, 0.0088),
    "ring":   (-0.014, 0.061, 0.044, 0.027, 0.021, 0.0082),
    "little": (-0.032, 0.055, 0.035, 0.019, 0.018, 0.0072),
}
THUMB = {"base": (0.018, 0.024, -0.010), "euler": (0.5, 0.0, 0.75), "lengths": (0.045, 0.032, 0.027),
         "radius": 0.0098}

GLOVE_RGBA = "0.16 0.16 0.18 1"
ROKOKO_SENSOR_RGBA = "0.72 0.72 0.70 1"
PIEZO_RGBA = "0.92 0.41 0.20 1"   # added fingertip FSR + piezo
MIC_RGBA = "0.11 0.69 0.48 1"     # added microphone

# Tapping pose: index pointing, other fingers curled into a loose fist.
TAP_POSE = {
    "index_mcp_flex": 0.30, "index_pip": 0.25, "index_dip": 0.15,
    **{f"{f}_mcp_flex": 1.35 for f in ("middle", "ring", "little")},
    **{f"{f}_pip": 1.55 for f in ("middle", "ring", "little")},
    **{f"{f}_dip": 0.95 for f in ("middle", "ring", "little")},
    "thumb_cmc_flex": 0.45, "thumb_cmc_abd": 0.35, "thumb_mcp": 0.55, "thumb_ip": 0.45,
}

LIMITS = {  # radians; hyper-extension clamped to about -10 degrees as the solver does
    "mcp_flex": (-0.17, 1.57), "mcp_abd": (-0.35, 0.35), "pip": (-0.05, 1.75), "dip": (-0.17, 1.40),
    "thumb_cmc_flex": (-0.3, 0.9), "thumb_cmc_abd": (-0.2, 0.9), "thumb_mcp": (-0.17, 1.0), "thumb_ip": (-0.17, 1.4),
}
AXES = {"0 1 0": np.array([0.0, 1.0, 0.0]), "0 0 1": np.array([0.0, 0.0, 1.0])}


# --- quaternion helpers (w, x, y, z), used to bake frozen joints into body orientations
def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qaxis(axis: np.ndarray, angle: float) -> np.ndarray:
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def _qeuler(a: float, b: float, c: float) -> np.ndarray:  # intrinsic x-y-z, MuJoCo's default
    q = _qaxis(np.array([1.0, 0, 0]), a)
    q = _qmul(q, _qaxis(np.array([0, 1.0, 0]), b))
    return _qmul(q, _qaxis(np.array([0, 0, 1.0]), c))


class _Builder:
    def __init__(self, pose: dict, active: set[str] | None, stiffness: dict[str, float]):
        self.pose, self.active, self.stiffness = pose, active, stiffness

    def body(self, name: str, pos: str, joints: list[tuple[str, str, tuple[float, float]]],
             base: np.ndarray | None = None) -> tuple[str, str]:
        """Opening tag + joint XML. Joints that are not active are frozen at their pose angle.

        Frozen rotations are applied before live ones, which is exact as long as a body's frozen
        joints come first or sit at zero (true for every pose used here).
        """
        q = base if base is not None else np.array([1.0, 0, 0, 0])
        xml = []
        for jname, axis, limit in joints:
            angle = self.pose.get(jname, 0.0)
            if self.active is None or jname in self.active:
                k = self.stiffness.get(jname, 0.3)
                xml.append(f'<joint name="{jname}" type="hinge" axis="{axis}" range="{limit[0]} {limit[1]}" '
                           f'springref="{angle}" stiffness="{k}" damping="{k * 0.02:.5f}"/>')
            else:
                q = _qmul(q, _qaxis(AXES[axis], angle))
        quat = " ".join(f"{v:.6f}" for v in q)
        return f'<body name="{name}" pos="{pos}" quat="{quat}">', "".join(xml)


def _capsule(length: float, radius: float) -> str:
    return f'<geom class="glove" type="capsule" fromto="0 0 0 {length} 0 0" size="{radius}"/>'


def _finger(b: _Builder, name: str, tap_pad: bool) -> str:
    y, l_meta, l_prox, l_int, l_dist, r = FINGERS[name]
    pad = (f'<geom name="tap_pad" type="sphere" size="0.0055" pos="-0.002 0 -0.003" '
           f'rgba="{PIEZO_RGBA}" contype="1" conaffinity="1" condim="1" mass="0.002"/>'
           f'<site name="fsr_piezo" pos="-0.002 0 -0.003" size="0.004"/>') if tap_pad else ""
    prox, prox_j = b.body(f"{name}_proximal", f"{l_meta} 0 0",
                          [(f"{name}_mcp_flex", "0 1 0", LIMITS["mcp_flex"]),
                           (f"{name}_mcp_abd", "0 0 1", LIMITS["mcp_abd"])])
    inter, inter_j = b.body(f"{name}_intermediate", f"{l_prox} 0 0", [(f"{name}_pip", "0 1 0", LIMITS["pip"])])
    dist, dist_j = b.body(f"{name}_distal", f"{l_int} 0 0", [(f"{name}_dip", "0 1 0", LIMITS["dip"])])
    return f"""
      <body name="{name}_metacarpal" pos="0 {y} 0">
        {prox}{prox_j}
          {_capsule(l_prox, r)}
          {inter}{inter_j}
            {_capsule(l_int, r * 0.94)}
            <site name="{name}_finger_sensor" type="box" pos="{l_int / 2} 0 {r}" size="0.007 0.0055 0.0022" rgba="{ROKOKO_SENSOR_RGBA}"/>
            {dist}{dist_j}
              {_capsule(l_dist, r * 0.88)}
              <body name="{name}_tip" pos="{l_dist + r * 0.6} 0 0">
                <site name="{name}_tip"/>
                {pad}
              </body>
            </body>
          </body>
        </body>
      </body>"""


def _thumb(b: _Builder) -> str:
    x, y, z = THUMB["base"]
    l_meta, l_prox, l_dist = THUMB["lengths"]
    r = THUMB["radius"]
    meta, meta_j = b.body("thumb_metacarpal", f"{x} {y} {z}",
                          [("thumb_cmc_flex", "0 1 0", LIMITS["thumb_cmc_flex"]),
                           ("thumb_cmc_abd", "0 0 1", LIMITS["thumb_cmc_abd"])],
                          base=_qeuler(*THUMB["euler"]))
    prox, prox_j = b.body("thumb_proximal", f"{l_meta} 0 0", [("thumb_mcp", "0 1 0", LIMITS["thumb_mcp"])])
    dist, dist_j = b.body("thumb_distal", f"{l_prox} 0 0", [("thumb_ip", "0 1 0", LIMITS["thumb_ip"])])
    return f"""
      {meta}{meta_j}
        {_capsule(l_meta, r)}
        {prox}{prox_j}
          {_capsule(l_prox, r * 0.92)}
          <site name="thumb_finger_sensor" type="box" pos="{l_prox / 2} 0 {r}" size="0.007 0.0055 0.0022" rgba="{ROKOKO_SENSOR_RGBA}"/>
          {dist}{dist_j}
            {_capsule(l_dist, r * 0.85)}
            <body name="thumb_tip" pos="{l_dist + r * 0.6} 0 0"><site name="thumb_tip"/></body>
          </body>
        </body>
      </body>"""


def hand_defaults() -> str:
    """MJCF <default> classes the hand needs. Put inside <default>."""
    return f'<default class="glove"><geom contype="0" conaffinity="0" rgba="{GLOVE_RGBA}" density="1100"/></default>'


def hand_body(pose: dict | None = None, *, active_joints: set[str] | None = None,
              stiffness: dict[str, float] | None = None, tap_pad: bool = True) -> str:
    """The forearm + hand as one mocap-driven body tree. Insert inside <worldbody>.

    active_joints: joints that stay movable; the rest are frozen at `pose` (None = all movable).
    For tapping only the index MCP swings. A real tap is made with the finger held stiff, so the
    contact time depends on the object, not on the finger wobbling.
    """
    b = _Builder(TAP_POSE if pose is None else pose, active_joints, stiffness or {})
    fingers = "".join(_finger(b, f, tap_pad and f == "index") for f in FINGERS)
    return f"""
    <body name="forearm" mocap="true">
      <geom class="glove" type="capsule" fromto="-0.20 0 0 -0.02 0 0" size="0.026"/>
      <site name="arm_sensor" type="box" pos="-0.05 0 0.027" size="0.012 0.01 0.004" rgba="{ROKOKO_SENSOR_RGBA}"/>
      <site name="mic" type="cylinder" pos="-0.035 0.0 0.030" size="0.006 0.003" rgba="{MIC_RGBA}"/>
      <body name="wrist" pos="0 0 0">
        <geom class="glove" type="box" pos="0.036 -0.003 0" size="0.040 0.041 0.0125"/>
        <geom class="glove" type="capsule" fromto="0.0 -0.03 0 0.0 0.03 0" size="0.016"/>
        <site name="palm" pos="0.033 0.006 0"/>
        <site name="hub" type="cylinder" pos="0.030 0.0 0.0135" size="0.014 0.003" rgba="{ROKOKO_SENSOR_RGBA}"/>
        {_thumb(b)}
        {fingers}
      </body>
    </body>"""


# Bodies corresponding to the 26 OpenXR hand joints (+ forearm). Palm is a site, as in the SDK
# solver, which synthesises it at the middle metacarpal's centre.
OPENXR_JOINTS = ["palm", "wrist", "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip"] + [
    f"{f}_{seg}" for f in FINGERS for seg in ("metacarpal", "proximal", "intermediate", "distal", "tip")
]
