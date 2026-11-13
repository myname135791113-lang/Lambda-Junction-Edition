"""Simulated exploration routine: a person wearing the Rokoko glove feels an object.

Routine (same for every object, 30-60 s for a person; the object is clamped down):
  1-2. two side enclosure grasps at different angles and heights
  3.   a top grasp: palm on the top (end cap / top face), fingers curling down the sides
  4-6. thumb-to-index caliper pinches across the two horizontal directions and the height

Each grasp is simulated kinematically, the way grasp planners "auto-grasp": the hand is placed
with the palm on the object, then every finger joint closes (proximal first) until a segment
would penetrate the object, located to 0.05 mm by bisection. That avoids unstable finger dynamics
and gives the hand configuration a person would hold.

Honesty rule: what leaves this module is only what the glove would record: joint angles (with
noise) and pressure-pad readings. The wearer's hand is drawn at a slightly different size than the
reconstruction assumes. Object and hand poses are never exported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .hand import ALL_JOINTS, FLEX_CHAINS, PAD_NAMES, PRESHAPES, hand_body, hand_defaults
from .objects import RefObject
from .touch import Frame

JOINT_NOISE_DEG = 1.0      # Rokoko rotational accuracy is quoted at about +-1 deg
SCALE_SD = 0.02            # wearer's bone lengths vs. the size-M defaults
CONTACT_TOL = 0.0012       # m: a pad this close counts as pressed (fingertip pulp compresses 1-3 mm)
SOFT_TISSUE = 0.003        # m: other finger parts may sink this far in (soft fingers conform)

OBJ_RGBA = {"cylinder": "0.80 0.15 0.20 1", "box": "0.95 0.95 0.92 1", "sphere": "0.70 0.52 0.32 1"}


def object_geom_xml(obj: RefObject, name: str = "object", pos: str = "0 0 0", extra: str = "") -> tuple[str, str]:
    """(asset_xml, geom_xml) for an upright object centred at `pos`."""
    d = obj.dims
    rgba = OBJ_RGBA[obj.shape]
    if obj.shape == "sphere":
        return "", f'<geom name="{name}" type="sphere" size="{d["radius"]}" pos="{pos}" rgba="{rgba}" {extra}/>'
    if obj.shape == "cylinder":
        return "", (f'<geom name="{name}" type="cylinder" size="{d["radius"]} {d["height"] / 2}" pos="{pos}" '
                    f'rgba="{rgba}" {extra}/>')
    hx, hy, hz = d["x"] / 2, d["y"] / 2, d["z"] / 2
    if obj.sloped_top_m > 0:  # Tetra Brik Edge: the front top edge is lower than the back one
        lo = hz - obj.sloped_top_m
        verts = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
                 (-hx, -hy, hz), (hx, -hy, lo), (hx, hy, lo), (-hx, hy, hz)]
        v = " ".join(f"{a:.5f} {b:.5f} {c:.5f}" for a, b, c in verts)
        return (f'<mesh name="{name}_mesh" vertex="{v}"/>',
                f'<geom name="{name}" type="mesh" mesh="{name}_mesh" pos="{pos}" rgba="{rgba}" {extra}/>')
    return "", f'<geom name="{name}" type="box" size="{hx} {hy} {hz}" pos="{pos}" rgba="{rgba}" {extra}/>'


def explore_scene_xml(obj: RefObject, scale: float = 1.0) -> str:
    asset, geom = object_geom_xml(obj)
    return f"""
<mujoco model="explore_{obj.key}">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/></visual>
  <default>{hand_defaults()}</default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.90 0.86 0.81" rgb2="1.0 0.98 0.95" width="256" height="256"/>
    {asset}
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.3 0.3 0.6" dir="0.3 -0.3 -0.6" diffuse="0.35 0.35 0.35" castshadow="false"/>
    {geom}
    {hand_body(pose={}, touch_pads=True, tap_pad=False, scale=scale)}
  </worldbody>
</mujoco>"""


@dataclass
class GraspRecord:
    """Simulation-side truth for one grasp (for tests and the report; never fed to the fitter)."""
    kind: str
    pads_on: list
    hand_pos: np.ndarray
    hand_rot: np.ndarray
    qpos: np.ndarray = field(repr=False, default=None)


_PINCH_TABLES: dict = {}


def _pinch_table(ex: "Explorer") -> list:
    """Facing thumb-index pinch poses at different openings: [(span_m, joints), ...]."""
    key = round(ex.scale, 6)
    if key in _PINCH_TABLES:
        return _PINCH_TABLES[key]
    curl = {k: v for k, v in PRESHAPES["pinch"].items() if not k.startswith(("thumb", "index"))}
    table = []
    for abd in np.linspace(0.3, 0.9, 4):
        for cflex in np.linspace(-0.3, 0.9, 5):
            for tmcp in (0.0, 0.4):
                for imcp in np.linspace(-0.17, 1.2, 6):
                    for ipip in (0.0, 0.4, 0.8):
                        q = dict(curl, thumb_cmc_abd=abd, thumb_cmc_flex=cflex, thumb_mcp=tmcp, thumb_ip=0.2,
                                 index_mcp_flex=imcp, index_pip=ipip, index_dip=0.7 * ipip)
                        ex.set_joints(q)
                        ex.set_hand(np.eye(3), np.zeros(3))
                        gi, gt = ex.pad_ids["index_distal_pad"], ex.pad_ids["thumb_distal_pad"]
                        zi = ex.data.geom_xmat[gi].reshape(3, 3)[:, 2]
                        zt = ex.data.geom_xmat[gt].reshape(3, 3)[:, 2]
                        ip, tp = ex.local_point("index_distal_pad"), ex.local_point("thumb_distal_pad")
                        a = ip - tp
                        span = np.linalg.norm(a)
                        a /= span
                        if zi @ a - zt @ a > 1.4 and (ip + tp)[2] < 0:   # pads face each other, palmar side
                            table.append((float(span), q))
    table.sort(key=lambda t: t[0])
    _PINCH_TABLES[key] = table
    return table


class Explorer:
    def __init__(self, obj: RefObject, scale: float = 1.0):
        self.obj = obj
        self.scale = scale
        self.model = mujoco.MjModel.from_xml_string(explore_scene_xml(obj, scale))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.obj_geom = m.geom("object").id
        self.qadr = {j: m.joint(j).qposadr[0] for j in ALL_JOINTS}
        self.limits = {j: m.jnt_range[m.joint(j).id] for j in ALL_JOINTS}
        self.pad_ids = {p: m.geom(p).id for p in PAD_NAMES}
        self.pad_set = set(self.pad_ids.values())
        # geoms moved by each joint = geoms in the subtree of the joint's body
        self.subtree_geoms = {}
        for j in ALL_JOINTS:
            root = m.jnt_bodyid[m.joint(j).id]
            bodies = {root}
            for b in range(m.nbody):
                p = b
                while p > 0:
                    if p == root:
                        bodies.add(b)
                        break
                    p = m.body_parentid[p]
            self.subtree_geoms[j] = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]
        self.palm_geoms = [g for g in range(m.ngeom)
                           if m.body(m.geom_bodyid[g]).name in ("wrist", "forearm") and g != self.obj_geom]
        self.fromto = np.zeros(6)

    # ------------------------------------------------------------ kinematics helpers
    def set_joints(self, q: dict) -> None:
        self.data.qpos[:] = 0
        for j, v in q.items():
            self.data.qpos[self.qadr[j]] = v
        mujoco.mj_kinematics(self.model, self.data)

    def set_hand(self, R: np.ndarray, p: np.ndarray) -> None:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.ravel())
        self.data.mocap_quat[0] = quat
        self.data.mocap_pos[0] = p
        mujoco.mj_kinematics(self.model, self.data)

    def dist(self, geoms) -> float:
        best = 1.0
        for g in geoms:
            d = mujoco.mj_geomDistance(self.model, self.data, self.obj_geom, g, 0.05, self.fromto)
            best = min(best, d)
        return best

    def local_point(self, geom_name: str, offset_dir: str = "palmar") -> np.ndarray:
        """Point on a geom in the hand (forearm) frame, after mj_kinematics with the hand at identity."""
        g = self.model.geom(geom_name).id
        z = self.data.geom_xmat[g].reshape(3, 3)[:, 2]
        return self.data.geom_xpos[g] - self.model.geom_size[g][0] * z

    # ------------------------------------------------------------ closing
    def _blocked(self, geoms) -> bool:
        """Pads may not enter the object; other finger parts may sink in by SOFT_TISSUE."""
        pads = [g for g in geoms if g in self.pad_set]
        rest = [g for g in geoms if g not in self.pad_set]
        return (bool(pads) and self.dist(pads) < 0) or (bool(rest) and self.dist(rest) < -SOFT_TISSUE)

    def close_chain(self, chain: list[str], step: float = 0.02) -> None:
        for j in chain:
            geoms = self.subtree_geoms[j]
            lo_lim, hi_lim = self.limits[j]
            a = self.data.qpos[self.qadr[j]]
            if self._blocked(geoms):      # already touching at the preshape
                continue
            while a < hi_lim:
                b = min(a + step, hi_lim)
                self.data.qpos[self.qadr[j]] = b
                mujoco.mj_kinematics(self.model, self.data)
                if self._blocked(geoms):
                    for _ in range(9):    # bisect to ~0.04 mrad (~0.05 mm at the fingertip)
                        mid = (a + b) / 2
                        self.data.qpos[self.qadr[j]] = mid
                        mujoco.mj_kinematics(self.model, self.data)
                        if self._blocked(geoms):
                            b = mid
                        else:
                            a = mid
                    self.data.qpos[self.qadr[j]] = a
                    mujoco.mj_kinematics(self.model, self.data)
                    break
                a = b

    def pressed_pads(self) -> list[str]:
        return [p for p, g in self.pad_ids.items() if self.dist([g]) < CONTACT_TOL]

    def record(self, grasp_id: int, kind: str, rng, t: float) -> tuple[Frame, list[str]]:
        on = self.pressed_pads()
        fsr = {p: (float(rng.uniform(0.6, 3.0)) if p in on else float(abs(rng.normal(0, 0.03))))
               for p in PAD_NAMES}
        joints = {j: float(self.data.qpos[self.qadr[j]] + np.radians(rng.normal(0, JOINT_NOISE_DEG)))
                  for j in ALL_JOINTS}
        return Frame(grasp_id, kind, joints, fsr, t), on

    # ------------------------------------------------------------ placements
    def _support(self, u: np.ndarray) -> float:
        """Distance from the object centre to its surface along unit direction u."""
        d = self.obj.dims
        if self.obj.shape == "sphere":
            return d["radius"]
        if self.obj.shape == "cylinder":
            horiz = np.hypot(u[0], u[1])
            if horiz < 1e-9:
                return d["height"] / 2
            return min(d["radius"] / horiz, d["height"] / 2 / max(abs(u[2]), 1e-9))
        half = np.array([d["x"], d["y"], d["z"]]) / 2
        return float(np.min(half / np.maximum(np.abs(u), 1e-9)))

    def place_palm(self, R: np.ndarray, center: np.ndarray, u: np.ndarray, preshape: dict) -> None:
        """Hand rotation R (hand->world), palm pressed on the surface point centre + support(u) * u."""
        self.set_joints(preshape)
        self.set_hand(np.eye(3), np.zeros(3))
        palm_local = np.mean([self.local_point(f"palm_pad_{i}") for i in range(4)], axis=0)
        target = center + (self._support(u) + 0.0003) * u
        self.set_hand(R, target - R @ palm_local)
        # the palm decides the distance; back off only if the palm itself penetrates
        for _ in range(60):
            if self.dist(self.palm_geoms) >= 0:
                break
            self.set_hand(R, self.data.mocap_pos[0] + 0.001 * u)
        # fingers that poke into the object are lifted away (opened toward their lower limits)
        for chain in FLEX_CHAINS.values():
            geoms = self.subtree_geoms[chain[0]]
            for _ in range(40):
                if self.dist(geoms) >= 0:
                    break
                for j in chain:
                    lo = self.limits[j][0]
                    self.data.qpos[self.qadr[j]] = max(lo, self.data.qpos[self.qadr[j]] - 0.05)
                mujoco.mj_kinematics(self.model, self.data)

    def enclosure(self, u: np.ndarray, axis: np.ndarray, z_off: float, preshape: dict) -> None:
        z_h = u / np.linalg.norm(u)                           # dorsal side away from the object
        y_h = axis - (axis @ z_h) * z_h
        y_h /= np.linalg.norm(y_h)
        x_h = np.cross(y_h, z_h)
        R = np.column_stack([x_h, y_h, z_h])
        self.place_palm(R, np.array([0, 0, z_off]), z_h, preshape)
        for chain in ("index", "middle", "ring", "little", "thumb"):
            self.close_chain(FLEX_CHAINS[chain])

    def _chord_ok(self, e: np.ndarray, o: np.ndarray, margin: float = 0.003) -> bool:
        """Does a pinch along e, shifted sideways by o, still measure the full extent?"""
        d = self.obj.dims
        if self.obj.shape == "sphere":
            return np.linalg.norm(o) < 0.1 * d["radius"]
        if self.obj.shape == "cylinder":
            r, h = d["radius"], d["height"]
            if abs(e[2]) > 0.9:                                   # height pinch: anywhere on the caps
                return np.hypot(o[0], o[1]) < r - margin
            side = np.cross(e, [0, 0, 1.0])
            return abs(o[2]) < h / 2 - margin and abs(o @ side) < 0.15 * r
        half = np.array([d["x"], d["y"], d["z"]]) / 2
        free = np.abs(e) < 0.5
        return bool(np.all(np.abs(o[free]) < half[free] - margin))

    def _slide_directions(self, e: np.ndarray) -> list[np.ndarray]:
        """Directions the pinch may move along while still spanning the full extent along e."""
        if self.obj.shape == "sphere":
            return []
        if self.obj.shape == "cylinder" and abs(e[2]) < 0.9:
            return [np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]
        free = [ax for ax in range(3) if abs(e[ax]) < 0.5]
        out = []
        for ax in free:
            for sgn in (1.0, -1.0):
                v = np.zeros(3)
                v[ax] = sgn
                out.append(v)
        if self.obj.shape == "box" and len(free) == 2:      # toward the corners: more room
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    v = np.zeros(3)
                    v[free[0]], v[free[1]] = s1, s2
                    out.append(v / np.linalg.norm(v))
        return out

    def pinch(self, e: np.ndarray, rng) -> bool:
        """Thumb and index pads on opposite sides of the object along direction e.

        Like a person, the pinch is made near an edge: the hand rolls around e and slides sideways
        (only in directions that keep the full extent between the pads) until it is clear of the
        object, then index and thumb close.
        """
        e = e / np.linalg.norm(e)
        need = 2 * self._support(e)
        wider = [(sp, q) for sp, q in _pinch_table(self) if sp > need + 0.006]
        if not wider:
            return False                                          # wider than the open pinch
        hand_geoms = [g for g in range(self.model.ngeom) if g != self.obj_geom]
        Ke = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
        for span0, pre in wider[:10]:                             # open just wider than the object
            self.set_joints(pre)
            self.set_hand(np.eye(3), np.zeros(3))
            ip, tp = self.local_point("index_distal_pad"), self.local_point("thumb_distal_pad")
            a, mid = ip - tp, (ip + tp) / 2
            a /= np.linalg.norm(a)
            palm = np.mean([self.local_point(f"palm_pad_{i}") for i in range(4)], axis=0)
            w_loc = palm - mid
            w_loc -= (w_loc @ a) * a                              # from the pinch toward the hand
            w_loc /= np.linalg.norm(w_loc)
            v, c = np.cross(a, e), a @ e
            if np.linalg.norm(v) < 1e-9:
                R0 = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
            else:
                K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R0 = np.eye(3) + K + K @ K * ((1 - c) / np.linalg.norm(v) ** 2)
            w0 = R0 @ w_loc
            w0 -= (w0 @ e) * e
            w0 /= np.linalg.norm(w0)
            for dvec in rng.permutation(self._slide_directions(e)):
                # roll about e so that "toward the hand" points along the allowed slide direction
                roll = np.arctan2(np.cross(w0, dvec) @ e, w0 @ dvec)
                R = (np.eye(3) + np.sin(roll) * Ke + (1 - np.cos(roll)) * Ke @ Ke) @ R0
                w = R @ w_loc
                for slide in np.arange(0.0, 0.09, 0.002):
                    if not self._chord_ok(e, w * slide):
                        break
                    self.set_joints(pre)
                    self.set_hand(R, w * slide - R @ mid)        # pinch midpoint sits at w * slide
                    if not self._blocked(hand_geoms):
                        self.close_chain(["index_mcp_flex", "index_pip", "index_dip"])
                        self.close_chain(["thumb_cmc_flex", "thumb_mcp", "thumb_ip"])
                        on = self.pressed_pads()
                        if any(p.startswith("index") for p in on) and any(p.startswith("thumb") for p in on):
                            return True
                        break
        return False


class Session:
    """One person exploring one object: the standard routine plus follow-up touches on request."""

    def __init__(self, obj: RefObject, rng: np.random.Generator, *, scale: float | None = None,
                 render: bool = False, size=(480, 360)):
        self.obj, self.rng = obj, rng
        self.scale = float(rng.normal(1.0, SCALE_SD)) if scale is None else scale
        self.ex = Explorer(obj, self.scale)
        self.frames: list[Frame] = []
        self.records: list[GraspRecord] = []
        self.images: list = []
        self.t = 0.0
        self.pre = PRESHAPES["sphere"] if obj.shape == "sphere" else PRESHAPES["open"]
        self.renderer = self.cam = None
        if render:
            self.renderer = mujoco.Renderer(self.ex.model, size[1], size[0])
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = [0, 0, 0]
            self.cam.distance = 0.42
            self.cam.azimuth, self.cam.elevation = 135.0, -20.0

    def _snap(self, kind: str) -> None:
        ex = self.ex
        fr, on = ex.record(len(self.records), kind, self.rng, self.t)
        self.t += float(self.rng.uniform(4, 8))
        self.frames.append(fr)
        self.records.append(GraspRecord(kind, on, ex.data.mocap_pos[0].copy(),
                                        ex.data.mocap_quat[0].copy(), ex.data.qpos.copy()))
        if self.renderer is not None:
            self.renderer.update_scene(ex.data, self.cam)
            self.images.append((kind, on, self.renderer.render().copy()))

    def side_enclosure(self, az: float) -> None:
        """Palm on the side, fingers wrapping around the long axis."""
        obj, rng = self.obj, self.rng
        u = np.array([np.cos(az), np.sin(az), 0.0])
        if obj.shape == "box":                                # palm flat on a side face
            u = (np.array([np.sign(np.cos(az)), 0, 0]) if abs(np.cos(az)) > abs(np.sin(az))
                 else np.array([0, np.sign(np.sin(az)), 0]))
        if obj.shape == "sphere":
            u = u + np.array([0, 0, rng.uniform(-0.4, 0.4)])
        z_off = 0.0 if obj.shape == "sphere" else rng.uniform(-0.25, 0.25) * obj.extents()[2]
        self.ex.enclosure(u, np.array([0, 0, 1.0]) * rng.choice([-1, 1]), z_off, self.pre)
        self._snap("enclosure")

    def top_grasp(self) -> None:
        az = self.rng.uniform(0, 2 * np.pi)
        self.ex.enclosure(np.array([0, 0, 1.0]), np.array([np.cos(az), np.sin(az), 0]), 0.0, self.pre)
        self._snap("top")

    def pinch(self, e: np.ndarray, attempts: int = 3) -> bool:
        if self.obj.shape == "sphere":
            e = self.rng.normal(size=3)
            e /= np.linalg.norm(e)
        for _ in range(attempts):                             # a person simply tries again
            if self.ex.pinch(e, self.rng):
                self._snap("caliper")
                return True
        return False

    def routine(self) -> None:
        az0 = self.rng.uniform(0, 2 * np.pi)
        self.side_enclosure(az0)                              # 1-2: side enclosures
        self.side_enclosure(az0 + self.rng.uniform(1.6, 2.6))
        self.top_grasp()                                      # 3: palm on the top
        for e in np.eye(3):                                   # 4-6: pinches across x, y, height
            self.pinch(e)

    def follow_up(self, reasons: list[str]) -> None:
        """'Touch a few more spots': one more enclosure, then pinches (height first if missing)."""
        self.side_enclosure(self.rng.uniform(0, 2 * np.pi))
        axes = [np.array([0, 0, 1.0])] if any("height" in r or "along z" in r for r in reasons) else []
        for e in axes + list(np.eye(3)):
            self.pinch(e)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def explore(obj: RefObject, rng: np.random.Generator, *, scale: float | None = None,
            render: bool = False, size=(480, 360)):
    """Run the standard routine. Returns (frames, records, images)."""
    s = Session(obj, rng, scale=scale, render=render, size=size)
    s.routine()
    s.close()
    return s.frames, s.records, s.images
