"""Dual-hand Rokoko Smartgloves II (+ optional Coil Pro) stream & synthetic hand generator.

Supported live packet schemas over UDP (default port 14043):
  1. Rokoko Studio "Custom Streaming" JSON v3 / v4 (raw UTF-8 JSON or LZ4-framed JSON).
     Joints appear either in `scene.actors[0].body.<jointName>` (e.g. `rightHand`,
     `rightIndexProximal`, `rightIndexMedial`, `rightIndexDistal`, `rightIndexTip`) or in
     flat/nested joint arrays.
  2. Rokoko Device SDK / OpenXR hand-solver JSON (`right_wrist` / `hub`, `index_proximal`,
     `index_intermediate`, `index_distal`, `index_tip`, etc.).

Coordinate convention output by this module (right-handed Z-up, metres):
  +X = right, +Y = forward, +Z = up.
Rokoko Studio's Unity left-handed Y-up metres `(x_u, y_u, z_u)` map cleanly to right-handed
Z-up metres via `(x_u, z_u, y_u)` (single axis swap flips handedness from LH to RH).

Palmar finger-pad offset:
  Rokoko reports skeletal joint/tip centres inside the finger. Every fingertip is shifted inward
  along its anatomical palmar normal (computed directly from the 3D bone chain so it never
  suffers from quaternion convention mismatches) by `pad_offset_m` (~7 mm default).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
import socket
import threading
import time
from typing import Any

import numpy as np

from glove_twin.objects import OBJECTS, RefObject

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
LZ4_MAGIC = b"\x04\x22\x4d\x18"
DEFAULT_PAD_OFFSET_M = 0.007  # ~7 mm from distal bone tip centre to palmar skin pulp

# Flexible token rules matching Rokoko Studio v3 camelCase, v4 Newton (Finger1..5), and OpenXR/SDK snake_case
_FINGER_ALIASES: dict[str, tuple[str, ...]] = {
    "thumb": ("thumb", "finger1"),
    "index": ("index", "finger2"),
    "middle": ("middle", "mid", "finger3"),
    "ring": ("ring", "finger4"),
    "pinky": ("little", "pinky", "finger5"),
}
_STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "mcp": ("proximal", "mcp", "metacarpal"),
    "pip": ("medial", "intermediate", "pip"),
    "dip": ("distal", "dip"),
    "tip": ("tip",),
}


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 1e-9:
        return v / n
    return np.array([0.0, 0.0, -1.0]) if fallback is None else fallback.copy()


def _orthonormal_frame(x_axis: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Build a right-handed 3x3 rotation matrix [x | y | z] with det = +1."""
    x = _unit(x_axis, np.array([0.0, 1.0, 0.0]))
    y_proj = y_hint - np.dot(y_hint, x) * x
    if np.linalg.norm(y_proj) < 1e-8:
        alt = np.array([1.0, 0.0, 0.0]) if abs(x[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y_proj = alt - np.dot(alt, x) * x
    y = _unit(y_proj, np.array([-1.0, 0.0, 0.0]))
    z = np.cross(x, y)
    return np.column_stack([x, y, _unit(z, np.array([0.0, 0.0, 1.0]))])


@dataclass
class HandPose:
    """Full 3D pose of one gloved hand in right-handed Z-up world coordinates (metres)."""
    side: str                                 # "left" | "right"
    wrist_pos: np.ndarray                     # (3,) m
    wrist_rot: np.ndarray                     # (3, 3) SO(3): col 0=fwd, col 1=across, col 2=dorsal
    tips_bone: dict[str, np.ndarray]          # finger -> (3,) m skeletal tip centre
    tips_pad: dict[str, np.ndarray]           # finger -> (3,) m palmar skin contact point
    palmar_normals: dict[str, np.ndarray]     # finger -> (3,) unit vector pointing OUT of finger pulp
    joints: dict[str, np.ndarray] = field(default_factory=dict)  # optional intermediate joints for 3D viz
    palm_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    palm_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))

    def wrist_transform(self) -> np.ndarray:
        """4x4 homogeneous transform T_world_hand."""
        T = np.eye(4)
        T[:3, :3] = self.wrist_rot
        T[:3, 3] = self.wrist_pos
        return T

    def pad_matrix(self) -> np.ndarray:
        """(5, 3) palmar contact points ordered thumb, index, middle, ring, pinky."""
        return np.vstack([self.tips_pad[f] for f in FINGER_NAMES])

    def bone_segments(self) -> list[tuple[list[float], list[float]]]:
        """Line segments [(p0, p1), ...] in metres for rendering the 3D hand skeleton."""
        segs: list[tuple[list[float], list[float]]] = []
        w = self.wrist_pos
        prev_mcp = None
        for f in FINGER_NAMES:
            chain = [w]
            for st in ("mcp", "pip", "dip"):
                key = f"{f}_{st}"
                if key in self.joints:
                    chain.append(self.joints[key])
            chain.append(self.tips_bone[f])
            for a, b in zip(chain[:-1], chain[1:]):
                segs.append((np.round(a, 4).tolist(), np.round(b, 4).tolist()))
            mcp = self.joints.get(f"{f}_mcp")
            if mcp is not None and f != "thumb":
                if prev_mcp is not None:
                    segs.append((np.round(prev_mcp, 4).tolist(), np.round(mcp, 4).tolist()))
                prev_mcp = mcp
        return segs


@dataclass
class DualHandFrame:
    """Synchronized snapshot of right and/or left hands."""
    timestamp: float
    right: HandPose | None
    left: HandPose | None
    mode: str = "coil"     # "planted" | "coil" | "synthetic"
    seq: int = 0


def build_hand_pose(
    side: str,
    wrist_pos: np.ndarray,
    tips_bone: dict[str, np.ndarray],
    joints: dict[str, np.ndarray] | None = None,
    *,
    pad_offset_m: float = DEFAULT_PAD_OFFSET_M,
    wrist_quat_xyzw_unity: tuple[float, float, float, float] | None = None,
) -> HandPose:
    """Compute anatomical wrist frame, palm normal, and palmar skin-pad offsets from 3D joints."""
    joints = dict(joints or {})
    idx_mcp = joints.get("index_mcp", tips_bone["index"])
    pnk_mcp = joints.get("pinky_mcp", tips_bone["pinky"])
    mid_mcp = joints.get("middle_mcp", 0.5 * (idx_mcp + pnk_mcp))

    v_fwd = mid_mcp - wrist_pos
    # Right hand palm-down (+Y fwd): index is at -X, pinky at +X -> idx - pnk = (-X), so fwd x across = +Z (dorsal)
    v_across = (idx_mcp - pnk_mcp) if side == "right" else (pnk_mcp - idx_mcp)

    if np.linalg.norm(v_fwd) > 1e-5 and np.linalg.norm(v_across) > 1e-5:
        R_wrist = _orthonormal_frame(v_fwd, v_across)
    elif wrist_quat_xyzw_unity is not None:
        R_wrist = _unity_quat_to_rh_zup_matrix(*wrist_quat_xyzw_unity)
    else:
        R_wrist = np.eye(3)

    dorsal = R_wrist[:, 2]
    palm_normal = -dorsal
    palm_center = 0.45 * wrist_pos + 0.55 * mid_mcp + 0.008 * palm_normal

    # Lateral axis pointing from pinky toward index for right hand (reversed for left so cross product stays palmar)
    lat_sign = 1.0 if side == "right" else -1.0
    u_lat = lat_sign * R_wrist[:, 1]

    tips_pad: dict[str, np.ndarray] = {}
    palmar_normals: dict[str, np.ndarray] = {}
    for f in FINGER_NAMES:
        tip = tips_bone[f]
        ref = joints.get(f"{f}_dip", joints.get(f"{f}_pip", joints.get(f"{f}_mcp", wrist_pos)))
        u_dist = _unit(tip - ref, R_wrist[:, 0])
        if f == "thumb":
            # Thumb opposes the fingers: blend palm normal and direction toward index/middle tips
            opp = 0.5 * (tips_bone["index"] + tips_bone["middle"]) - tip
            if np.linalg.norm(opp) > 1e-5:
                n_palmar = _unit(0.65 * _unit(opp) + 0.35 * palm_normal, palm_normal)
            else:
                n_palmar = palm_normal.copy()
        else:
            cross = np.cross(u_lat, u_dist)
            n_palmar = _unit(cross, palm_normal) if np.linalg.norm(cross) > 1e-5 else palm_normal.copy()
        palmar_normals[f] = n_palmar
        tips_pad[f] = tip + pad_offset_m * n_palmar

    return HandPose(
        side=side,
        wrist_pos=wrist_pos,
        wrist_rot=R_wrist,
        tips_bone=tips_bone,
        tips_pad=tips_pad,
        palmar_normals=palmar_normals,
        joints=joints,
        palm_center=palm_center,
        palm_normal=palm_normal,
    )


def _unity_quat_to_rh_zup_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a Unity LH Y-up quaternion (x, y, z, w) to RH Z-up 3x3 rotation via basis swap P."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    R_lh = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # swaps Y <-> Z, det(P) = -1
    return P @ R_lh @ P  # det(P) * det(R_lh) * det(P) = (-1)*(+1)*(-1) = +1


# --------------------------------------------------------------------------- Rokoko JSON packet parser

def walk_rokoko_joints(
    obj: Any, prefix: str = ""
) -> list[tuple[str, list[float], tuple[float, float, float, float] | None]]:
    """Extract `(joint_name, [x, y, z], optional_quat_xyzw)` from any Rokoko JSON v3/v4 or SDK dict."""
    found: list[tuple[str, list[float], tuple[float, float, float, float] | None]] = []
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("id")
        pos = obj.get("position")
        rot = obj.get("rotation") or obj.get("orientation")
        if pos is not None and (isinstance(name, str) or prefix):
            jname = str(name) if isinstance(name, str) else prefix
            p_xyz = _extract_xyz(pos)
            q_xyzw = _extract_xyzw(rot)
            if p_xyz is not None:
                found.append((jname, p_xyz, q_xyzw))
        for k, v in obj.items():
            if k in ("position", "rotation", "orientation"):
                continue
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            # In Rokoko Studio v4 `body: {"rightHand": {"position": {...}, "rotation": {...}}}`
            if isinstance(v, dict) and "position" in v and "name" not in v and "id" not in v:
                p_xyz = _extract_xyz(v.get("position"))
                q_xyzw = _extract_xyzw(v.get("rotation") or v.get("orientation"))
                if p_xyz is not None:
                    found.append((str(k), p_xyz, q_xyzw))
            else:
                found.extend(walk_rokoko_joints(v, child_prefix))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(walk_rokoko_joints(item, prefix))
    return found


def _extract_xyz(pos: Any) -> list[float] | None:
    try:
        if isinstance(pos, dict) and all(k in pos for k in ("x", "y", "z")):
            vals = [float(pos["x"]), float(pos["y"]), float(pos["z"])]
        elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
            vals = [float(pos[0]), float(pos[1]), float(pos[2])]
        else:
            return None
    except (TypeError, ValueError):
        return None
    return vals if all(math.isfinite(v) for v in vals) else None


def _extract_xyzw(rot: Any) -> tuple[float, float, float, float] | None:
    try:
        if isinstance(rot, dict) and all(k in rot for k in ("x", "y", "z", "w")):
            vals = (float(rot["x"]), float(rot["y"]), float(rot["z"]), float(rot["w"]))
        elif isinstance(rot, (list, tuple)) and len(rot) >= 4:
            vals = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
        else:
            return None
    except (TypeError, ValueError):
        return None
    return vals if all(math.isfinite(v) for v in vals) else None


def _joint_tokens(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return {tok.lower() for tok in re.findall(r"finger[1-5]|[A-Za-z]+|\d+", spaced, flags=re.IGNORECASE)}


def _unity_to_rh_zup(xyz: list[float]) -> np.ndarray:
    """Unity LH Y-up metres (x, y, z) -> Right-Handed Z-up metres (x, z, y)."""
    return np.array([xyz[0], xyz[2], xyz[1]], dtype=float)


def parse_rokoko_frame(
    packet: dict[str, Any],
    *,
    mode: str = "planted",
    planted_anchor_right: np.ndarray | None = None,
    pad_offset_m: float = DEFAULT_PAD_OFFSET_M,
) -> DualHandFrame:
    """Parse a decoded Rokoko JSON packet into a `DualHandFrame`.

    In `mode="planted"` (Smartgloves II without Coil Pro), the right wrist position is anchored
    at `planted_anchor_right` (default `[0.0, 0.30, 0.035]` m on the table) while preserving full
    3D wrist rotation and finger kinematics relative to the wrist!
    """
    raw_joints = walk_rokoko_joints(packet)
    hands: dict[str, HandPose | None] = {"right": None, "left": None}

    # Check if single-hand SDK stream omits "left"/"right" prefixes (defaults to right hand)
    tokenized_joints = [(name, _joint_tokens(name), xyz, quat) for name, xyz, quat in raw_joints]
    any_side_tag = any(("right" in toks or "left" in toks) for _, toks, _, _ in tokenized_joints)

    for side in ("right", "left"):
        tips_bone: dict[str, np.ndarray] = {}
        inter: dict[str, np.ndarray] = {}
        wrist_pos: np.ndarray | None = None
        wrist_quat: tuple[float, float, float, float] | None = None

        for name, toks, xyz, quat in tokenized_joints:
            if any_side_tag:
                if side not in toks:
                    continue
            elif side == "left":
                continue

            p = _unity_to_rh_zup(xyz)
            has_finger_tok = any(fk in toks for aliases in _FINGER_ALIASES.values() for fk in aliases)

            # Match wrist / hand hub (exclude finger joints)
            if (toks & {"wrist", "hand", "hub", "palm"}) and not has_finger_tok:
                if wrist_pos is None or ("wrist" in toks or "hub" in toks or name.lower().endswith("hand")):
                    wrist_pos = p
                    wrist_quat = quat
                continue

            for finger, f_keys in _FINGER_ALIASES.items():
                if not any(fk in toks for fk in f_keys):
                    continue
                for stage, s_keys in _STAGE_ALIASES.items():
                    if any(sk in toks for sk in s_keys):
                        if stage == "tip":
                            tips_bone.setdefault(finger, p)
                        elif stage == "mcp" and ("proximal" in toks or "mcp" in toks):
                            inter[f"{finger}_{stage}"] = p
                        else:
                            inter.setdefault(f"{finger}_{stage}", p)
                        break

        if len(tips_bone) < 5:
            continue

        if wrist_pos is None:
            # Estimate wrist behind MCPs if wrist node had an unexpected name
            mcps = [inter[f"{f}_mcp"] for f in FINGER_NAMES if f"{f}_mcp" in inter]
            wrist_pos = np.mean(mcps, axis=0) if mcps else np.mean(list(tips_bone.values()), axis=0)

        if mode == "planted":
            anchor = (
                planted_anchor_right
                if (side == "right" and planted_anchor_right is not None)
                else (np.array([0.0, 0.25, 0.035]) if side == "right" else np.array([-0.24, 0.22, 0.14]))
            )
            delta = anchor - wrist_pos
            wrist_pos = wrist_pos + delta
            tips_bone = {k: v + delta for k, v in tips_bone.items()}
            inter = {k: v + delta for k, v in inter.items()}

        hands[side] = build_hand_pose(
            side,
            wrist_pos,
            tips_bone,
            inter,
            pad_offset_m=pad_offset_m,
            wrist_quat_xyzw_unity=wrist_quat,
        )

    ts = float(packet.get("timestamp", time.time()))
    return DualHandFrame(timestamp=ts, right=hands["right"], left=hands["left"], mode=mode)


class RokokoUDPReceiver:
    """Non-blocking background UDP listener for Rokoko Studio / Device SDK streams."""

    def __init__(
        self,
        port: int = 14043,
        *,
        mode: str = "planted",
        pad_offset_m: float = DEFAULT_PAD_OFFSET_M,
        max_age_s: float = 1.0,
        dump_first: bool = False,
    ):
        self.port = port
        self.mode = mode
        self.pad_offset_m = pad_offset_m
        self.max_age_s = max_age_s
        self.dump_first = dump_first
        self.planted_anchor_right = np.array([0.0, 0.25, 0.035])
        self._lock = threading.Lock()
        self._latest_raw: dict[str, Any] | None = None
        self._latest_frame: DualHandFrame | None = None
        self._last_recv_mono: float = 0.0
        self._packets_received = 0
        self._last_error: str | None = None
        self._running = True

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(0.25)
        self._sock.bind(("0.0.0.0", port))
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self.mode = mode

    def _listen_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1 << 20)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                if data[:4] == LZ4_MAGIC:
                    try:
                        import lz4.frame  # type: ignore
                    except ImportError:
                        self._last_error = (
                            "Received LZ4-compressed Rokoko packet. Install `pip install lz4` "
                            "or switch Rokoko Studio Custom Streaming format to uncompressed JSON."
                        )
                        continue
                    data = lz4.frame.decompress(data)

                pkt = json.loads(data.decode("utf-8"))
                if self.dump_first:
                    self.dump_first = False
                    joints = [n for n, _, _ in walk_rokoko_joints(pkt)]
                    print("=== Rokoko Packet Received ===")
                    print(f"Detected {len(joints)} joints: {sorted(set(joints))}")
                frame = parse_rokoko_frame(
                    pkt,
                    mode=self.mode,
                    planted_anchor_right=self.planted_anchor_right,
                    pad_offset_m=self.pad_offset_m,
                )
                with self._lock:
                    self._packets_received += 1
                    frame.seq = self._packets_received
                    self._latest_raw = pkt
                    self._latest_frame = frame
                    self._last_recv_mono = time.monotonic()
                    self._last_error = None
            except Exception as exc:  # pragma: no cover - defensive against malformed UDP
                with self._lock:
                    self._last_error = str(exc)

    def latest(self, *, require_fresh: bool = False) -> DualHandFrame | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            if require_fresh and self.max_age_s > 0 and (time.monotonic() - self._last_recv_mono) > self.max_age_s:
                return None
            return self._latest_frame

    def status(self) -> dict[str, Any]:
        with self._lock:
            age_s = (time.monotonic() - self._last_recv_mono) if self._last_recv_mono > 0 else None
            stale = bool(age_s is None or (self.max_age_s > 0 and age_s > self.max_age_s))
            return {
                "port": self.port,
                "mode": self.mode,
                "packets_received": self._packets_received,
                "stale": stale,
                "frame_age_ms": None if age_s is None else round(age_s * 1000.0, 1),
                "has_right": bool(self._latest_frame and self._latest_frame.right is not None),
                "has_left": bool(self._latest_frame and self._latest_frame.left is not None),
                "last_error": self._last_error,
            }

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- Synthetic Dual-Hand Stream

class SyntheticExplorationStream:
    """Realistic animated 2-hand stream for instant offline testing & live demo rehearsal.

    Simulates a person wearing both Rokoko Smartgloves:
      1. Tracing the top, sides, and base seam of a hidden catalog object on `z = 0` while holding
         a Left-Hand OK-Pinch (`👌`), plus taking 5-finger grasp snapshots (`✊`).
      2. Grabbing the object (`✌️` LOCK_GRASP), lifting & carrying it across the table to a new
         random `(x, y, yaw)` position, setting it back on `z = 0`, and signalling `PLACED` (`✌️`).
    """

    # Finger layout in local hand frame: +X = forward along fingers, +Y = across toward thumb, +Z = dorsal
    _MCP_OFFSETS = {
        "thumb": np.array([0.025, 0.036, -0.008]),
        "index": np.array([0.068, 0.026, 0.0]),
        "middle": np.array([0.068, 0.006, 0.0]),
        "ring": np.array([0.063, -0.014, 0.0]),
        "pinky": np.array([0.056, -0.032, 0.0]),
    }
    _SEG_LENGTHS = {
        "thumb": (0.030, 0.022, 0.020),
        "index": (0.042, 0.024, 0.019),
        "middle": (0.046, 0.028, 0.020),
        "ring": (0.043, 0.026, 0.019),
        "pinky": (0.035, 0.019, 0.017),
    }

    def __init__(
        self,
        object_key: str = "redbull",
        *,
        seed: int = 7,
        noise_m: float = 0.0016,
        pad_offset_m: float = DEFAULT_PAD_OFFSET_M,
        auto_cycle: bool = True,
    ):
        self.rng = np.random.default_rng(seed)
        self.noise_m = noise_m
        self.pad_offset_m = pad_offset_m
        self.auto_cycle = auto_cycle
        self._seq = 0
        self._t0 = time.time()
        self.manual_gesture: str | None = None
        self.set_object(object_key)

    def set_object(self, object_key: str) -> None:
        self.obj: RefObject = OBJECTS[object_key]
        self.object_key = object_key
        self._t0 = time.time()
        self.initial_xy = np.array([0.04, 0.34])
        self.initial_yaw = float(self.rng.uniform(-0.35, 0.35))
        self.target_xy = np.array([0.22, 0.42])
        self.target_yaw = self.initial_yaw + 0.65  # ~37 deg rotation during carry

    def ground_truth(self) -> dict[str, Any]:
        ext = self.obj.extents()
        return {
            "object_key": self.object_key,
            "label": self.obj.label,
            "shape": self.obj.shape,
            "extents_m": [round(float(v), 4) for v in ext],
            "initial_pose": {"x": float(self.initial_xy[0]), "y": float(self.initial_xy[1]), "z": 0.0, "yaw": self.initial_yaw},
            "placed_pose": {"x": float(self.target_xy[0]), "y": float(self.target_xy[1]), "z": 0.0, "yaw": self.target_yaw},
        }

    def _surface_point_local(self, az: float, frac_h: float, top_cap: bool = False) -> np.ndarray:
        """Point on the object's outer surface in object-centred coordinates (bottom at -H/2)."""
        d = self.obj.dims
        c, s = math.cos(az), math.sin(az)
        if self.obj.shape == "sphere":
            R = d["radius"]
            z = float(np.clip(frac_h, -0.82, 0.96)) * R
            rho = math.sqrt(max(R * R - z * z, 0.0))
            return np.array([rho * c, rho * s, z])
        if self.obj.shape == "cylinder":
            R, H = d["radius"], d["height"]
            if top_cap:
                r = R * float(np.clip(abs(frac_h), 0.1, 0.92))
                return np.array([r * c, r * s, H / 2])
            return np.array([R * c, R * s, float(np.clip(frac_h, -0.48, 0.48)) * H])
        hx, hy, hz = d["x"] / 2, d["y"] / 2, d["z"] / 2
        if top_cap:
            return np.array([hx * 0.8 * c, hy * 0.8 * s, hz])
        scale = min(hx / max(abs(c), 1e-6), hy / max(abs(s), 1e-6))
        return np.array([scale * c, scale * s, float(np.clip(frac_h, -0.48, 0.48)) * 2 * hz])

    def _obj_world(self, p_local: np.ndarray, xy: np.ndarray, yaw: float, z_lift: float = 0.0) -> np.ndarray:
        H = float(self.obj.extents()[2])
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ p_local + np.array([xy[0], xy[1], H / 2 + z_lift])

    def _make_hand_from_curls(
        self,
        side: str,
        wrist_pos: np.ndarray,
        R_wrist: np.ndarray,
        curls: dict[str, float],
        *,
        pinch_index: bool = False,
        override_tips_pad: dict[str, np.ndarray] | None = None,
    ) -> HandPose:
        """Synthesize a consistent 3D articulated hand for visualization and testing."""
        lat_sign = 1.0 if side == "right" else -1.0
        joints: dict[str, np.ndarray] = {}
        tips_bone: dict[str, np.ndarray] = {}

        for f in FINGER_NAMES:
            mcp_local = self._MCP_OFFSETS[f].copy()
            mcp_local[1] *= lat_sign
            p = wrist_pos + R_wrist @ mcp_local
            joints[f"{f}_mcp"] = p

            curl = curls.get(f, 0.25)
            l1, l2, l3 = self._SEG_LENGTHS[f]
            fwd = R_wrist[:, 0]
            palmar = -R_wrist[:, 2]
            if f == "thumb":
                inward = -lat_sign * R_wrist[:, 1]
                d1 = _unit(0.75 * fwd + 0.45 * palmar * curl + 0.35 * inward * curl)
                d2 = _unit(0.55 * fwd + 0.65 * palmar * curl + 0.55 * inward * curl)
                d3 = _unit(0.30 * fwd + 0.70 * palmar * curl + 0.70 * inward * curl)
            else:
                # Cumulative flexion angles matching real human anatomy (MCP ~1.35 + PIP ~1.55 + DIP ~0.95 rad)
                a1, a2, a3 = 1.35 * curl, 2.90 * curl, 3.85 * curl
                d1 = math.cos(a1) * fwd + math.sin(a1) * palmar
                d2 = math.cos(a2) * fwd + math.sin(a2) * palmar
                d3 = math.cos(a3) * fwd + math.sin(a3) * palmar

            p1 = p + l1 * d1
            p2 = p1 + l2 * d2
            p3 = p2 + l3 * d3
            joints[f"{f}_pip"] = p1
            joints[f"{f}_dip"] = p2
            tips_bone[f] = p3

        if pinch_index:
            meet = 0.5 * (tips_bone["thumb"] + tips_bone["index"])
            tips_bone["thumb"] = meet + R_wrist @ np.array([0.0, lat_sign * 0.005, 0.0])
            tips_bone["index"] = meet - R_wrist @ np.array([0.0, lat_sign * 0.005, 0.0])

        pose = build_hand_pose(side, wrist_pos, tips_bone, joints, pad_offset_m=self.pad_offset_m)

        if override_tips_pad:
            for f, target_pad in override_tips_pad.items():
                noisy_pad = target_pad + self.rng.normal(0.0, self.noise_m, 3)
                pose.tips_pad[f] = noisy_pad
                pose.tips_bone[f] = noisy_pad - self.pad_offset_m * pose.palmar_normals[f]
        return pose

    def _left_hand_for_gesture(self, gesture: str) -> HandPose:
        wrist = np.array([-0.24, 0.28, 0.16])
        R = _orthonormal_frame(np.array([0.15, 0.98, 0.05]), np.array([0.98, -0.15, 0.0]))
        if gesture == "pinch":
            # 👌 OK Pinch: thumb+index pinched (< 15 mm), middle/ring/pinky extended straight!
            return self._make_hand_from_curls(
                "left", wrist, R, {"thumb": 0.75, "index": 0.75, "middle": 0.08, "ring": 0.10, "pinky": 0.12},
                pinch_index=True,
            )
        if gesture == "fist":
            # ✊ Closed Fist: all fingers curled tightly
            return self._make_hand_from_curls(
                "left", wrist, R, {"thumb": 0.85, "index": 1.05, "middle": 1.08, "ring": 1.08, "pinky": 1.08}
            )
        if gesture == "peace":
            # ✌️ Peace Sign: index & middle straight, ring & pinky curled
            return self._make_hand_from_curls(
                "left", wrist, R, {"thumb": 0.85, "index": 0.05, "middle": 0.05, "ring": 1.08, "pinky": 1.08}
            )
        # 🖐️ Open Idle
        return self._make_hand_from_curls(
            "left", wrist, R, {"thumb": 0.15, "index": 0.12, "middle": 0.12, "ring": 0.14, "pinky": 0.16}
        )

    def step(self, dt: float | None = None) -> DualHandFrame:
        """Advance synthetic timeline (or evaluate current wall-clock time)."""
        now = time.time() if dt is None else (self._t0 + self._seq * dt)
        elapsed = (now - self._t0) % 18.0 if self.auto_cycle else (now - self._t0)
        self._seq += 1

        H = float(self.obj.extents()[2])
        R_max = float(max(self.obj.extents()[:2])) / 2.0

        # Choreographed 18-second full demo cycle:
        #   0.0 - 5.5 s : OK-Pinch index tracing (helical sweep + top cap + base seam!)
        #   5.8 - 6.3 s : Fist #1 (5-finger side wrap snapshot)
        #   6.8 - 7.3 s : Fist #2 (rotated 60 deg 5-finger snapshot)
        #   8.2 - 8.7 s : Peace #1 -> LOCK_GRASP on object at initial_xy
        #   8.7 - 12.2 s: Smooth 6-DoF lift + carry + yaw rotation to target_xy
        #  12.4 - 12.9 s: Peace #2 -> PLACED on flat surface at target_xy
        #  13.0 - 18.0 s: Hands withdraw while robot executes
        left_gesture = self.manual_gesture or "idle"
        xy = self.initial_xy.copy()
        yaw = self.initial_yaw
        z_lift = 0.0
        override_pads: dict[str, np.ndarray] = {}
        right_curls = {"thumb": 0.35, "index": 0.20, "middle": 0.75, "ring": 0.82, "pinky": 0.85}

        if elapsed < 5.5:
            left_gesture = self.manual_gesture or "pinch"
            u = elapsed / 5.5
            if u < 0.55:
                # Helical side trace across corners/walls
                az = 4.0 * math.pi * (u / 0.55)
                frac_h = -0.44 + 0.90 * (u / 0.55)
                p_loc = self._surface_point_local(az, frac_h, top_cap=False)
            elif u < 0.78:
                # Top surface trace
                v = (u - 0.55) / 0.23
                az = 2.0 * math.pi * v
                p_loc = self._surface_point_local(az, 0.2 + 0.7 * v, top_cap=True)
            else:
                # Base-seam sweep right where object meets the table!
                v = (u - 0.78) / 0.22
                az = 2.0 * math.pi * v
                p_loc = self._surface_point_local(az, -0.47, top_cap=False)

            p_surf = self._obj_world(p_loc, xy, yaw)
            wrist_pos = p_surf + np.array([0.0, -0.13, 0.05])
            R_right = _orthonormal_frame(p_surf - wrist_pos, np.array([-1.0, 0.0, 0.0]))
            override_pads = {"index": p_surf}

        elif elapsed < 7.6:
            # Two 5-finger enclosure grasps rotated by 60 deg
            grasp_idx = 0 if elapsed < 6.5 else 1
            in_pulse = (5.8 <= elapsed <= 6.3) or (6.8 <= elapsed <= 7.3)
            left_gesture = self.manual_gesture or ("fist" if in_pulse else "idle")
            base_az = -0.5 * math.pi + grasp_idx * (math.pi / 3.0)
            approach = np.array([math.cos(base_az), math.sin(base_az), 0.0])
            obj_mid = np.array([xy[0], xy[1], 0.52 * H])
            wrist_pos = obj_mid + (R_max + 0.11) * approach + np.array([0.0, 0.0, 0.02])
            R_right = _orthonormal_frame(-approach, np.array([0.0, 0.0, 1.0]))
            right_curls = {"thumb": 0.55, "index": 0.55, "middle": 0.58, "ring": 0.60, "pinky": 0.62}
            # Place thumb on one side and 4 fingers stacked on the opposite side
            override_pads["thumb"] = self._obj_world(
                self._surface_point_local(base_az - 0.9 - yaw, 0.0), xy, yaw
            )
            for k, fname in enumerate(("index", "middle", "ring", "pinky")):
                fh = 0.30 - 0.20 * k
                override_pads[fname] = self._obj_world(
                    self._surface_point_local(base_az + 0.85 - yaw, fh), xy, yaw
                )

        elif elapsed < 13.1:
            # Grasp, lift, carry across the table, set down, signal PLACED
            if 8.2 <= elapsed <= 8.65:
                left_gesture = self.manual_gesture or "peace"
                alpha = 0.0
            elif elapsed < 8.2:
                alpha = 0.0
            elif elapsed <= 12.2:
                s = (elapsed - 8.65) / (12.2 - 8.65)
                alpha = float(np.clip(3 * s * s - 2 * s * s * s, 0.0, 1.0))
                z_lift = 0.11 * math.sin(math.pi * alpha)
            else:
                alpha = 1.0
                z_lift = 0.0
                if 12.4 <= elapsed <= 12.85:
                    left_gesture = self.manual_gesture or "peace"

            xy = (1.0 - alpha) * self.initial_xy + alpha * self.target_xy
            yaw = (1.0 - alpha) * self.initial_yaw + alpha * self.target_yaw

            # Rigid wrist-to-object relationship throughout the single grasp!
            cy, sy = math.cos(yaw), math.sin(yaw)
            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            wrist_offset_obj = np.array([0.0, -(R_max + 0.105), 0.55 * H + 0.015])
            wrist_pos = np.array([xy[0], xy[1], z_lift]) + Rz @ wrist_offset_obj
            R_right = Rz @ _orthonormal_frame(np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
            right_curls = {"thumb": 0.55, "index": 0.55, "middle": 0.58, "ring": 0.60, "pinky": 0.62}

            override_pads["thumb"] = self._obj_world(self._surface_point_local(-2.3, 0.02), xy, yaw, z_lift)
            for k, fname in enumerate(("index", "middle", "ring", "pinky")):
                override_pads[fname] = self._obj_world(
                    self._surface_point_local(-0.85, 0.28 - 0.18 * k), xy, yaw, z_lift
                )
        else:
            # Hands pulled back cleanly so Unitree G1 / MuJoCo picks up at target_xy
            wrist_pos = np.array([0.05, 0.16, 0.18])
            R_right = _orthonormal_frame(np.array([0.0, 1.0, -0.1]), np.array([-1.0, 0.0, 0.0]))
            right_curls = {"thumb": 0.18, "index": 0.15, "middle": 0.15, "ring": 0.18, "pinky": 0.20}

        right_hand = self._make_hand_from_curls(
            "right", wrist_pos, R_right, right_curls, override_tips_pad=override_pads
        )
        left_hand = self._left_hand_for_gesture(left_gesture)
        return DualHandFrame(timestamp=now, right=right_hand, left=left_hand, mode="synthetic", seq=self._seq)
