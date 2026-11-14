"""Left-hand gesture recognition & debounced trigger state machine.

Why use the left hand as the measurement controller:
  1. Rokoko finger curl angles live purely in the left wrist's local frame (zero room-drift,
     works identically with Smartgloves-only or with Coil Pro).
  2. Making a gesture on the left hand never shakes or tenses the right (scanning) hand.
  3. Separating 1-finger continuous tracing (`👌` OK-Pinch) from 5-finger snapshots (`✊` Fist)
     solves the missing-FSR problem: you never accidentally add floating non-touching fingers.

Supported Left-Hand Signs (scale-normalized to the wearer's palm length):
  - `idle`  (🖐️ Open / relaxed hand)
  - `pinch` (👌 Thumb + Index pinched, Middle/Ring/Pinky extended) -> continuous Right-Index stream
  - `fist`  (✊ All 4 fingers curled into palm, rising edge)       -> 5-finger Right-Hand snapshot
  - `peace` (✌️ Index + Middle extended, Ring + Pinky curled)      -> step relocation (`LOCK_GRASP` -> `PLACED`)
  - `caliper` (🤏 Thumb + Pinky pinched, rising edge)              -> record Right Thumb-Index caliper width
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rokoko_stream import HandPose


@dataclass
class GestureEvent:
    """Evaluated gesture output for one frame."""
    raw_sign: str                     # "idle" | "pinch" | "fist" | "peace" | "caliper" | "no_left_hand"
    active_sign: str                  # debounced current sign
    stream_index: bool = False        # True every frame while OK-pinch (or manual trace toggle) is held
    snapshot_5: bool = False          # True on the single rising-edge frame of a Fist trigger
    caliper_pinch: bool = False       # True on the single rising-edge frame of a Caliper trigger
    phase_toggle: bool = False        # True on the single rising-edge frame of a Peace trigger
    pinch_dist_mm: float = 0.0
    finger_extensions: dict[str, float] | None = None


def classify_left_hand_sign(hand: HandPose | None) -> tuple[str, float, dict[str, float]]:
    """Classify static left-hand posture from scale-normalized 3D tip-to-wrist geometry."""
    if hand is None:
        return "no_left_hand", 0.0, {}

    # Palm scale (~68 mm typical wrist-to-middle-MCP; clamp to sane adult range so bad packets don't divide by 0)
    mid_mcp = hand.joints.get("middle_mcp")
    palm_len = float(np.linalg.norm(mid_mcp - hand.wrist_pos)) if mid_mcp is not None else 0.068
    palm_len = float(np.clip(palm_len, 0.045, 0.095))
    scale = palm_len / 0.068  # ~1.0 for size-M adult hand

    pinch_idx_m = float(np.linalg.norm(hand.tips_bone["thumb"] - hand.tips_bone["index"]))
    pinch_pnk_m = float(np.linalg.norm(hand.tips_bone["thumb"] - hand.tips_bone["pinky"]))

    # Forward finger extension along the hand's forward axis (+col 0 of wrist_rot) relative to palm_len:
    #   - Straight extended finger: tip is ~2.1 - 2.5x palm_len ahead of wrist (~1.1 - 1.4x ahead of MCP)
    #   - Curled finger in fist/peace: tip curls back to palm, so forward projection is ~0.4 - 1.15x palm_len!
    fwd_axis = hand.wrist_rot[:, 0]
    ext: dict[str, float] = {}
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        fwd_proj = float(np.dot(hand.tips_bone[f] - hand.wrist_pos, fwd_axis)) / palm_len
        ext[f] = round(fwd_proj, 3)

    is_curled = {f: ext[f] < 1.45 for f in ("index", "middle", "ring", "pinky")}
    is_straight = {f: ext[f] > 1.65 for f in ("index", "middle", "ring", "pinky")}

    # 1. Caliper trigger: thumb + pinky tips touching while index/middle stay open
    if pinch_pnk_m < 0.028 * scale and (is_straight["index"] or is_straight["middle"]):
        return "caliper", round(pinch_idx_m * 1000.0, 1), ext

    # 2. 👌 OK-Pinch: thumb + index touching (< 28 mm) AND at least 2 of (middle, ring, pinky) extended!
    #    Requiring extended outer fingers prevents any confusion with a closed fist.
    outer_open = sum(1 for f in ("middle", "ring", "pinky") if not is_curled[f])
    if pinch_idx_m < 0.028 * scale and outer_open >= 2:
        return "pinch", round(pinch_idx_m * 1000.0, 1), ext

    # 3. ✊ Closed Fist: all 4 non-thumb fingers curled inward
    if all(is_curled[f] for f in ("index", "middle", "ring", "pinky")):
        return "fist", round(pinch_idx_m * 1000.0, 1), ext

    # 4. ✌️ Peace Sign: index & middle straight, ring & pinky curled
    if is_straight["index"] and is_straight["middle"] and is_curled["ring"] and is_curled["pinky"]:
        return "peace", round(pinch_idx_m * 1000.0, 1), ext

    return "idle", round(pinch_idx_m * 1000.0, 1), ext


class GestureController:
    """Debounces raw left-hand classifications and merges keyboard/UI button overrides."""

    def __init__(self, *, hold_sec: float = 0.12, cooldown_sec: float = 0.65):
        self.hold_sec = hold_sec
        self.cooldown_sec = cooldown_sec
        self._candidate: str = "idle"
        self._candidate_since: float = 0.0
        self._latched_discrete: str = "idle"
        self._last_fire_time: dict[str, float] = {}

        # Manual overrides (for single-glove operation, CLI keys, or web UI buttons)
        self.manual_trace_enabled: bool = False
        self._queued_snapshot_5: bool = False
        self._queued_caliper: bool = False
        self._queued_phase_toggle: bool = False

    def queue_command(self, cmd: str) -> None:
        """Inject a manual trigger (`trace_toggle`, `trace_on`, `trace_off`, `snapshot`, `caliper`, `phase`)."""
        if cmd == "trace_toggle":
            self.manual_trace_enabled = not self.manual_trace_enabled
        elif cmd == "trace_on":
            self.manual_trace_enabled = True
        elif cmd == "trace_off":
            self.manual_trace_enabled = False
        elif cmd == "snapshot":
            self._queued_snapshot_5 = True
        elif cmd == "caliper":
            self._queued_caliper = True
        elif cmd == "phase":
            self._queued_phase_toggle = True

    def reset(self) -> None:
        self._candidate = "idle"
        self._latched_discrete = "idle"
        self.manual_trace_enabled = False
        self._queued_snapshot_5 = False
        self._queued_caliper = False
        self._queued_phase_toggle = False

    def update(self, left_hand: HandPose | None, now: float) -> GestureEvent:
        raw, pinch_mm, ext = classify_left_hand_sign(left_hand)

        if raw != self._candidate:
            self._candidate = raw
            self._candidate_since = now

        held = (now - self._candidate_since) >= self.hold_sec
        active = raw if (raw == "pinch" or held) else "idle"

        if raw in ("idle", "pinch", "no_left_hand"):
            self._latched_discrete = "idle"

        fire_snap = self._queued_snapshot_5
        fire_cal = self._queued_caliper
        fire_phase = self._queued_phase_toggle
        self._queued_snapshot_5 = False
        self._queued_caliper = False
        self._queued_phase_toggle = False

        if held and active in ("fist", "peace", "caliper") and self._latched_discrete != active:
            last_t = self._last_fire_time.get(active, -1e9)
            if (now - last_t) >= self.cooldown_sec:
                self._latched_discrete = active
                self._last_fire_time[active] = now
                if active == "fist":
                    fire_snap = True
                elif active == "caliper":
                    fire_cal = True
                elif active == "peace":
                    fire_phase = True

        stream_idx = (active == "pinch") or self.manual_trace_enabled
        shown_sign = "pinch" if stream_idx and active in ("idle", "no_left_hand") else active

        return GestureEvent(
            raw_sign=raw,
            active_sign=shown_sign,
            stream_index=stream_idx,
            snapshot_5=fire_snap,
            caliper_pinch=fire_cal,
            phase_toggle=fire_phase,
            pinch_dist_mm=pinch_mm,
            finger_extensions=ext,
        )

    def describe_state(self, ev: GestureEvent) -> dict[str, Any]:
        badges = {
            "idle": "OPEN HAND (IDLE)",
            "no_left_hand": "SINGLE-GLOVE / UI MODE",
            "pinch": "OK-PINCH: STREAMING RIGHT INDEX",
            "fist": "FIST: 5-FINGER SNAPSHOT",
            "peace": "PEACE: LOCK / PLACE TOGGLE",
            "caliper": "CALIPER PINCH WIDTH",
        }
        return {
            "sign": ev.active_sign,
            "raw_sign": ev.raw_sign,
            "badge": badges.get(ev.active_sign, ev.active_sign.upper()),
            "stream_index": ev.stream_index,
            "manual_trace": self.manual_trace_enabled,
            "pinch_dist_mm": ev.pinch_dist_mm,
        }
