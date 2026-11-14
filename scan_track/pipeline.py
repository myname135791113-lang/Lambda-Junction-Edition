"""Unified stateful pipeline orchestrating stream -> gestures -> 3D hull -> catalog -> relocation -> Unitree G1."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from glove_twin.objects import RefObject

from .calibration import WorkspaceCalibration
from .catalog_matcher import (
    CatalogMatchResult,
    build_matched_object_card,
    load_catalog,
    match_scan_to_catalog,
)
from .cloud_hull import ScanSummary, TactileCloudBuilder
from .gestures import GestureController, GestureEvent
from .relocation import ObjectPoseEstimate, RelocationTracker, make_se3
from .rokoko_stream import (
    DualHandFrame,
    HandPose,
    RokokoUDPReceiver,
    SyntheticExplorationStream,
)
from .unitree_g1_bridge import (
    build_unitree_g1_pick_plan,
    dispatch_unitree_g1_plan,
    verify_in_mujoco,
)


class ScanTrackSession:
    """Thread-safe controller for live hardware or synthetic 3D tactile scanning & tracking."""

    def __init__(
        self,
        *,
        source: str = "synthetic",
        mode: str = "planted",
        rokoko_port: int = 14043,
        synthetic_object: str = "redbull",
        catalog_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
        sessions_root: str | Path = "sessions",
        g1_udp: tuple[str, int] | None = None,
        g1_arm: str = "auto",
        dump_first_packet: bool = False,
    ):
        self.source_kind = source
        self.mode = mode
        self.g1_udp = g1_udp
        self.g1_arm = g1_arm
        self._lock = threading.Lock()

        self.catalog: dict[str, RefObject] = load_catalog(catalog_path)
        self.cal = (
            WorkspaceCalibration.load(calibration_path)
            if (calibration_path and Path(calibration_path).exists())
            else WorkspaceCalibration()
        )
        self.gestures = GestureController()
        self.cloud = TactileCloudBuilder(voxel_m=0.004, z_table=self.cal.z_table)
        self.relocation = RelocationTracker(z_table=self.cal.z_table)

        self.sessions_root = Path(sessions_root)
        self.session_dir = self.sessions_root / time.strftime(f"%Y%m%d_%H%M%S_{source}")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.rokoko: RokokoUDPReceiver | None = None
        self.synthetic: SyntheticExplorationStream | None = None
        if source == "rokoko":
            self.rokoko = RokokoUDPReceiver(
                port=rokoko_port, mode=mode, dump_first=dump_first_packet
            )
        else:
            self.synthetic = SyntheticExplorationStream(object_key=synthetic_object)

        self._last_frame_ws: DualHandFrame | None = None
        self._last_gesture_ev: GestureEvent | None = None
        self._last_scan: ScanSummary = self.cloud.summarize()
        self._last_match: CatalogMatchResult = match_scan_to_catalog(self._last_scan, self.catalog)
        self._locked_key: str | None = None
        self._object_card: dict[str, Any] | None = None
        self._mujoco_result: dict[str, Any] | None = None
        self._g1_plan: dict[str, Any] | None = None
        self._event_log: list[str] = ["Session initialized. Ready to scan on flat surface."]
        self._prev_synth_t: float = 0.0
        self._fps_ema: float = 0.0
        self._last_tick_t: float = time.time()

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self._event_log.append(f"[{stamp}] {msg}")
        if len(self._event_log) > 25:
            self._event_log = self._event_log[-25:]

    def _transform_hand_to_ws(self, h: HandPose | None) -> HandPose | None:
        if h is None:
            return None
        w_pos = self.cal.raw_to_workspace_point(h.wrist_pos)
        w_rot = self.cal.raw_to_workspace_rot(h.wrist_rot)
        tips_b = {k: self.cal.raw_to_workspace_point(v) for k, v in h.tips_bone.items()}
        tips_p = {k: self.cal.raw_to_workspace_point(v) for k, v in h.tips_pad.items()}
        nrms = {k: self.cal.raw_to_workspace_rot(v.reshape(3, 1)).ravel() for k, v in h.palmar_normals.items()}
        jnts = {k: self.cal.raw_to_workspace_point(v) for k, v in h.joints.items()}
        return HandPose(
            side=h.side,
            wrist_pos=w_pos,
            wrist_rot=w_rot,
            tips_bone=tips_b,
            tips_pad=tips_p,
            palmar_normals=nrms,
            joints=jnts,
            palm_center=self.cal.raw_to_workspace_point(h.palm_center),
            palm_normal=self.cal.raw_to_workspace_rot(h.palm_normal.reshape(3, 1)).ravel(),
        )

    def tick(self, dt: float | None = None) -> dict[str, Any]:
        """Advance one processing step and return the live state snapshot."""
        with self._lock:
            now = time.time()
            wall_dt = max(now - self._last_tick_t, 1e-4)
            self._last_tick_t = now
            inst_fps = 1.0 / wall_dt
            self._fps_ema = inst_fps if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst_fps)

            if self.synthetic is not None:
                raw_frame = self.synthetic.step(dt=dt)
                # Detect wrap-around of the 18s synthetic demo loop -> reset cleanly for continuous play
                cycle_pos = (raw_frame.timestamp - self.synthetic._t0) % 18.0 if self.synthetic.auto_cycle else 0.0
                if self.synthetic.auto_cycle and cycle_pos < self._prev_synth_t - 5.0:
                    self._reset_unlocked()
                    self._log(f"Auto-restarted synthetic demo cycle ({self.synthetic.object_key}).")
                self._prev_synth_t = cycle_pos
            else:
                assert self.rokoko is not None
                raw_frame = self.rokoko.latest()
                if raw_frame is None:
                    return self._snapshot_unlocked()

            frame_ws = DualHandFrame(
                timestamp=raw_frame.timestamp,
                right=self._transform_hand_to_ws(raw_frame.right),
                left=self._transform_hand_to_ws(raw_frame.left),
                mode=raw_frame.mode,
                seq=raw_frame.seq,
            )
            self._last_frame_ws = frame_ws

            ev = self.gestures.update(frame_ws.left, raw_frame.timestamp)
            self._last_gesture_ev = ev

            right = frame_ws.right
            if right is not None:
                if self.relocation.phase == "scanning":
                    if ev.stream_index:
                        self.cloud.add_points(right.tips_pad["index"])
                    if ev.snapshot_5:
                        n_new = self.cloud.add_points(right.pad_matrix())
                        self._log(f"5-finger snapshot captured (+{n_new} new voxels).")
                    if ev.caliper_pinch:
                        dist = float(np.linalg.norm(right.tips_pad["thumb"] - right.tips_pad["index"]))
                        self.cloud.add_caliper_width(dist)
                        self.cloud.add_points(np.vstack([right.tips_pad["thumb"], right.tips_pad["index"]]))
                        self._log(f"Caliper pinch recorded: {dist * 1000.0:.1f} mm.")

                    self._last_scan = self.cloud.summarize()
                    self._last_match = match_scan_to_catalog(self._last_scan, self.catalog)
                    if self._last_scan.n_voxels >= 3:
                        self.relocation.set_scan_pose(
                            self._last_scan.center_xy,
                            self._last_scan.height_m,
                            self._last_scan.yaw_rad,
                        )

                    if ev.phase_toggle and self._last_scan.n_voxels >= 5:
                        self._enter_carrying_unlocked(right)

                elif self.relocation.phase == "carrying":
                    T_ws_hand = make_se3(right.wrist_rot, right.wrist_pos)
                    self.relocation.update_carry(T_ws_hand)
                    if ev.phase_toggle:
                        self._finalize_placement_unlocked(right)

            return self._snapshot_unlocked()

    def _enter_carrying_unlocked(self, right: HandPose) -> None:
        self._locked_key = self._last_match.best_key or next(iter(self.catalog))
        ref_obj = self.catalog[self._locked_key]
        H_init = float(ref_obj.extents()[2]) if self._last_scan.height_m < 0.02 else self._last_scan.height_m
        T_ws_hand = make_se3(right.wrist_rot, right.wrist_pos)
        self.relocation.lock_grasp(
            T_ws_hand,
            fallback_xy=right.palm_center[:2],
            height_m=H_init,
            yaw_rad=self._last_scan.yaw_rad,
        )
        conf = self._last_match.probabilities.get(self._locked_key, 0.0) * 100.0
        self._log(f"LOCK_GRASP: Locked onto {ref_obj.label} ({conf:.0f}%). Pick up & move freely!")

    def _finalize_placement_unlocked(self, right: HandPose) -> None:
        key = self._locked_key or self._last_match.best_key or next(iter(self.catalog))
        ref_obj = self.catalog[key]
        T_ws_hand = make_se3(right.wrist_rot, right.wrist_pos)
        placed = self.relocation.place_on_table(T_ws_hand, right.pad_matrix(), ref_obj)

        z_bot = placed.position_m[2] - 0.5 * placed.resting_height_m
        robot_pose = self.cal.workspace_to_robot_pose(
            np.array([placed.position_m[0], placed.position_m[1], z_bot]),
            placed.yaw_rad,
        )
        self._object_card = build_matched_object_card(
            ref_obj,
            self._last_scan,
            self._last_match,
            final_pose_ws=placed.to_dict(),
            final_pose_robot=robot_pose,
        )
        self._mujoco_result = verify_in_mujoco(ref_obj, self._object_card)
        self._g1_plan = build_unitree_g1_pick_plan(
            ref_obj,
            self._object_card,
            placed,
            self.cal,
            arm=self.g1_arm,
            mujoco_check=self._mujoco_result,
        )

        # Save cloud.ply, object_card.json, and unitree_g1_pick.json atomically
        self.cloud.export_ply(str(self.session_dir / "cloud.ply"))
        card_tmp = self.session_dir / "object_card.json.tmp"
        card_tmp.write_text(json.dumps(self._object_card, indent=2), encoding="utf-8")
        os.replace(card_tmp, self.session_dir / "object_card.json")
        dispatch_unitree_g1_plan(
            self._g1_plan,
            self.session_dir / "unitree_g1_pick.json",
            udp_target=self.g1_udp,
        )
        mj_tag = self._mujoco_result["outcome"].upper()
        self._log(
            f"PLACED at ws=({placed.position_m[0]*100:.1f}, {placed.position_m[1]*100:.1f}) cm | "
            f"MuJoCo: {mj_tag} | Unitree G1 ({self._g1_plan['arm']} arm) plan ready!"
        )

    def _reset_unlocked(self) -> None:
        self.gestures.reset()
        self.cloud.reset(z_table=self.cal.z_table)
        self.relocation.reset(z_table=self.cal.z_table)
        self._last_scan = self.cloud.summarize()
        self._last_match = match_scan_to_catalog(self._last_scan, self.catalog)
        self._locked_key = None
        self._object_card = None
        self._mujoco_result = None
        self._g1_plan = None

    def handle_action(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle interactive commands from Web UI buttons or CLI shortcuts."""
        payload = payload or {}
        with self._lock:
            if action == "reset":
                self._reset_unlocked()
                if self.synthetic is not None:
                    self.synthetic.set_object(self.synthetic.object_key)
                self._log("Session reset. Ready for a fresh scan.")

            elif action == "calibrate_table":
                if self._last_frame_ws and self._last_frame_ws.right is not None:
                    # Use raw sensor frame if Rokoko is active, else current hand pads
                    raw = self.rokoko.latest() if self.rokoko is not None else self._last_frame_ws
                    if raw and raw.right is not None:
                        pts = np.vstack([raw.right.pad_matrix(), raw.right.palm_center])
                        self.cal = WorkspaceCalibration.from_table_points(pts)
                        self.cal.save(self.session_dir / "calibration.json")
                        self._reset_unlocked()
                        self._log("Table plane calibrated from flat right hand (z_table = 0.0 m)!")

            elif action in ("trace_toggle", "snapshot", "caliper", "phase"):
                # Allow immediate phase advancement even if fewer than 5 voxels when clicked manually
                if action == "phase" and self._last_frame_ws and self._last_frame_ws.right is not None:
                    if self.relocation.phase == "scanning":
                        self._enter_carrying_unlocked(self._last_frame_ws.right)
                    elif self.relocation.phase == "carrying":
                        self._finalize_placement_unlocked(self._last_frame_ws.right)
                else:
                    self.gestures.queue_command(action)

            elif action == "set_mode":
                new_mode = str(payload.get("mode", "planted"))
                self.mode = new_mode
                if self.rokoko is not None:
                    self.rokoko.set_mode(new_mode)
                self._log(f"Switched tracking mode to: {new_mode.upper()}.")

            elif action == "set_synthetic_object" and self.synthetic is not None:
                obj_key = str(payload.get("object_key", "redbull"))
                if obj_key in self.catalog:
                    self.synthetic.set_object(obj_key)
                    self._reset_unlocked()
                    self._log(f"Switched synthetic target object to: {self.catalog[obj_key].label}.")

            elif action == "set_auto_cycle" and self.synthetic is not None:
                self.synthetic.auto_cycle = bool(payload.get("enabled", True))
                if not self.synthetic.auto_cycle:
                    self.synthetic.manual_gesture = "idle"
                else:
                    self.synthetic.manual_gesture = None
                self._log(f"Synthetic auto-choreography: {'ON' if self.synthetic.auto_cycle else 'MANUAL'}.")

            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        pts = self.cloud.points()
        hull = self._last_scan.hull
        pose = self.relocation._current_pose
        matched_key = self._locked_key or self._last_match.best_key
        matched_obj = self.catalog.get(matched_key) if matched_key else None

        hands_payload: dict[str, Any] = {"right": None, "left": None}
        if self._last_frame_ws is not None:
            for side, h in (("right", self._last_frame_ws.right), ("left", self._last_frame_ws.left)):
                if h is not None:
                    hands_payload[side] = {
                        "wrist": np.round(h.wrist_pos, 4).tolist(),
                        "bones": h.bone_segments(),
                        "pads": {k: np.round(v, 4).tolist() for k, v in h.tips_pad.items()},
                    }

        hull_payload = None
        if hull is not None:
            hull_payload = {
                "vertices": np.round(hull.vertices, 4).tolist(),
                "faces": hull.faces.tolist(),
                "volume_cm3": round(hull.volume_m3 * 1e6, 1),
                "area_cm2": round(hull.area_m2 * 1e4, 1),
            }

        tracked_mesh_info = None
        if pose is not None and matched_obj is not None:
            ext = matched_obj.extents()
            tracked_mesh_info = {
                "key": matched_obj.key,
                "label": matched_obj.label,
                "shape": matched_obj.shape,
                "extents_m": [round(float(v), 4) for v in ext],
                "pose": pose.to_dict(),
            }

        return {
            "source": self.source_kind,
            "mode": self.mode,
            "fps": round(self._fps_ema, 1),
            "phase": self.relocation.phase,
            "session_dir": str(self.session_dir),
            "rokoko_status": None if self.rokoko is None else self.rokoko.status(),
            "synthetic_truth": None if self.synthetic is None else self.synthetic.ground_truth(),
            "calibration": {
                "z_table": self.cal.z_table,
                "calibrated_table": self.cal.calibrated_table,
                "calibrated_robot": self.cal.calibrated_robot,
                "alignment_rms_mm": self.cal.alignment_rms_mm,
            },
            "gesture": (
                {"sign": "idle", "badge": "WAITING FOR STREAM", "stream_index": False, "pinch_dist_mm": 0.0}
                if self._last_gesture_ev is None
                else self.gestures.describe_state(self._last_gesture_ev)
            ),
            "hands": hands_payload,
            "cloud": {
                "n_raw": self._last_scan.n_raw,
                "n_voxels": self._last_scan.n_voxels,
                "points": np.round(pts, 4).tolist(),
                "height_mm": round(self._last_scan.height_m * 1000.0, 1),
                "length_mm": round(self._last_scan.length_m * 1000.0, 1),
                "width_mm": round(self._last_scan.width_m * 1000.0, 1),
                "calipers_mm": [round(c * 1000.0, 1) for c in self._last_scan.calipers_m],
                "primitive_weights": self._last_scan.primitive_fit.get("weights", {}),
            },
            "hull": hull_payload,
            "catalog": {
                "items": {
                    k: {
                        "label": o.label,
                        "shape": o.shape,
                        "extents_mm": [round(float(v) * 1000.0, 1) for v in o.extents()],
                        "mass_g": round(o.mass_kg * 1000.0, 1),
                        "max_grip_n": o.max_grip_n,
                    }
                    for k, o in self.catalog.items()
                },
                "best_key": matched_key,
                "confident": self._last_match.confident,
                "probabilities": self._last_match.probabilities,
                "advice": self._last_match.advice,
            },
            "tracked_object": tracked_mesh_info,
            "object_card": self._object_card,
            "mujoco_result": self._mujoco_result,
            "g1_plan": self._g1_plan,
            "events": list(self._event_log[-10:]),
        }

    def close(self) -> None:
        if self.rokoko is not None:
            self.rokoko.close()
