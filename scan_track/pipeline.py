"""Unified stateful pipeline orchestrating stream -> gestures -> 3D hull -> catalog -> relocation -> Unitree G1."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import math
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
from .cloud_hull import ScanSummary, TactileCloudBuilder, fit_reference_pose_xy_yaw
from .gestures import GestureController, GestureEvent
from .relocation import ObjectPoseEstimate, RelocationTracker, invert_se3, make_se3
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
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        cand = self.sessions_root / stamp
        idx = 1
        while cand.exists():
            cand = self.sessions_root / f"{stamp}_{idx}"
            idx += 1
        cand.mkdir(parents=True, exist_ok=False)
        self.session_dir = cand

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
        self._placed_dirty: bool = False
        self._placed_at_ts: float = -1e9
        self._logged_90_key: str | None = None
        self._undo_stack: list[dict[str, Any]] = []
        self._was_streaming: bool = False
        self._event_log: list[str] = ["Session initialized. Ready to scan on flat surface."]
        self._prev_synth_t: float = 0.0
        self._last_processed_seq: int = -1
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
                    self._undo_stack.clear()
                    self._log(f"Auto-restarted synthetic demo cycle ({self.synthetic.object_key}).")
                self._prev_synth_t = cycle_pos
            else:
                assert self.rokoko is not None
                raw_frame = self.rokoko.latest(require_fresh=True)
                if raw_frame is None or raw_frame.seq == self._last_processed_seq:
                    return self._snapshot_unlocked()
                self._last_processed_seq = raw_frame.seq

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
                    if ev.phase_toggle:
                        self._enter_carrying_unlocked(right)
                    else:
                        self._ingest_tactile_events_unlocked(right, ev)
                        self._refresh_scan_and_match_unlocked()
                        if self._last_scan.n_voxels >= 3:
                            xy_s, h_s, yaw_s, R_s, up_s = self._estimate_scan_pose_unlocked()
                            self.relocation.set_scan_pose(xy_s, h_s, yaw_s, rotation=R_s, upright=up_s)

                elif self.relocation.phase == "carrying":
                    # Strictly forbid adding or queueing any points while carrying the object
                    self.cloud.locked = True
                    self.gestures.clear_scan_commands()
                    ev.stream_index = False
                    ev.snapshot_5 = False
                    ev.caliper_pinch = False
                    T_ws_hand = make_se3(right.wrist_rot, right.wrist_pos)
                    self.relocation.update_carry(T_ws_hand)
                    if ev.phase_toggle:
                        self._finalize_placement_unlocked(right, placed_ts=raw_frame.timestamp)

                elif self.relocation.phase == "placed":
                    if ev.phase_toggle:
                        self._enter_carrying_unlocked(right)
                    elif (raw_frame.timestamp - self._placed_at_ts) < 0.45:
                        # Ignore transient finger curl/pinch while releasing Peace sign right after placement
                        ev.stream_index = False
                        ev.snapshot_5 = False
                        ev.caliper_pinch = False
                    else:
                        changed, discrete = self._ingest_tactile_events_unlocked(right, ev)
                        if changed or discrete:
                            prev_locked = self._locked_key
                            self._refresh_scan_and_match_unlocked()
                            self._locked_key = self._confirmed_catalog_key_unlocked()
                            ref_obj = self.catalog.get(self._locked_key) if self._locked_key else None
                            xy_s, h_s, yaw_s, R_s, up_s = self._estimate_scan_pose_unlocked(ref_override=ref_obj)
                            newly_locked = prev_locked != self._locked_key and self._locked_key is not None
                            if newly_locked:
                                self.relocation._current_pose = None
                            self.relocation.refresh_placed_from_scan(
                                xy_s,
                                h_s,
                                yaw_s,
                                n_voxels=self._last_scan.n_voxels,
                                ref_obj=ref_obj,
                                rotation=R_s if newly_locked else None,
                                upright=up_s if newly_locked else None,
                            )
                            self._placed_dirty = True
                        if self._placed_dirty and (discrete or not ev.stream_index):
                            if self.relocation._current_pose is not None:
                                self._write_placed_outputs_unlocked(self.relocation._current_pose)
                            self._placed_dirty = False
            elif self.relocation.phase == "carrying":
                self.cloud.locked = True
                self.gestures.clear_scan_commands()
                ev.stream_index = False
                ev.snapshot_5 = False
                ev.caliper_pinch = False

            return self._snapshot_unlocked()

    # ------------------------------------------------------------------ undo
    def _push_undo_unlocked(self) -> None:
        """Remember the state just before a committed action (points added, object locked/placed)."""
        self._undo_stack.append(
            {
                "cloud": copy.deepcopy(self.cloud),
                "relocation": copy.deepcopy(self.relocation),
                "locked_key": self._locked_key,
                "placed_dirty": self._placed_dirty,
                "placed_at_ts": self._placed_at_ts,
                "logged_90_key": self._logged_90_key,
            }
        )
        if len(self._undo_stack) > 60:
            self._undo_stack.pop(0)

    def _undo_unlocked(self) -> str:
        if self.relocation.phase == "carrying":
            msg = "Undo unavailable while carrying: place the object down first."
            self._log(msg)
            return msg
        # Snapshots taken in the "carrying" phase (pushed when placing) are never restored:
        # undoing a placement must go back to the state before the pick-up, so skip past them.
        idx = len(self._undo_stack) - 1
        while idx >= 0 and self._undo_stack[idx]["relocation"].phase == "carrying":
            idx -= 1
        if idx < 0:
            self._log("Undo: nothing to undo.")
            return "Nothing to undo."
        st = self._undo_stack[idx]
        del self._undo_stack[idx:]
        self.cloud = st["cloud"]
        self.relocation = st["relocation"]
        self._locked_key = st["locked_key"]
        self._placed_dirty = st["placed_dirty"]
        self._placed_at_ts = st.get("placed_at_ts", -1e9)
        self._logged_90_key = st["logged_90_key"]
        self.cloud.locked = self.relocation.phase == "carrying"
        self.gestures.clear_scan_commands()
        self._was_streaming = False
        self._refresh_scan_and_match_unlocked()
        if self.relocation.phase == "placed" and self.relocation._current_pose is not None:
            self._write_placed_outputs_unlocked(self.relocation._current_pose)
        else:
            self._object_card = self._mujoco_result = self._g1_plan = None
            for stale_name in ("object_card.json", "unitree_g1_pick.json"):
                stale_p = self.session_dir / stale_name
                if stale_p.exists():
                    stale_p.unlink()
        msg = f"Undo: restored previous state ({self.relocation.phase}, {self._last_scan.n_voxels} voxels)."
        self._log(msg)
        return msg

    def _ingest_tactile_events_unlocked(
        self, right: HandPose, ev: GestureEvent
    ) -> tuple[bool, bool]:
        """Record any active trace, 5-finger snapshot, or caliper width; return (changed, discrete)."""
        if self.relocation.phase == "carrying" or self.cloud.locked:
            return False, False
        changed = False
        discrete = False
        if (ev.snapshot_5 or ev.caliper_pinch) or (ev.stream_index and not self._was_streaming):
            self._push_undo_unlocked()   # one undo step per snapshot / caliper / trace stroke
        self._was_streaming = bool(ev.stream_index)
        if ev.stream_index:
            n_idx = self.cloud.add_points(right.tips_pad["index"])
            changed = changed or (n_idx > 0)
        if ev.snapshot_5:
            n_new = self.cloud.add_points(right.pad_matrix())
            self._log(f"5-finger snapshot captured (+{n_new} new voxels).")
            changed = changed or (n_new > 0)
            discrete = True
        if ev.caliper_pinch:
            dist = float(np.linalg.norm(right.tips_pad["thumb"] - right.tips_pad["index"]))
            self.cloud.add_caliper_width(dist)
            n_cal = self.cloud.add_points(np.vstack([right.tips_pad["thumb"], right.tips_pad["index"]]))
            self._log(f"Caliper pinch recorded: {dist * 1000.0:.1f} mm.")
            changed = changed or (n_cal > 0)
            discrete = True
        return changed, discrete

    def _confirmed_catalog_key_unlocked(self) -> str | None:
        """Return the catalog key strictly when the current scan exceeds 90% confidence."""
        best_k = self._last_match.best_key
        if best_k is not None and best_k in self.catalog and self._last_scan.n_voxels >= 10:
            prob = float(self._last_match.probabilities.get(best_k, 0.0))
            if prob > 0.90:
                return best_k
        return None

    def _refresh_scan_and_match_unlocked(self) -> None:
        self._last_scan = self.cloud.summarize()
        self._last_match = match_scan_to_catalog(self._last_scan, self.catalog)
        confirmed_k = self._confirmed_catalog_key_unlocked()
        if confirmed_k is not None:
            prob = float(self._last_match.probabilities.get(confirmed_k, 0.0))
            if self._logged_90_key != confirmed_k:
                self._logged_90_key = confirmed_k
                self._log(
                    f"CONFIRMED >90%: {self.catalog[confirmed_k].label} ({prob * 100.0:.0f}%) — 3D structure locked!"
                )
        else:
            if self._logged_90_key is not None:
                prev_label = self.catalog[self._logged_90_key].label
                prev_prob = float(self._last_match.probabilities.get(self._logged_90_key, 0.0)) * 100.0
                self._logged_90_key = None
                self._log(
                    f"Confidence in {prev_label} fell to {prev_prob:.0f}% (<=90%) — stopped rendering 3D object."
                )
            if self.relocation.phase != "carrying":
                self._locked_key = None

    def _estimate_scan_pose_unlocked(
        self, ref_override: RefObject | None = None
    ) -> tuple[np.ndarray, float, float, np.ndarray, bool]:
        xy = self._last_scan.center_xy
        h = self._last_scan.height_m
        yaw = self._last_scan.yaw_rad
        cy0, sy0 = math.cos(yaw), math.sin(yaw)
        R_scan = np.array([[cy0, -sy0, 0.0], [sy0, cy0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        upright = True

        confirmed_key = self._confirmed_catalog_key_unlocked()
        key = (ref_override.key if ref_override is not None else None) or confirmed_key
        ref_obj = ref_override or (self.catalog.get(key) if key else None)
        if ref_obj is not None and self._last_scan.n_voxels >= 10:
            pts = self.cloud.points()
            ext = ref_obj.extents()
            if ref_obj.shape == "sphere":
                xy, yaw = fit_reference_pose_xy_yaw(pts, "sphere", ref_obj.dims, self.cal.z_table, xy, yaw)
                h = float(ext[2])
                cy, sy = math.cos(yaw), math.sin(yaw)
                R_scan = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
                upright = True
            elif ref_obj.shape == "cylinder":
                cyl_fit = self._last_scan.primitive_fit.get("fits", {}).get("cylinder")
                is_side = bool(cyl_fit and cyl_fit.get("sideways", False)) or (
                    abs(h - float(ext[0])) + 0.015 < abs(h - float(ext[2]))
                    and self._last_scan.length_m > 0.5 * (float(ext[0]) + float(ext[2]))
                )
                if is_side:
                    xy, yaw = fit_reference_pose_xy_yaw(pts, "cylinder_side", ref_obj.dims, self.cal.z_table, xy, yaw)
                    h = float(ext[0])
                    cy, sy = math.cos(yaw), math.sin(yaw)
                    R_scan = np.array([[0.0, sy, cy], [0.0, -cy, sy], [1.0, 0.0, 0.0]], dtype=float)
                    upright = False
                else:
                    xy, yaw = fit_reference_pose_xy_yaw(pts, "cylinder", ref_obj.dims, self.cal.z_table, xy, yaw)
                    h = float(ext[2])
                    cy, sy = math.cos(yaw), math.sin(yaw)
                    R_scan = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
                    upright = True
            else:
                best_k_up = 2
                best_axes = (0, 1)
                best_lwh = (float(ext[0]), float(ext[1]), float(ext[2]))
                best_sc = float("inf")
                for k_up in (2, 1, 0):
                    hr = float(ext[k_up])
                    horiz = [j for j in range(3) if j != k_up]
                    j_long = max(horiz, key=lambda j: float(ext[j]))
                    j_short = [j for j in horiz if j != j_long][0]
                    lr, wr = float(ext[j_long]), float(ext[j_short])
                    sc = (
                        ((self._last_scan.height_m - hr) / max(0.011, 0.08 * hr)) ** 2
                        + ((self._last_scan.length_m - lr) / max(0.011, 0.08 * lr)) ** 2
                        + ((self._last_scan.width_m - wr) / max(0.011, 0.08 * wr)) ** 2
                    )
                    if sc < best_sc:
                        best_sc = sc
                        best_k_up = k_up
                        best_axes = (j_long, j_short)
                        best_lwh = (lr, wr, hr)

                if best_k_up == 2:
                    xy, yaw = fit_reference_pose_xy_yaw(pts, "box", ref_obj.dims, self.cal.z_table, xy, yaw)
                    h = float(ext[2])
                    cy, sy = math.cos(yaw), math.sin(yaw)
                    R_scan = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
                    upright = True
                else:
                    j_long, j_short = best_axes
                    lr, wr, hr = best_lwh
                    xy, yaw = fit_reference_pose_xy_yaw(
                        pts, "box", {"x": lr, "y": wr, "z": hr}, self.cal.z_table, xy, yaw
                    )
                    h = hr
                    cy, sy = math.cos(yaw), math.sin(yaw)
                    R_scan = np.zeros((3, 3), dtype=float)
                    R_scan[:, best_k_up] = [0.0, 0.0, 1.0]
                    R_scan[:, j_long] = [cy, sy, 0.0]
                    if (j_long - best_k_up) % 3 == 1:
                        R_scan[:, j_short] = np.cross(R_scan[:, best_k_up], R_scan[:, j_long])
                    else:
                        R_scan[:, j_short] = np.cross(R_scan[:, j_long], R_scan[:, best_k_up])
                    upright = False

        return xy, h, yaw, R_scan, upright

    def _enter_carrying_unlocked(self, right: HandPose) -> bool:
        if self.relocation.phase == "scanning" and self._last_scan.n_voxels == 0:
            self._log("Cannot pick up yet: scan an object first (0 voxels).")
            return False

        self._push_undo_unlocked()
        self.cloud.locked = True
        self.gestures.clear_scan_commands()
        self._was_streaming = False
        if self._last_gesture_ev is not None:
            self._last_gesture_ev.stream_index = False
            self._last_gesture_ev.snapshot_5 = False
            self._last_gesture_ev.caliper_pinch = False
        self._placed_dirty = False
        self._locked_key = self._confirmed_catalog_key_unlocked()
        ref_obj = self.catalog.get(self._locked_key) if self._locked_key else None
        if ref_obj is not None:
            H_init = float(ref_obj.extents()[2]) if self._last_scan.height_m < 0.02 else self._last_scan.height_m
        else:
            H_init = max(self._last_scan.height_m, 0.02)
        fallback_xy = (
            self._last_scan.center_xy
            if self._last_scan.n_voxels >= 1
            else right.palm_center[:2]
        )
        if self.relocation.phase == "scanning":
            if ref_obj is not None:
                xy_s, h_s, yaw_s, R_s, up_s = self._estimate_scan_pose_unlocked(ref_override=ref_obj)
                self.relocation.set_scan_pose(xy_s, h_s, yaw_s, rotation=R_s, upright=up_s)
            else:
                self.relocation.set_scan_pose(fallback_xy, H_init, self._last_scan.yaw_rad)
        T_ws_hand = make_se3(right.wrist_rot, right.wrist_pos)
        self.relocation.lock_grasp(
            T_ws_hand,
            fallback_xy=fallback_xy,
            height_m=H_init,
            yaw_rad=self._last_scan.yaw_rad,
        )
        if ref_obj is not None and self._locked_key is not None:
            conf = self._last_match.probabilities.get(self._locked_key, 0.0) * 100.0
            self._log(f"LOCK_GRASP: Locked onto {ref_obj.label} ({conf:.0f}%). Point capture locked while carrying!")
        else:
            self._log("LOCK_GRASP: Carrying unmatched 3D scan (no catalog shape assigned). Point capture locked!")
        return True

    def _finalize_placement_unlocked(
        self, right: HandPose, *, placed_ts: float | None = None
    ) -> None:
        self._push_undo_unlocked()
        key = self._confirmed_catalog_key_unlocked()
        self._locked_key = key
        ref_obj = self.catalog.get(key) if key else None
        pose_before = self.relocation._initial_pose
        T_ws_hand = make_se3(right.wrist_rot, right.wrist_pos)
        placed = self.relocation.place_on_table(T_ws_hand, right.pad_matrix(), ref_obj)

        # Relocate accumulated 3D cloud & Convex Hull in full 3D SE(3) onto the placed pose
        # so the point cloud, convex hull, and rendered 3D object ALWAYS share the exact same orientation!
        if pose_before is not None:
            T_before = make_se3(pose_before.rotation, pose_before.position_m)
            T_placed = make_se3(placed.rotation, placed.position_m)
            self.cloud.transform_se3(
                T_before,
                T_placed,
                snap_min_z_to_table=(ref_obj is None),
            )
            self._refresh_scan_and_match_unlocked()
            if ref_obj is None and self._last_scan.n_voxels >= 1:
                h_new = max(self._last_scan.height_m, 0.02)
                placed.resting_height_m = h_new
                placed.position_m[2] = self.cal.z_table + 0.5 * h_new

        self.cloud.locked = False
        self.gestures.clear_scan_commands()
        if placed_ts is not None:
            self._placed_at_ts = placed_ts
        elif self._last_frame_ws is not None:
            self._placed_at_ts = self._last_frame_ws.timestamp
        else:
            self._placed_at_ts = time.time()
        self._placed_dirty = False
        self._write_placed_outputs_unlocked(placed)
        if self._locked_key is not None and self._g1_plan is not None:
            mj_tag = self._mujoco_result["outcome"].upper() if self._mujoco_result else "OK"
            arm_tag = self._g1_plan["arm"] if self._g1_plan else "auto"
            self._log(
                f"PLACED at ws=({placed.position_m[0]*100:.1f}, {placed.position_m[1]*100:.1f}) cm | "
                f"MuJoCo: {mj_tag} | Unitree G1 ({arm_tag} arm) plan ready!"
            )
        else:
            self._log(
                f"PLACED unmatched scan at ws=({placed.position_m[0]*100:.1f}, {placed.position_m[1]*100:.1f}) cm "
                "(no catalog shape assigned)."
            )

    def _write_placed_outputs_unlocked(self, placed: ObjectPoseEstimate) -> None:
        self.cloud.export_ply(str(self.session_dir / "cloud.ply"))
        key = self._confirmed_catalog_key_unlocked()
        self._locked_key = key
        ref_obj = self.catalog.get(key) if key else None
        if ref_obj is None:
            self._object_card = None
            self._mujoco_result = None
            self._g1_plan = None
            for stale_name in ("object_card.json", "unitree_g1_pick.json"):
                stale_p = self.session_dir / stale_name
                if stale_p.exists():
                    stale_p.unlink()
            return

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

        # Save object_card.json and unitree_g1_pick.json atomically
        card_tmp = self.session_dir / f"object_card.json.{os.getpid()}.tmp"
        card_tmp.write_text(json.dumps(self._object_card, indent=2), encoding="utf-8")
        os.replace(card_tmp, self.session_dir / "object_card.json")
        dispatch_unitree_g1_plan(
            self._g1_plan,
            self.session_dir / "unitree_g1_pick.json",
            udp_target=self.g1_udp,
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
        self._placed_dirty = False
        self._placed_at_ts = -1e9
        self._logged_90_key = None
        self._was_streaming = False

    def handle_action(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle interactive commands from Web UI buttons or CLI shortcuts."""
        payload = payload or {}
        with self._lock:
            if action == "undo":
                self._undo_unlocked()

            elif action == "reset":
                self._push_undo_unlocked()
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
                        self._undo_stack.clear()
                        self._last_frame_ws = DualHandFrame(
                            timestamp=raw.timestamp,
                            right=self._transform_hand_to_ws(raw.right),
                            left=self._transform_hand_to_ws(raw.left),
                            mode=raw.mode,
                            seq=raw.seq,
                        )
                        self._log("Table plane calibrated from flat right hand (z_table = 0.0 m)!")

            elif action in ("trace_toggle", "trace_on", "trace_off", "snapshot", "caliper", "phase"):
                right_now = self._last_frame_ws.right if self._last_frame_ws else None
                if action == "phase" and right_now is not None:
                    if self.relocation.phase in ("scanning", "placed"):
                        self._enter_carrying_unlocked(right_now)
                    elif self.relocation.phase == "carrying":
                        self._finalize_placement_unlocked(right_now)
                elif self.relocation.phase == "carrying":
                    # Never allow manual/UI/keyboard scan commands to add or queue points during pick-up
                    self.cloud.locked = True
                    self.gestures.clear_scan_commands()
                    self._log("Ignored scan command: cannot add points during pick-up (CARRYING).")
                elif action in ("snapshot", "caliper") and right_now is not None and self.relocation.phase in ("scanning", "placed"):
                    synth_ev = GestureEvent(
                        raw_sign="idle",
                        active_sign="fist" if action == "snapshot" else "caliper",
                        snapshot_5=(action == "snapshot"),
                        caliper_pinch=(action == "caliper"),
                    )
                    self._ingest_tactile_events_unlocked(right_now, synth_ev)
                    prev_locked = self._locked_key
                    self._refresh_scan_and_match_unlocked()
                    if self.relocation.phase == "scanning" and self._last_scan.n_voxels >= 3:
                        xy_s, h_s, yaw_s, R_s, up_s = self._estimate_scan_pose_unlocked()
                        self.relocation.set_scan_pose(xy_s, h_s, yaw_s, rotation=R_s, upright=up_s)
                    elif self.relocation.phase == "placed":
                        self._locked_key = self._confirmed_catalog_key_unlocked()
                        ref_obj = self.catalog.get(self._locked_key) if self._locked_key else None
                        xy_s, h_s, yaw_s, R_s, up_s = self._estimate_scan_pose_unlocked(ref_override=ref_obj)
                        newly_locked = prev_locked != self._locked_key and self._locked_key is not None
                        if newly_locked:
                            self.relocation._current_pose = None
                        placed_est = self.relocation.refresh_placed_from_scan(
                            xy_s,
                            h_s,
                            yaw_s,
                            n_voxels=self._last_scan.n_voxels,
                            ref_obj=ref_obj,
                            rotation=R_s if newly_locked else None,
                            upright=up_s if newly_locked else None,
                        )
                        self._write_placed_outputs_unlocked(placed_est)
                        self._placed_dirty = False
                else:
                    self.gestures.queue_command(action)
                    if (
                        action == "trace_toggle"
                        and not self.gestures.manual_trace_enabled
                        and self.relocation.phase == "placed"
                        and self._placed_dirty
                        and self.relocation._current_pose is not None
                    ):
                        self._write_placed_outputs_unlocked(self.relocation._current_pose)
                        self._placed_dirty = False

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
                    self._undo_stack.clear()
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
        confirmed_key = self._confirmed_catalog_key_unlocked()
        if self.relocation.phase == "scanning":
            matched_key = confirmed_key or self._last_match.best_key
        else:
            matched_key = confirmed_key
        matched_obj = self.catalog.get(matched_key) if matched_key else None
        best_prob = float(self._last_match.probabilities.get(matched_key, 0.0)) if matched_key else 0.0
        confirmed_90 = bool(
            confirmed_key is not None
            and matched_obj is not None
            and self._last_scan.n_voxels >= 10
            and best_prob > 0.90
        )

        # Visually carry the 3D point cloud & Convex Hull in-hand during CARRYING
        pts_vis = pts
        hull_verts_vis = None if hull is None else hull.vertices
        if (
            self.relocation.phase == "carrying"
            and self.relocation._initial_pose is not None
            and pose is not None
        ):
            T0 = make_se3(self.relocation._initial_pose.rotation, self.relocation._initial_pose.position_m)
            Tt = make_se3(pose.rotation, pose.position_m)
            T_carry = Tt @ invert_se3(T0)
            R_c, t_c = T_carry[:3, :3], T_carry[:3, 3]
            if len(pts) > 0:
                pts_vis = (R_c @ pts.T).T + t_c
            if hull_verts_vis is not None and len(hull_verts_vis) > 0:
                hull_verts_vis = (R_c @ hull_verts_vis.T).T + t_c

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
        if hull is not None and hull_verts_vis is not None:
            hull_payload = {
                "vertices": np.round(hull_verts_vis, 4).tolist(),
                "faces": hull.faces.tolist(),
                "volume_cm3": round(hull.volume_m3 * 1e6, 1),
                "area_cm2": round(hull.area_m2 * 1e4, 1),
            }

        tracked_mesh_info = None
        if pose is not None and matched_obj is not None and confirmed_90:
            ext = matched_obj.extents()
            tracked_mesh_info = {
                "key": matched_obj.key,
                "label": matched_obj.label,
                "shape": matched_obj.shape,
                "material": matched_obj.material,
                "container": bool(matched_obj.container),
                "sloped_top_m": round(float(matched_obj.sloped_top_m), 4),
                "mass_g": round(float(matched_obj.mass_kg) * 1000.0, 1),
                "max_grip_n": round(float(matched_obj.max_grip_n), 1),
                "extents_m": [round(float(v), 4) for v in ext],
                "confidence": round(best_prob, 4),
                "confirmed_90": confirmed_90,
                "pose": pose.to_dict(),
            }

        raw_fits = self._last_scan.primitive_fit.get("fits", {})
        prim_fits_payload = {
            sh: {
                "dims_m": {dk: round(float(dv), 4) for dk, dv in fd.get("dims_m", {}).items()},
                "center_xy": [round(float(c), 4) for c in fd.get("center_xy", [0.0, 0.0])],
                "yaw_rad": round(float(fd.get("yaw_rad", 0.0)), 4),
                "rms_mm": round(float(fd.get("rms_m", 0.0)) * 1000.0, 2),
            }
            for sh, fd in raw_fits.items()
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
            "gesture": self._describe_gesture_for_snapshot(),
            "hands": hands_payload,
            "cloud": {
                "n_raw": self._last_scan.n_raw,
                "n_voxels": self._last_scan.n_voxels,
                "points": np.round(pts_vis, 4).tolist(),
                "height_mm": round(self._last_scan.height_m * 1000.0, 1),
                "length_mm": round(self._last_scan.length_m * 1000.0, 1),
                "width_mm": round(self._last_scan.width_m * 1000.0, 1),
                "calipers_mm": [round(c * 1000.0, 1) for c in self._last_scan.calipers_m],
                "primitive_best": self._last_scan.primitive_fit.get("best"),
                "primitive_weights": self._last_scan.primitive_fit.get("weights", {}),
                "primitive_fits": prim_fits_payload,
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
                "confirmed_90": confirmed_90,
                "best_probability": round(best_prob, 4),
                "probabilities": self._last_match.probabilities,
                "advice": self._last_match.advice,
            },
            "tracked_object": tracked_mesh_info,
            "object_card": self._object_card,
            "mujoco_result": self._mujoco_result,
            "g1_plan": self._g1_plan,
            "events": list(self._event_log[-10:]),
        }

    def _describe_gesture_for_snapshot(self) -> dict[str, Any]:
        if self._last_gesture_ev is None:
            return {"sign": "idle", "badge": "WAITING FOR STREAM", "stream_index": False, "pinch_dist_mm": 0.0}
        desc = self.gestures.describe_state(self._last_gesture_ev)
        if self.relocation.phase == "carrying":
            desc["stream_index"] = False
            desc["manual_trace"] = False
            if desc.get("sign") != "peace":
                desc["sign"] = "idle"
                desc["badge"] = "CARRYING — SCAN LOCKED (✌️ TO PLACE)"
        return desc

    def close(self) -> None:
        if self.rokoko is not None:
            self.rokoko.close()
