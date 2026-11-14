"""Unit and end-to-end integration tests for `scan_track`."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from glove_twin.objects import OBJECTS
from scan_track.calibration import WorkspaceCalibration, fit_table_plane, kabsch_se3
from scan_track.catalog_matcher import (
    build_matched_object_card,
    match_scan_to_catalog,
    validate_card_schema,
)
from scan_track.cloud_hull import (
    TactileCloudBuilder,
    _incremental_hull_3d_numpy,
    compute_convex_hull_3d,
    minimum_area_obb_2d,
)
from scan_track.gestures import GestureController, classify_left_hand_sign
from scan_track.pipeline import ScanTrackSession
from scan_track.relocation import RelocationTracker, make_se3, upright_se3
from scan_track.rokoko_stream import SyntheticExplorationStream, parse_rokoko_frame
from scan_track.unitree_g1_bridge import build_unitree_g1_pick_plan, verify_in_mujoco


class CalibrationAndGeometryTests(unittest.TestCase):
    def test_kabsch_se3_recovers_exact_transform(self):
        rng = np.random.default_rng(12)
        src = rng.uniform(-0.3, 0.3, size=(6, 3))
        src[:, 2] = 0.0  # coplanar table landmarks still recover det(R)=+1 cleanly!
        ang = 0.62
        R_true = np.array([[math.cos(ang), -math.sin(ang), 0.0],
                           [math.sin(ang), math.cos(ang), 0.0],
                           [0.0, 0.0, 1.0]])
        t_true = np.array([0.42, -0.08, 0.03])
        dst = (src @ R_true.T) + t_true
        R_est, t_est, rms = kabsch_se3(src, dst)
        self.assertAlmostEqual(float(np.linalg.det(R_est)), 1.0, places=6)
        self.assertLess(rms, 1e-9)
        np.testing.assert_allclose(R_est, R_true, atol=1e-7)
        np.testing.assert_allclose(t_est, t_true, atol=1e-7)

    def test_table_plane_leveling(self):
        # Tilted table plane (5 deg roll) at z = 0.12 m
        rng = np.random.default_rng(3)
        xy = rng.uniform(-0.2, 0.2, size=(10, 2))
        roll = math.radians(5.0)
        pts = np.column_stack([xy[:, 0], xy[:, 1], 0.12 + math.tan(roll) * xy[:, 1]])
        n, _, rms = fit_table_plane(pts)
        self.assertLess(rms, 1e-9)
        self.assertGreater(n[2], 0.99)
        cal = WorkspaceCalibration.from_table_points(pts)
        levelled = cal.raw_to_workspace_points(pts)
        np.testing.assert_allclose(levelled[:, 2], 0.0, atol=1e-6)

    def test_pure_numpy_and_default_3d_convex_hull_volume(self):
        # 0.10 x 0.08 x 0.12 m box = 0.00096 m^3
        corners = np.array([
            [x, y, z]
            for x in (0.0, 0.10)
            for y in (0.0, 0.08)
            for z in (0.0, 0.12)
        ], dtype=float)
        interior = np.array([[0.05, 0.04, 0.06], [0.03, 0.02, 0.05]])
        pts = np.vstack([corners, interior])

        hull_np = _incremental_hull_3d_numpy(pts)
        self.assertIsNotNone(hull_np)
        assert hull_np is not None
        self.assertAlmostEqual(hull_np.volume_m3, 0.10 * 0.08 * 0.12, delta=1e-5)

        hull_auto = compute_convex_hull_3d(pts)
        self.assertIsNotNone(hull_auto)
        assert hull_auto is not None
        self.assertAlmostEqual(hull_auto.volume_m3, 0.10 * 0.08 * 0.12, delta=1e-5)

    def test_rotating_calipers_obb_2d(self):
        yaw = math.radians(32.0)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        rect = np.array([[-0.06, -0.025], [0.06, -0.025], [0.06, 0.025], [-0.06, 0.025]])
        pts = (rect @ R.T) + np.array([0.15, 0.30])
        obb = minimum_area_obb_2d(pts)
        self.assertAlmostEqual(obb["length_m"], 0.12, places=4)
        self.assertAlmostEqual(obb["width_m"], 0.05, places=4)
        np.testing.assert_allclose(obb["center_xy"], [0.15, 0.30], atol=1e-5)


class RokokoAndGestureTests(unittest.TestCase):
    def test_left_hand_gestures_are_mutually_distinct(self):
        synth = SyntheticExplorationStream("redbull", seed=1)
        for expected in ("idle", "pinch", "fist", "peace"):
            lh = synth._left_hand_for_gesture(expected)
            sign, _, _ = classify_left_hand_sign(lh)
            self.assertEqual(sign, expected)

    def test_gesture_controller_debounce_and_cooldown(self):
        synth = SyntheticExplorationStream("redbull", seed=2)
        gc = GestureController(hold_sec=0.10, cooldown_sec=0.50)
        lh_fist = synth._left_hand_for_gesture("fist")
        lh_idle = synth._left_hand_for_gesture("idle")

        # Initial frame before hold_sec: should not fire yet
        ev0 = gc.update(lh_fist, 1.00)
        self.assertFalse(ev0.snapshot_5)
        # Held past 0.10 s: fires exactly once!
        ev1 = gc.update(lh_fist, 1.15)
        self.assertTrue(ev1.snapshot_5)
        # Still holding fist: latched, does not double-fire
        ev2 = gc.update(lh_fist, 1.25)
        self.assertFalse(ev2.snapshot_5)
        # Release to idle and re-fist after cooldown -> fires again
        gc.update(lh_idle, 1.35)
        gc.update(lh_fist, 1.70)
        ev3 = gc.update(lh_fist, 1.85)
        self.assertTrue(ev3.snapshot_5)

    def test_parse_rokoko_studio_v4_packet_and_palmar_offset(self):
        # Construct a Unity Y-up packet for a right hand flat palm-down at y=0.05m above table
        body = {
            "rightHand": {"position": {"x": 0.0, "y": 0.05, "z": 0.25}},
            "rightThumbProximal": {"position": {"x": -0.035, "y": 0.05, "z": 0.28}},
            "rightThumbTip": {"position": {"x": -0.055, "y": 0.05, "z": 0.33}},
            "rightIndexProximal": {"position": {"x": -0.025, "y": 0.05, "z": 0.32}},
            "rightIndexDistal": {"position": {"x": -0.025, "y": 0.05, "z": 0.37}},
            "rightIndexTip": {"position": {"x": -0.025, "y": 0.05, "z": 0.39}},
            "rightMiddleProximal": {"position": {"x": -0.005, "y": 0.05, "z": 0.32}},
            "rightMiddleTip": {"position": {"x": -0.005, "y": 0.05, "z": 0.40}},
            "rightRingProximal": {"position": {"x": 0.015, "y": 0.05, "z": 0.31}},
            "rightRingTip": {"position": {"x": 0.015, "y": 0.05, "z": 0.39}},
            "rightLittleProximal": {"position": {"x": 0.032, "y": 0.05, "z": 0.30}},
            "rightLittleTip": {"position": {"x": 0.032, "y": 0.05, "z": 0.37}},
        }
        pkt = {"version": 4, "timestamp": 12.3, "scene": {"actors": [{"body": body}]}}
        fr = parse_rokoko_frame(pkt, mode="coil", pad_offset_m=0.007)
        self.assertIsNotNone(fr.right)
        assert fr.right is not None
        # Palm-down hand -> palmar normal points downward (-Z), so pad Z < bone Z by ~7 mm!
        self.assertLess(fr.right.tips_pad["index"][2], fr.right.tips_bone["index"][2] - 0.005)

        # Injecting NaN into a finger tip drops the corrupt hand cleanly rather than poisoning downstream math
        body["rightIndexTip"]["position"]["x"] = float("nan")
        fr_nan = parse_rokoko_frame(pkt, mode="coil", pad_offset_m=0.007)
        self.assertIsNone(fr_nan.right)

        # Also verify true Rokoko Studio JSON v4 Newton flat joint array (RightFinger1..5)
        newton_joints = [
            {"name": "RightHand", "position": {"x": 0.0, "y": 0.05, "z": 0.25}},
            {"name": "RightFinger1Metacarpal", "position": {"x": -0.020, "y": 0.05, "z": 0.26}},
            {"name": "RightFinger1Proximal", "position": {"x": -0.035, "y": 0.05, "z": 0.28}},
            {"name": "RightFinger1Tip", "position": {"x": -0.055, "y": 0.05, "z": 0.33}},
            {"name": "RightFinger2Metacarpal", "position": {"x": -0.025, "y": 0.05, "z": 0.28}},
            {"name": "RightFinger2Proximal", "position": {"x": -0.025, "y": 0.05, "z": 0.32}},
            {"name": "RightFinger2Distal", "position": {"x": -0.025, "y": 0.05, "z": 0.37}},
            {"name": "RightFinger2Tip", "position": {"x": -0.025, "y": 0.05, "z": 0.39}},
            {"name": "RightFinger3Proximal", "position": {"x": -0.005, "y": 0.05, "z": 0.32}},
            {"name": "RightFinger3Tip", "position": {"x": -0.005, "y": 0.05, "z": 0.40}},
            {"name": "RightFinger4Proximal", "position": {"x": 0.015, "y": 0.05, "z": 0.31}},
            {"name": "RightFinger4Tip", "position": {"x": 0.015, "y": 0.05, "z": 0.39}},
            {"name": "RightFinger5Proximal", "position": {"x": 0.032, "y": 0.05, "z": 0.30}},
            {"name": "RightFinger5Tip", "position": {"x": 0.032, "y": 0.05, "z": 0.37}},
        ]
        pkt_newton = {"version": "3,0", "timestamp": 12.4, "scene": {"newtons": [{"joints": newton_joints}]}}
        fr_newton = parse_rokoko_frame(pkt_newton, mode="coil", pad_offset_m=0.007)
        self.assertIsNotNone(fr_newton.right)
        assert fr_newton.right is not None
        self.assertAlmostEqual(fr_newton.right.joints["index_mcp"][1], 0.32, places=5)
        self.assertLess(fr_newton.right.tips_pad["index"][2], fr_newton.right.tips_bone["index"][2] - 0.005)


class RelocationAndEndToEndTests(unittest.TestCase):
    def test_relocation_snaps_vertical_drift_and_tracks_move(self):
        tracker = RelocationTracker(z_table=0.0)
        obj = OBJECTS["redbull"]
        tracker.set_scan_pose(np.array([0.05, 0.30]), obj.extents()[2], yaw_rad=0.1)

        # Wrist grasps from -Y side at t0
        T_h0 = upright_se3(np.array([0.05, 0.18]), 0.08, yaw_rad=0.1)
        tracker.lock_grasp(T_h0)

        # Carry to new spot (+20 cm X, +10 cm Y, rotated +45 deg), PLUS 18 mm Coil Pro vertical drift!
        d_yaw = math.radians(45.0)
        c, s = math.cos(d_yaw), math.sin(d_yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T_delta = make_se3(Rz, np.array([0.20, 0.10, 0.018]))  # +18 mm Z drift!
        T_h1 = T_delta @ T_h0
        expected_xy = (Rz[:2, :2] @ np.array([0.05, 0.30])) + np.array([0.20, 0.10])

        placed = tracker.place_on_table(T_h1, finger_pads_ws=None, ref_obj=obj)
        np.testing.assert_allclose(placed.position_m[:2], expected_xy, atol=1e-5)
        # Table snap must completely eliminate the +18 mm Z drift: bottom sits strictly at z = 0.0!
        self.assertAlmostEqual(placed.position_m[2], 0.5 * obj.extents()[2], places=6)

    def test_end_to_end_pipeline_all_catalog_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            for key in ("redbull", "profeel", "wood_sphere"):
                sess = ScanTrackSession(
                    source="synthetic",
                    synthetic_object=key,
                    sessions_root=tmp,
                )
                assert sess.synthetic is not None
                sess.synthetic.auto_cycle = False
                truth = sess.synthetic.ground_truth()

                dt = 1.0 / 30.0
                snap = {}
                for _ in range(int(13.5 / dt)):
                    snap = sess.tick(dt=dt)
                sess.close()

                self.assertEqual(snap["phase"], "placed", msg=f"{key} did not reach placed phase")
                self.assertEqual(snap["catalog"]["best_key"], key, msg=f"wrong catalog match for {key}")
                self.assertIsNotNone(snap["object_card"])
                self.assertEqual(validate_card_schema(snap["object_card"]), [])

                # Check relocation accuracy (< 12 mm on XY)
                pose = snap["tracked_object"]["pose"]
                err_mm = math.hypot(
                    pose["x_m"] - truth["placed_pose"]["x"],
                    pose["y_m"] - truth["placed_pose"]["y"],
                ) * 1000.0
                self.assertLess(err_mm, 12.0, msg=f"relocation error too high on {key}: {err_mm:.1f} mm")

                # Check MuJoCo verification & Unitree G1 plan files on disk
                self.assertEqual(snap["mujoco_result"]["outcome"], "lifted")
                self.assertTrue(snap["g1_plan"]["safe_to_execute"])
                self.assertEqual(len(snap["g1_plan"]["waypoints"]), 4)
                self.assertEqual(len(snap["g1_plan"]["dex3_1_hand"]["q_grasp_rad"]), 7)
                self.assertTrue((Path(snap["session_dir"]) / "unitree_g1_pick.json").exists())
                self.assertTrue((Path(snap["session_dir"]) / "object_card.json").exists())
                self.assertTrue((Path(snap["session_dir"]) / "cloud.ply").exists())


if __name__ == "__main__":
    unittest.main()
