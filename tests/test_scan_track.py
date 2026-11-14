"""Unit and end-to-end integration tests for `scan_track`."""
from __future__ import annotations

import json
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
    fit_reference_pose_xy_yaw,
    minimum_area_obb_2d,
)
from scan_track.gestures import GestureController, classify_left_hand_sign, is_open_hand_sign
from scan_track.live_server import INDEX_HTML
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

    def test_gesture_requires_open_hand_between_gestures(self):
        synth = SyntheticExplorationStream("redbull", seed=3)
        gc = GestureController(hold_sec=0.10, pinch_hold_sec=0.05, cooldown_sec=0.20, open_hold_sec=0.06)
        lh_open = synth._left_hand_for_gesture("idle")
        lh_pinch = synth._left_hand_for_gesture("pinch")
        lh_fist = synth._left_hand_for_gesture("fist")
        lh_peace = synth._left_hand_for_gesture("peace")

        # Mid-transition posture between peace and fist: classified as raw "idle", but ring/pinky stay curled
        lh_half_curled = synth._make_hand_from_curls(
            "left",
            lh_open.wrist_pos,
            lh_open.wrist_rot,
            {"thumb": 0.85, "index": 0.45, "middle": 0.45, "ring": 1.08, "pinky": 1.08},
        )
        self.assertEqual(classify_left_hand_sign(lh_half_curled)[0], "idle")
        self.assertFalse(is_open_hand_sign(lh_half_curled))
        self.assertTrue(is_open_hand_sign(lh_open))

        # 1. Open hand -> Pinch activates cleanly
        gc.update(lh_open, 0.00)
        gc.update(lh_open, 0.10)
        gc.update(lh_pinch, 0.20)
        ev_pinch = gc.update(lh_pinch, 0.28)
        self.assertEqual(ev_pinch.active_sign, "pinch")
        self.assertTrue(ev_pinch.stream_index)

        # Direct Pinch -> Fist (no open hand) MUST be blocked
        gc.update(lh_fist, 0.35)
        ev_fist_blocked = gc.update(lh_fist, 0.55)
        self.assertEqual(ev_fist_blocked.active_sign, "idle")
        self.assertFalse(ev_fist_blocked.snapshot_5)
        self.assertFalse(ev_fist_blocked.stream_index)
        self.assertTrue(ev_fist_blocked.awaiting_open_hand)
        self.assertEqual(gc.describe_state(ev_fist_blocked)["badge"], "OPEN HAND FIRST")

        # Chaining into Peace or back into Pinch without opening hand stays blocked
        gc.update(lh_peace, 0.65)
        ev_peace_blocked = gc.update(lh_peace, 0.85)
        self.assertEqual(ev_peace_blocked.active_sign, "idle")
        self.assertFalse(ev_peace_blocked.phase_toggle)
        gc.update(lh_pinch, 0.95)
        ev_pinch_blocked = gc.update(lh_pinch, 1.05)
        self.assertFalse(ev_pinch_blocked.stream_index)

        # 2. Returning to Open Hand -> Peace works; Peace -> half-curled "idle" -> Fist stays blocked!
        gc.update(lh_open, 1.20)
        ev_open = gc.update(lh_open, 1.30)
        self.assertFalse(ev_open.awaiting_open_hand)
        gc.update(lh_peace, 1.40)
        ev_peace_ok = gc.update(lh_peace, 1.55)
        self.assertTrue(ev_peace_ok.phase_toggle)

        gc.update(lh_half_curled, 1.65)
        gc.update(lh_half_curled, 1.85)
        gc.update(lh_fist, 1.95)
        ev_fist_blocked2 = gc.update(lh_fist, 2.15)
        self.assertFalse(ev_fist_blocked2.snapshot_5)
        self.assertEqual(ev_fist_blocked2.active_sign, "idle")
        self.assertTrue(ev_fist_blocked2.awaiting_open_hand)

        # 3. Genuine Open Hand -> Fist -> Open Hand -> Pinch succeeds
        gc.update(lh_open, 2.30)
        gc.update(lh_open, 2.40)
        gc.update(lh_fist, 2.50)
        ev_fist_ok = gc.update(lh_fist, 2.65)
        self.assertTrue(ev_fist_ok.snapshot_5)

        gc.update(lh_open, 2.80)
        gc.update(lh_open, 2.90)
        gc.update(lh_pinch, 3.00)
        ev_pinch_ok = gc.update(lh_pinch, 3.10)
        self.assertTrue(ev_pinch_ok.stream_index)

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
            for key in ("redbull", "kantolan_suola_cracker", "wood_sphere"):
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

    def test_post_placement_actions_and_repeated_relocation_not_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(
                source="synthetic",
                synthetic_object="redbull",
                sessions_root=tmp,
            )
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False

            dt = 1.0 / 30.0
            snap = {}
            for _ in range(int(13.5 / dt)):
                snap = sess.tick(dt=dt)
            self.assertEqual(snap["phase"], "placed")

            placed1_xy = np.array([
                snap["tracked_object"]["pose"]["x_m"],
                snap["tracked_object"]["pose"]["y_m"],
            ])
            # Cloud & Convex Hull must have moved onto placed1_xy alongside the object
            cloud_pts1 = np.asarray(snap["cloud"]["points"])
            np.testing.assert_allclose(cloud_pts1[:, :2].mean(axis=0), placed1_xy, atol=0.025)

            # 1. Position right hand around placed1_xy; Caliper & Snapshot actions in PLACED must NOT be blocked
            assert sess._last_frame_ws is not None and sess._last_frame_ws.right is not None
            r0 = sess._last_frame_ws.right
            r0.tips_pad["thumb"] = np.array([placed1_xy[0] - 0.0265, placed1_xy[1], 0.075])
            for idx_f, fname in enumerate(("index", "middle", "ring", "pinky")):
                r0.tips_pad[fname] = np.array([placed1_xy[0] + 0.0265, placed1_xy[1], 0.095 - 0.015 * idx_f])

            n_cal_before = len(snap["cloud"]["calipers_mm"])
            snap = sess.handle_action("caliper")
            self.assertEqual(snap["phase"], "placed")
            self.assertEqual(len(snap["cloud"]["calipers_mm"]), n_cal_before + 1)

            voxels_before = snap["cloud"]["n_voxels"]
            snap = sess.handle_action("snapshot")
            self.assertEqual(snap["phase"], "placed")
            self.assertGreater(snap["cloud"]["n_voxels"], voxels_before)

            # 2. Re-picking in PLACED (`phase` / Peace sign) must enter CARRYING from placed1_xy
            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "carrying")

            # Shift right wrist by (+0.14, -0.08) m and place down at spot #2
            assert sess._last_frame_ws is not None and sess._last_frame_ws.right is not None
            delta_xy = np.array([0.14, -0.08, 0.0])
            sess._last_frame_ws.right.wrist_pos = sess._last_frame_ws.right.wrist_pos + delta_xy
            for k in sess._last_frame_ws.right.tips_pad:
                sess._last_frame_ws.right.tips_pad[k] = sess._last_frame_ws.right.tips_pad[k] + delta_xy

            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "placed")
            placed2_xy = np.array([
                snap["tracked_object"]["pose"]["x_m"],
                snap["tracked_object"]["pose"]["y_m"],
            ])
            expected2_xy = placed1_xy + delta_xy[:2]
            np.testing.assert_allclose(placed2_xy, expected2_xy, atol=0.015)

            # Cloud and saved Unitree G1 plan must reflect spot #2
            cloud_pts2 = np.asarray(snap["cloud"]["points"])
            np.testing.assert_allclose(cloud_pts2[:, :2].mean(axis=0), placed2_xy, atol=0.025)

            # 3. Also verify Left-Hand Peace gesture in tick() re-enters CARRYING from PLACED
            sess.synthetic.manual_gesture = "idle"
            for _ in range(30):  # clear 0.80s cooldown
                snap = sess.tick(dt=dt)
            self.assertEqual(snap["phase"], "placed")

            sess.synthetic.manual_gesture = "peace"
            for _ in range(12):  # hold > 0.25s debounce + settle
                snap = sess.tick(dt=dt)
            self.assertEqual(snap["phase"], "carrying")

            sess.synthetic.manual_gesture = "idle"
            for _ in range(30):  # clear 0.80s cooldown
                snap = sess.tick(dt=dt)
            sess.synthetic.manual_gesture = "peace"
            for _ in range(12):
                snap = sess.tick(dt=dt)
            self.assertEqual(snap["phase"], "placed")
            sess.close()

    def test_reference_pose_fit_ignores_stray_snapshot_fingers_at_coil_offset(self):
        # Model real Coil Pro offset (-0.63 m, +0.19 m) with a Red Bull rim trace + 5 stray outer fingers
        true_xy = np.array([-0.629, 0.189])
        redbull = OBJECTS["redbull"]
        R_can = float(redbull.dims["radius"])
        angles = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
        rim_pts = np.column_stack([
            true_xy[0] + R_can * np.cos(angles),
            true_xy[1] + R_can * np.sin(angles),
            np.full_like(angles, 0.118),
        ])
        stray_fingers = np.array([
            [-0.703, 0.188, 0.087],
            [-0.696, 0.194, 0.081],
            [-0.690, 0.178, 0.100],
            [-0.680, 0.182, 0.106],
            [-0.650, 0.117, 0.087],
        ])
        pts = np.vstack([rim_pts, stray_fingers])
        obb = minimum_area_obb_2d(pts[:, :2])
        raw_err_mm = float(np.linalg.norm(obb["center_xy"] - true_xy)) * 1000.0
        self.assertGreater(raw_err_mm, 14.0)

        fitted_xy, _ = fit_reference_pose_xy_yaw(
            pts, redbull.shape, redbull.dims, z_table=0.0, init_xy=obb["center_xy"], init_yaw=obb["yaw_rad"]
        )
        fit_err_mm = float(np.linalg.norm(fitted_xy - true_xy)) * 1000.0
        self.assertLess(fit_err_mm, 2.5)

    def test_confirmed_90_metadata_and_movable_dashboard_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            for key in ("redbull", "kantolan_suola_cracker", "wood_sphere"):
                sess = ScanTrackSession(source="synthetic", synthetic_object=key, sessions_root=tmp)
                assert sess.synthetic is not None
                sess.synthetic.auto_cycle = False
                dt = 1.0 / 30.0
                snap = {}
                # Run through 7.5 s of scanning (still in `scanning` phase before 8.2 s LOCK_GRASP!)
                for _ in range(int(7.5 / dt)):
                    snap = sess.tick(dt=dt)
                sess.close()

                self.assertEqual(snap["phase"], "scanning")
                self.assertTrue(snap["catalog"]["confirmed_90"], msg=f"{key} should exceed 90% in scanning")
                self.assertIsNotNone(snap["tracked_object"])
                tobj = snap["tracked_object"]
                self.assertTrue(tobj["confirmed_90"])
                self.assertGreater(tobj["confidence"], 0.90)
                # Confirmed structure in scanning must sit flush on z_table = 0.0
                self.assertAlmostEqual(tobj["pose"]["z_bottom_m"], 0.0, places=4)
                # Entire state snapshot must serialize cleanly to JSON for /api/state
                encoded = json.dumps(snap)
                self.assertGreater(len(encoded), 200)

        self.assertIn("3D Space / X-Z Plane", INDEX_HTML)
        self.assertIn("renderConfirmedStructure", INDEX_HTML)
        self.assertIn("centerOnActiveScene", INDEX_HTML)
        self.assertIn('class="btn-stack"', INDEX_HTML)
        self.assertIn('id="btn-obj-vis"', INDEX_HTML)
        self.assertIn('id="btn-hull-vis"', INDEX_HTML)
        self.assertIn('id="btn-pts-vis"', INDEX_HTML)
        self.assertLess(INDEX_HTML.index('id="btn-obj-vis"'), INDEX_HTML.index('id="btn-hull-vis"'))
        self.assertLess(INDEX_HTML.index('id="btn-hull-vis"'), INDEX_HTML.index('id="btn-pts-vis"'))
        self.assertLess(INDEX_HTML.index('id="btn-pts-vis"'), INDEX_HTML.index('id="confirmed-banner"'))
        self.assertIn("toggleObjectVisMode()", INDEX_HTML)
        self.assertIn("toggleHullVisMode()", INDEX_HTML)
        self.assertIn("togglePointsVis()", INDEX_HTML)
        self.assertIn("Object: Visible", INDEX_HTML)
        self.assertIn("Object: See-Through", INDEX_HTML)
        self.assertIn("Hull: Visible", INDEX_HTML)
        self.assertIn("Hull: See-Through", INDEX_HTML)
        self.assertIn("Points: Visible", INDEX_HTML)
        self.assertIn("Points: Hidden", INDEX_HTML)
        self.assertNotIn('id="btn-obj-solid"', INDEX_HTML)
        self.assertNotIn('id="btn-obj-see-through"', INDEX_HTML)
        self.assertNotIn('id="btn-drag-mode"', INDEX_HTML)
        self.assertNotIn('resetCameraView()"', INDEX_HTML)
        self.assertNotIn('class="primary" id="btn-phase"', INDEX_HTML)
        self.assertNotIn('id="g1-summary"', INDEX_HTML)
        self.assertNotIn("MuJoCo Twin", INDEX_HTML)

    def test_cannot_add_points_during_pickup_carrying(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="redbull", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            dt = 1.0 / 30.0

            # Scan just a few initial points, return to Open Hand (`idle`), then signal Peace to pick up
            sess.synthetic.manual_gesture = "pinch"
            for _ in range(10):
                snap = sess.tick(dt=dt)
            self.assertEqual(snap["phase"], "scanning")

            sess.synthetic.manual_gesture = "idle"
            for _ in range(4):
                snap = sess.tick(dt=dt)

            sess.synthetic.manual_gesture = "peace"
            for _ in range(8):
                snap = sess.tick(dt=dt)
            self.assertEqual(snap["phase"], "carrying")
            self.assertTrue(sess.cloud.locked)

            n_raw_locked = snap["cloud"]["n_raw"]
            n_vox_locked = snap["cloud"]["n_voxels"]
            n_cal_locked = len(snap["cloud"]["calipers_mm"])

            # 1. Left-hand Pinch, Fist, and Caliper gestures during CARRYING must add ZERO points
            for gest in ("pinch", "idle", "fist", "idle"):
                sess.synthetic.manual_gesture = gest
                for _ in range(15):
                    snap = sess.tick(dt=dt)
                    self.assertEqual(snap["phase"], "carrying")
                    self.assertFalse(snap["gesture"]["stream_index"])
                    self.assertEqual(snap["cloud"]["n_raw"], n_raw_locked)
                    self.assertEqual(snap["cloud"]["n_voxels"], n_vox_locked)
                    self.assertEqual(len(snap["cloud"]["calipers_mm"]), n_cal_locked)

            # 2. Manual UI / keyboard commands during CARRYING must also add ZERO points & not queue
            for cmd in ("trace_toggle", "trace_on", "snapshot", "caliper"):
                snap = sess.handle_action(cmd)
                self.assertEqual(snap["phase"], "carrying")
                self.assertFalse(snap["gesture"]["stream_index"])
                self.assertEqual(snap["cloud"]["n_raw"], n_raw_locked)
                self.assertEqual(snap["cloud"]["n_voxels"], n_vox_locked)
                self.assertEqual(len(snap["cloud"]["calipers_mm"]), n_cal_locked)

            # 3. Direct cloud guard must also reject points while locked
            self.assertEqual(sess.cloud.add_points(np.array([[0.12, 0.28, 0.09]])), 0)
            sess.cloud.add_caliper_width(0.055)
            self.assertEqual(len(sess.cloud.calipers_m), n_cal_locked)

            # 4. Placing object down must NOT fire any stale/queued scan command or transition glitch
            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "placed")
            self.assertFalse(sess.cloud.locked)
            sess.synthetic.manual_gesture = "pinch"  # within 0.45s release guard right after PLACED
            snap = sess.tick(dt=dt)
            self.assertEqual(snap["cloud"]["n_raw"], n_raw_locked)
            sess.close()

    def test_pickup_blocked_when_zero_voxels_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="redbull", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            sess.synthetic.manual_gesture = "idle"
            dt = 1.0 / 30.0
            for _ in range(4):
                snap = sess.tick(dt=dt)
            self.assertEqual(snap["cloud"]["n_voxels"], 0)
            self.assertEqual(snap["phase"], "scanning")

            # UI / Spacebar phase command with 0 voxels must stay in scanning
            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "scanning")
            self.assertFalse(sess.cloud.locked)
            self.assertIsNone(snap["tracked_object"])
            self.assertIsNone(snap["catalog"]["best_key"])

            # Left-hand Peace gesture with 0 voxels must also stay in scanning
            sess.synthetic.manual_gesture = "peace"
            for _ in range(8):
                snap = sess.tick(dt=dt)
            self.assertEqual(snap["phase"], "scanning")
            self.assertFalse(sess.cloud.locked)
            self.assertIsNone(snap["tracked_object"])
            sess.close()

    def test_unmatched_pickup_never_assigns_redbull_or_rendered_shape(self):
        # 1. Sparse early scan (<10 voxels) picked up & placed must NOT lock Red Bull or render a 3D mesh
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="kantolan_suola_cracker", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            dt = 1.0 / 30.0

            sess.synthetic.manual_gesture = "pinch"
            for _ in range(8):
                snap = sess.tick(dt=dt)
            self.assertGreaterEqual(snap["cloud"]["n_voxels"], 1)
            self.assertLess(snap["cloud"]["n_voxels"], 10)

            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "carrying")
            self.assertIsNone(snap["tracked_object"])
            self.assertIsNone(snap["catalog"]["best_key"])
            self.assertFalse(snap["catalog"]["confirmed_90"])

            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "placed")
            self.assertIsNone(snap["tracked_object"])
            self.assertIsNone(snap["catalog"]["best_key"])
            self.assertFalse(snap["catalog"]["confirmed_90"])
            self.assertIsNone(snap["object_card"])
            self.assertIsNone(snap["g1_plan"])
            self.assertTrue((sess.session_dir / "cloud.ply").exists())
            self.assertFalse((sess.session_dir / "object_card.json").exists())
            self.assertFalse((sess.session_dir / "unitree_g1_pick.json").exists())
            sess.close()

        # 2. Dense scan (>=25 voxels) of an out-of-catalog shape (45x45x50 mm cube) must remain unmatched
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="redbull", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            sess.synthetic.manual_gesture = "idle"
            sess.tick(dt=1.0 / 30.0)

            rng = np.random.default_rng(99)
            cx, cy = 0.08, 0.34
            # Generate surface points on a small 45x45x50 mm cube sitting on z=0
            cube_pts = []
            for z in np.linspace(0.008, 0.050, 6):
                for u in np.linspace(-0.0225, 0.0225, 5):
                    cube_pts.append([cx + u, cy - 0.0225, z])
                    cube_pts.append([cx + u, cy + 0.0225, z])
                    cube_pts.append([cx - 0.0225, cy + u, z])
                    cube_pts.append([cx + 0.0225, cy + u, z])
            for ux in np.linspace(-0.020, 0.020, 4):
                for uy in np.linspace(-0.020, 0.020, 4):
                    cube_pts.append([cx + ux, cy + uy, 0.050])
            pts_arr = np.asarray(cube_pts, dtype=float) + rng.normal(0.0, 0.0008, size=(len(cube_pts), 3))
            sess.cloud.add_points(pts_arr)
            sess.cloud.add_caliper_width(0.045)
            sess._refresh_scan_and_match_unlocked()

            self.assertGreaterEqual(sess._last_scan.n_voxels, 25)
            self.assertIsNone(sess._last_match.best_key)
            self.assertFalse(sess._last_match.confident)
            self.assertLess(max(sess._last_match.probabilities.values()), 0.50)

            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "carrying")
            self.assertIsNone(snap["tracked_object"])
            self.assertIsNone(snap["catalog"]["best_key"])

            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "placed")
            self.assertIsNone(snap["tracked_object"])
            self.assertIsNone(snap["catalog"]["best_key"])
            self.assertIsNone(snap["object_card"])
            self.assertIsNone(snap["g1_plan"])
            sess.close()

        # Dashboard JS must not synthesize a custom_<primitive> 3D mesh when tracked_object is null
        self.assertNotIn("custom_", INDEX_HTML)
        self.assertIn("const showSolidStructure = Boolean(tracked && isConfirmed90);", INDEX_HTML)

    def test_unmatched_placed_shape_locks_once_confirmed_in_placed_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="wood_sphere", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            dt = 1.0 / 30.0

            # Pick up after a tiny 6-frame trace (unmatched) and place immediately
            sess.synthetic.manual_gesture = "pinch"
            for _ in range(6):
                sess.tick(dt=dt)
            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "carrying")
            self.assertIsNone(snap["tracked_object"])

            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "placed")
            self.assertIsNone(snap["tracked_object"])
            self.assertIsNone(snap["object_card"])

            # Wait past 0.45s placement release guard, then complete scanning the placed wood_sphere (R=35mm)
            sess.synthetic.manual_gesture = "idle"
            for _ in range(20):
                sess.tick(dt=dt)

            assert sess.relocation._current_pose is not None
            placed_xy = sess.relocation._current_pose.position_m[:2].copy()
            r = OBJECTS["wood_sphere"].dims["radius"]
            sph_pts = []
            for phi in np.linspace(-0.35 * math.pi, 0.48 * math.pi, 7):
                z = r + r * math.sin(phi)
                if z < 0.008:
                    continue
                rc = r * math.cos(phi)
                for th in np.linspace(0.0, 2.0 * math.pi, 10, endpoint=False):
                    sph_pts.append([placed_xy[0] + rc * math.cos(th), placed_xy[1] + rc * math.sin(th), z])
            sess.cloud.add_points(np.asarray(sph_pts, dtype=float))
            sess.cloud.add_caliper_width(2.0 * r)

            # Place 5 fingertip pads on the placed sphere surface and trigger discrete tactile refresh in `placed` phase
            assert sess._last_frame_ws is not None and sess._last_frame_ws.right is not None
            for idx_f, fname in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                ang = 2.0 * math.pi * idx_f / 5.0
                sess._last_frame_ws.right.tips_pad[fname] = np.array(
                    [placed_xy[0] + r * math.cos(ang), placed_xy[1] + r * math.sin(ang), r],
                    dtype=float,
                )
            snap = sess.handle_action("snapshot")
            self.assertEqual(snap["phase"], "placed")
            self.assertTrue(snap["catalog"]["confirmed_90"])
            self.assertEqual(snap["catalog"]["best_key"], "wood_sphere")
            self.assertIsNotNone(snap["tracked_object"])
            self.assertEqual(snap["tracked_object"]["key"], "wood_sphere")
            self.assertIsNotNone(snap["object_card"])
            self.assertIsNotNone(snap["g1_plan"])
            self.assertTrue((sess.session_dir / "object_card.json").exists())
            self.assertTrue((sess.session_dir / "unitree_g1_pick.json").exists())
            sess.close()

    def test_undo_action_restores_scan_carrying_placed_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="kantolan_suola_cracker", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            dt = 1.0 / 30.0

            # Stroke 1: pinch trace adds points
            sess.synthetic.manual_gesture = "pinch"
            for _ in range(15):
                snap = sess.tick(dt=dt)
            v1 = snap["cloud"]["n_voxels"]
            self.assertGreater(v1, 0)

            # Open hand, then discrete 5-finger snapshot adds more points
            sess.synthetic.manual_gesture = "idle"
            for _ in range(4):
                sess.tick(dt=dt)
            snap = sess.handle_action("snapshot")
            v2 = snap["cloud"]["n_voxels"]
            self.assertGreaterEqual(v2, v1)

            # Lock grasp -> carrying (locks cloud), then place -> placed
            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "carrying")
            self.assertTrue(sess.cloud.locked)
            snap = sess.handle_action("phase")
            self.assertEqual(snap["phase"], "placed")
            self.assertFalse(sess.cloud.locked)

            # Reset clears voxels, Undo restores placed state
            snap = sess.handle_action("reset")
            self.assertEqual(snap["cloud"]["n_voxels"], 0)
            snap = sess.handle_action("undo")
            self.assertEqual(snap["phase"], "placed")
            self.assertEqual(snap["cloud"]["n_voxels"], v2)

            # Undo placement -> back to carrying (cloud locked)
            snap = sess.handle_action("undo")
            self.assertEqual(snap["phase"], "carrying")
            self.assertTrue(sess.cloud.locked)

            # Undo lock-grasp -> back to scanning (cloud unlocked so scanning can resume!)
            snap = sess.handle_action("undo")
            self.assertEqual(snap["phase"], "scanning")
            self.assertFalse(sess.cloud.locked)
            self.assertEqual(snap["cloud"]["n_voxels"], v2)

            # Undo snapshot -> back to Stroke 1 voxel count
            snap = sess.handle_action("undo")
            self.assertEqual(snap["cloud"]["n_voxels"], v1)

            # Undo Stroke 1 -> back to 0 voxels
            snap = sess.handle_action("undo")
            self.assertEqual(snap["cloud"]["n_voxels"], 0)
            sess.close()

        self.assertIn("[U]</b> Undo", INDEX_HTML)
        self.assertIn("showUndoToast", INDEX_HTML)

    def test_confidence_starts_at_zero_not_33_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="redbull", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            sess.synthetic.manual_gesture = "idle"

            # 1. At startup (0 voxels), every catalog probability and primitive BIC weight must be 0.0 (never 33%)
            snap0 = sess.tick(dt=1.0 / 30.0)
            self.assertEqual(snap0["cloud"]["n_voxels"], 0)
            for k, prob in snap0["catalog"]["probabilities"].items():
                self.assertEqual(prob, 0.0, msg=f"{k} initial probability must be 0.0, got {prob}")
            for prim, w in snap0["cloud"]["primitive_weights"].items():
                self.assertEqual(w, 0.0, msg=f"{prim} initial BIC weight must be 0.0, got {w}")
            self.assertEqual(snap0["catalog"]["best_probability"], 0.0)
            self.assertIsNone(snap0["catalog"]["best_key"])
            self.assertFalse(snap0["catalog"]["confirmed_90"])
            self.assertIsNone(snap0["tracked_object"])

            # 2. For 1..3 voxels (before 3D geometry exists), catalog probabilities remain strictly 0.0
            pts_3 = [
                [0.05, 0.30, 0.02],
                [0.08, 0.30, 0.05],
                [0.05, 0.34, 0.08],
            ]
            for idx, pt in enumerate(pts_3, start=1):
                sess.cloud.add_points(np.array([pt], dtype=float))
                sess._refresh_scan_and_match_unlocked()
                snap_k = sess._snapshot_unlocked()
                self.assertEqual(snap_k["cloud"]["n_voxels"], idx)
                for k, prob in snap_k["catalog"]["probabilities"].items():
                    self.assertEqual(prob, 0.0, msg=f"{k} probability at {idx} voxels must be 0.0")
                for prim, w in snap_k["cloud"]["primitive_weights"].items():
                    self.assertEqual(w, 0.0, msg=f"{prim} weight at {idx} voxels (<7) must be 0.0")

            # 3. Reset after a >90% scan also returns all probabilities & primitive weights strictly to 0.0
            sess.synthetic.manual_gesture = None
            dt = 1.0 / 30.0
            for _ in range(int(7.5 / dt)):
                snap_hi = sess.tick(dt=dt)
            self.assertTrue(snap_hi["catalog"]["confirmed_90"])
            snap_reset = sess.handle_action("reset")
            for k, prob in snap_reset["catalog"]["probabilities"].items():
                self.assertEqual(prob, 0.0)
            for prim, w in snap_reset["cloud"]["primitive_weights"].items():
                self.assertEqual(w, 0.0)
            self.assertIsNone(snap_reset["tracked_object"])
            sess.close()

    def test_object_stops_rendering_immediately_when_confidence_falls_below_90(self):
        # 1. In `scanning`: once >90% is reached, any new scan points dropping confidence <=90% stop rendering immediately
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="redbull", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            dt = 1.0 / 30.0
            snap = {}
            for _ in range(int(7.5 / dt)):
                snap = sess.tick(dt=dt)

            self.assertEqual(snap["phase"], "scanning")
            self.assertTrue(snap["catalog"]["confirmed_90"])
            self.assertGreater(snap["catalog"]["probabilities"]["redbull"], 0.90)
            self.assertIsNotNone(snap["tracked_object"])
            self.assertEqual(snap["tracked_object"]["key"], "redbull")

            # Check borderline 80%..90% drop first (must NOT keep rendering via hysteresis!)
            cx, cy = sess._last_scan.center_xy
            saw_drop_below_90 = False
            for extra_z in np.linspace(0.142, 0.195, 12):
                sess.cloud.add_points(
                    np.array(
                        [
                            [cx - 0.025, cy, float(extra_z)],
                            [cx + 0.025, cy, float(extra_z)],
                        ],
                        dtype=float,
                    )
                )
                sess._refresh_scan_and_match_unlocked()
                snap_step = sess._snapshot_unlocked()
                p_rb = snap_step["catalog"]["probabilities"]["redbull"]
                if p_rb <= 0.90:
                    saw_drop_below_90 = True
                    self.assertFalse(
                        snap_step["catalog"]["confirmed_90"],
                        msg=f"confirmed_90 must clear immediately at {p_rb:.3f} (<= 0.90)",
                    )
                    self.assertIsNone(
                        snap_step["tracked_object"],
                        msg=f"tracked_object must stop rendering immediately at {p_rb:.3f} (<= 0.90)",
                    )
            self.assertTrue(saw_drop_below_90)
            sess.close()

        # 2. Snapshot in `scanning` dropping >90% -> <=90% stops rendering, and Undo restores >90% render
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="redbull", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            dt = 1.0 / 30.0
            for _ in range(int(7.5 / dt)):
                snap = sess.tick(dt=dt)
            self.assertTrue(snap["catalog"]["confirmed_90"])
            self.assertIsNotNone(snap["tracked_object"])

            assert sess._last_frame_ws is not None and sess._last_frame_ws.right is not None
            cx, cy = sess._last_scan.center_xy
            for idx_f, fname in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                sess._last_frame_ws.right.tips_pad[fname] = np.array(
                    [cx + 0.04 * (idx_f - 2), cy + 0.03, 0.190], dtype=float
                )
            snap_drop = sess.handle_action("snapshot")
            self.assertLessEqual(snap_drop["catalog"]["probabilities"]["redbull"], 0.90)
            self.assertFalse(snap_drop["catalog"]["confirmed_90"])
            self.assertIsNone(snap_drop["tracked_object"])

            snap_undo = sess.handle_action("undo")
            self.assertTrue(snap_undo["catalog"]["confirmed_90"])
            self.assertIsNotNone(snap_undo["tracked_object"])
            self.assertEqual(snap_undo["tracked_object"]["key"], "redbull")
            sess.close()

        # 3. In `placed`: further scanning that drops confidence <=90% clears rendered object & stale card/G1 plan
        with tempfile.TemporaryDirectory() as tmp:
            sess = ScanTrackSession(source="synthetic", synthetic_object="redbull", sessions_root=tmp)
            assert sess.synthetic is not None
            sess.synthetic.auto_cycle = False
            dt = 1.0 / 30.0
            for _ in range(int(13.5 / dt)):
                snap = sess.tick(dt=dt)
            self.assertEqual(snap["phase"], "placed")
            self.assertTrue(snap["catalog"]["confirmed_90"])
            self.assertIsNotNone(snap["tracked_object"])
            self.assertTrue((sess.session_dir / "object_card.json").exists())
            self.assertTrue((sess.session_dir / "unitree_g1_pick.json").exists())

            sess.synthetic.manual_gesture = "idle"
            for _ in range(20):
                sess.tick(dt=dt)

            assert sess._last_frame_ws is not None and sess._last_frame_ws.right is not None
            px = snap["tracked_object"]["pose"]["x_m"]
            py = snap["tracked_object"]["pose"]["y_m"]
            for idx_f, fname in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                sess._last_frame_ws.right.tips_pad[fname] = np.array(
                    [px + 0.045 * (idx_f - 2), py + 0.04, 0.195], dtype=float
                )
            snap_placed_drop = sess.handle_action("snapshot")
            self.assertEqual(snap_placed_drop["phase"], "placed")
            self.assertLessEqual(snap_placed_drop["catalog"]["probabilities"]["redbull"], 0.90)
            self.assertFalse(snap_placed_drop["catalog"]["confirmed_90"])
            self.assertIsNone(snap_placed_drop["tracked_object"])
            self.assertIsNone(snap_placed_drop["object_card"])
            self.assertIsNone(snap_placed_drop["g1_plan"])
            self.assertFalse((sess.session_dir / "object_card.json").exists())
            self.assertFalse((sess.session_dir / "unitree_g1_pick.json").exists())
            sess.close()

        self.assertNotIn("drawPreviewWireframe", INDEX_HTML)
        self.assertIn("const isConfirmed90 = Boolean(tracked && tracked.confirmed_90 && conf > 0.90);", INDEX_HTML)


class UndoBehaviourTests(unittest.TestCase):
    def _session(self, tmp):
        sess = ScanTrackSession(source="synthetic", synthetic_object="redbull", sessions_root=tmp)
        assert sess.synthetic is not None
        sess.synthetic.auto_cycle = False
        return sess

    def _tick_until(self, sess, phase, limit_s=20.0):
        dt = 1.0 / 30.0
        snap = {}
        for _ in range(int(limit_s / dt)):
            snap = sess.tick(dt=dt)
            if sess.relocation.phase == phase:
                return snap
        self.fail(f"never reached phase {phase!r}")

    def test_undo_is_blocked_while_carrying(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)
            self._tick_until(sess, "carrying")
            depth = len(sess._undo_stack)
            snap = sess.handle_action("undo")
            sess.close()
            self.assertEqual(snap["phase"], "carrying")
            self.assertEqual(len(sess._undo_stack), depth)

    def test_undo_place_returns_to_state_before_pickup(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)
            self._tick_until(sess, "carrying")
            # State right after the pick-up gesture started: the stack top is the pre-pickup snapshot.
            before = sess._undo_stack[-1]
            before_phase = before["relocation"].phase
            before_voxels = before["cloud"].summarize().n_voxels
            self._tick_until(sess, "placed")
            snap = sess.handle_action("undo")
            sess.close()
            self.assertNotEqual(snap["phase"], "carrying")
            self.assertEqual(snap["phase"], before_phase)
            self.assertEqual(sess._last_scan.n_voxels, before_voxels)


if __name__ == "__main__":
    unittest.main()
