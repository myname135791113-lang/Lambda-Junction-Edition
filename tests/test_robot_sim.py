"""MuJoCo tests: Rokoko hand model, simulated taps, robot grasp A/B. Skipped without mujoco."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


@unittest.skipIf(mujoco is None, "mujoco not installed")
class RokokoHandTests(unittest.TestCase):
    def setUp(self):
        from glove_twin.robot_sim import SCENARIOS, tap_scene_xml
        self.model = mujoco.MjModel.from_xml_string(tap_scene_xml(SCENARIOS[0]))

    def test_openxr_joints_present(self):
        from glove_twin.hand import OPENXR_JOINTS
        self.assertEqual(len(OPENXR_JOINTS), 26)
        for name in OPENXR_JOINTS:
            found = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0 or \
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
            self.assertTrue(found, name)

    def test_rokoko_sensor_sites_on_the_documented_segments(self):
        expected = {"index_finger_sensor": "index_intermediate", "middle_finger_sensor": "middle_intermediate",
                    "ring_finger_sensor": "ring_intermediate", "little_finger_sensor": "little_intermediate",
                    "thumb_finger_sensor": "thumb_proximal", "hub": "wrist", "arm_sensor": "forearm"}
        for site, body in expected.items():
            self.assertEqual(self.model.body(int(self.model.site(site).bodyid[0])).name, body)

    def test_full_hand_has_all_finger_joints(self):
        from glove_twin.hand import hand_body, hand_defaults
        xml = (f'<mujoco><compiler angle="radian"/><default>{hand_defaults()}</default>'
               f'<worldbody>{hand_body()}</worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual(m.njnt, 4 * 4 + 4)  # 4 fingers x (MCP flex, MCP abd, PIP, DIP) + thumb x 4


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TapSimulationTests(unittest.TestCase):
    def test_contact_time_follows_material(self):
        from glove_twin.materials import MATERIALS
        from glove_twin.robot_sim import CALIBRATION_OBJECTS, TAP_FS, TAP_PRE, tap_object
        rng = np.random.default_rng(0)
        for obj in CALIBRATION_OBJECTS:
            f = tap_object(obj, rng).force_true[TAP_PRE:]
            contact_ms = int(np.argmin(f > 0)) / TAP_FS * 1000
            target = np.hypot(MATERIALS[obj.material].contact_ms, 0.3)
            self.assertAlmostEqual(contact_ms, target, delta=0.15 * target + 0.07, msg=obj.material)

    def test_rokoko_frames_miss_the_tap(self):
        from glove_twin.robot_sim import SCENARIOS, TAP_FS, TAP_PRE, tap_object
        r = tap_object(SCENARIOS[0], np.random.default_rng(1))
        contact_ms = np.count_nonzero(r.force_true[TAP_PRE:] > 0) / TAP_FS * 1000
        inside = np.count_nonzero((r.rokoko_t_ms >= 0) & (r.rokoko_t_ms <= contact_ms))
        self.assertLessEqual(inside, 1)  # 83 Hz frames are 12 ms apart; the tap lasts < 1 ms

    def test_hand_taps_identify_the_scenarios(self):
        from glove_twin.classifier import TapClassifier
        from glove_twin.features import extract_features
        from glove_twin.robot_sim import CALIBRATION_OBJECTS, SCENARIOS, tap_object
        rng = np.random.default_rng(2)
        clf = TapClassifier([(extract_features(tap_object(o, rng).tap), o.material)
                             for o in CALIBRATION_OBJECTS for _ in range(5)])
        for obj in SCENARIOS:
            probs = clf.predict_session([extract_features(tap_object(obj, rng).tap) for _ in range(4)])
            self.assertEqual(next(iter(probs)), obj.material)

    def test_no_sensor_clipping(self):
        from glove_twin.robot_sim import CALIBRATION_OBJECTS, tap_object
        rng = np.random.default_rng(3)
        for obj in CALIBRATION_OBJECTS:
            tap = tap_object(obj, rng).tap
            self.assertLess(np.abs(tap.mic).max(), 0.99, obj.material)
            self.assertLess(np.abs(tap.piezo).max(), 0.99, obj.material)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class GraspTests(unittest.TestCase):
    def test_ab_outcomes(self):
        from glove_twin.materials import DEFAULT_DENSITY, DEFAULT_FRICTION, MATERIALS
        from glove_twin.robot_sim import SCENARIOS, grasp_object, plan_grip_force
        expected_default = {"steel_cube": "dropped", "glass_vase": "crushed",
                            "cardboard_box": "crushed", "wood_block": "lifted"}
        for obj in SCENARIOS:
            m = MATERIALS[obj.material]
            default = plan_grip_force(DEFAULT_DENSITY * obj.volume_cm3 * 1e-6, DEFAULT_FRICTION)
            glove = plan_grip_force(obj.mass_kg, m.friction, m.max_grip_n)
            self.assertEqual(grasp_object(obj, default).outcome, expected_default[obj.name], obj.name)
            self.assertEqual(grasp_object(obj, glove).outcome, "lifted", obj.name)

    def test_plan_respects_fragility(self):
        from glove_twin.robot_sim import plan_grip_force
        self.assertLessEqual(plan_grip_force(5.0, 0.3, max_grip_n=15), 12.0)
        self.assertAlmostEqual(plan_grip_force(1.0, 0.5), 1.5 * 9.81)


if __name__ == "__main__":
    unittest.main()
