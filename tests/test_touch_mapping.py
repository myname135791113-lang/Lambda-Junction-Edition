"""Touch reconstruction, the simulated exploration session, the object card and the robot hand."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


class CatalogTests(unittest.TestCase):
    def test_masses_match_sources(self):
        from glove_twin.objects import OBJECTS
        expected = {"redbull": 0.270, "profeel": 0.2658, "wood_sphere": 0.323}
        for key, m in expected.items():
            self.assertAlmostEqual(OBJECTS[key].mass_kg, m, delta=0.03 * m, msg=key)

    def test_wood_density_is_average_at_12_percent_moisture(self):
        from glove_twin.objects import WOOD_DENSITY_AVG
        self.assertAlmostEqual(WOOD_DENSITY_AVG, (0.64 + 0.55 + 0.46) / 3 * 1.12 * 1000, places=6)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class ReconstructionTests(unittest.TestCase):
    def test_zero_noise_reconstruction_matches_the_scene(self):
        """touch.reconstruct (hand-only model, joint angles only) == the exploration scene's pads."""
        from glove_twin.explore_sim import Explorer
        from glove_twin.hand import ALL_JOINTS, PRESHAPES
        from glove_twin.objects import OBJECTS
        from glove_twin.touch import Frame, reconstruct
        ex = Explorer(OBJECTS["redbull"], scale=1.0)
        ex.enclosure(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 0.0, PRESHAPES["open"])
        on = ex.pressed_pads()
        self.assertGreaterEqual(len(on), 6)
        q = {j: float(ex.data.qpos[ex.qadr[j]]) for j in ALL_JOINTS}
        grasp = reconstruct([Frame(0, "enclosure", q, {p: 1.0 for p in on})])[0]
        wrist = ex.model.body("wrist").id
        Rw, pw = ex.data.xmat[wrist].reshape(3, 3), ex.data.xpos[wrist]
        for name, p in zip(grasp.segments, grasp.points):
            g = ex.pad_ids[name]
            z = ex.data.geom_xmat[g].reshape(3, 3)[:, 2]
            truth = Rw.T @ (ex.data.geom_xpos[g] - ex.model.geom_size[g][0] * z - pw)
            self.assertLess(np.linalg.norm(p - truth), 1e-4, name)

    def test_frames_round_trip(self):
        from glove_twin.touch import Frame, load_frames, save_frames
        frames = [Frame(0, "caliper", {"index_pip": 0.3}, {"index_distal_pad": 1.2}, 1.5)]
        with tempfile.TemporaryDirectory() as d:
            save_frames(Path(d) / "f.json", frames)
            self.assertEqual(load_frames(Path(d) / "f.json"), frames)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class SessionTests(unittest.TestCase):
    def test_glove_output_is_honest(self):
        """Only joint angles and pad readings leave the simulator: no hand or object pose."""
        from glove_twin.explore_sim import explore
        from glove_twin.objects import OBJECTS
        frames, _, _ = explore(OBJECTS["profeel"], np.random.default_rng(0))
        self.assertGreaterEqual(len(frames), 5)
        for f in frames:
            self.assertEqual(set(vars(f)), {"grasp_id", "kind", "joints", "fsr", "t"})

    def test_full_pipeline_each_object(self):
        from glove_twin.mapping import calibrate, map_object
        from glove_twin.objects import OBJECTS
        off = calibrate(np.random.default_rng(1000))
        for key, o in OBJECTS.items():
            res = map_object(o, np.random.default_rng(2), pad_offset=off).shape
            self.assertEqual(res.best, o.shape, key)
            for (k, v), t in zip(res.dims.items(), o.dims_vector()):
                if np.isfinite(res.sigma[k]):                 # measured dimensions only
                    self.assertLess(abs(v - t), 0.012, f"{key} {k}")


@unittest.skipIf(mujoco is None, "mujoco not installed")
class CardAndRobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from glove_twin.mapping import make_card, map_object
        from glove_twin.objects import OBJECTS
        cls.cards = {}
        off = {"enclosure": 0.002, "caliper": 0.0028}
        for key, o in OBJECTS.items():
            rng = np.random.default_rng(4)
            cls.cards[key] = make_card(o, map_object(o, rng, pad_offset=off, max_follow_ups=1), rng)

    def test_cards_are_valid_json(self):
        from glove_twin.object_card import validate_card
        for key, card in self.cards.items():
            self.assertEqual(validate_card(card), [], key)
            json.dumps(card)

    def test_container_mass_is_not_material_derived(self):
        for key in ("redbull", "profeel"):
            card = self.cards[key]
            self.assertTrue(card["container"], key)
            self.assertLess(card["mass_kg"], 0.32, key)      # aluminium-density would be ~0.79 kg

    def test_robot_lifts_with_card_force_and_drops_at_30_percent(self):
        from glove_twin.objects import OBJECTS
        from glove_twin.robot_hand import robot_grasp
        for key, card in self.cards.items():
            g = card["grasp"]
            self.assertEqual(robot_grasp(OBJECTS[key], g, g["grip_force_n"]).outcome, "lifted", key)
            self.assertEqual(robot_grasp(OBJECTS[key], g, 0.3 * g["grip_force_n"]).outcome, "dropped", key)

    def test_squeezing_above_the_limit_crushes(self):
        from glove_twin.objects import OBJECTS
        from glove_twin.robot_hand import robot_grasp
        for key in ("redbull", "profeel"):
            o = OBJECTS[key]
            self.assertEqual(robot_grasp(o, self.cards[key]["grasp"], 1.5 * o.max_grip_n).outcome, "crushed", key)


if __name__ == "__main__":
    unittest.main()
