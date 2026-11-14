"""Shape fitting on synthetic touch sessions (no MuJoCo): fast checks of the math."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from glove_twin import shape_fit as sf
from glove_twin.objects import OBJECTS


class GeometryTests(unittest.TestCase):
    def test_rotation_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            w = rng.normal(size=3)
            w *= rng.uniform(0.01, 3.0) / np.linalg.norm(w)
            np.testing.assert_allclose(sf.rodrigues(sf.rot_to_vec(sf.rodrigues(w))), sf.rodrigues(w), atol=1e-9)

    def test_signed_distances(self):
        x = np.array([[0.06, 0, 0], [0, 0, 0], [0, 0, 0.08]])
        np.testing.assert_allclose(sf.sdf("sphere", np.array([0.05]), x), [0.01, -0.05, 0.03], atol=1e-12)
        np.testing.assert_allclose(sf.sdf("cylinder", np.array([0.02, 0.1]), x), [0.04, -0.02, 0.03], atol=1e-12)
        np.testing.assert_allclose(sf.sdf("box", np.array([0.04, 0.04, 0.1]), x), [0.04, -0.02, 0.03], atol=1e-12)

    def test_24_cube_rotations(self):
        self.assertEqual(len(sf.SIGNED_PERMS), 24)
        for P in sf.SIGNED_PERMS:
            self.assertAlmostEqual(np.linalg.det(P), 1.0)


class FitAccuracyTests(unittest.TestCase):
    """Plan targets: >= 90 % correct shape over 10 seeds per object; dimensions within 5-10 mm.

    Synthetic noise: 1.6 mm per contact (Rokoko finger accuracy), 20 deg normal error, 1 mm bias.
    """

    def test_model_objects(self):
        for key in ("redbull", "kantolan_suola_cracker", "wood_sphere"):
            o = OBJECTS[key]
            dims = o.extents() if o.shape == "box" else o.dims_vector()
            hits, worst = 0, 0.0
            for seed in range(10):
                res = sf.fit_shape(sf.synthetic_session(o.shape, dims, np.random.default_rng(seed)))
                if res.best == o.shape:
                    hits += 1
                    worst = max(worst, float(np.abs(np.array(list(res.dims.values())) - o.dims_vector()).max()))
            self.assertGreaterEqual(hits, 9, key)
            self.assertLess(worst, 0.010, key)

    def test_box_labels_do_not_matter(self):
        """The same box given with its extents in a different order fits to the same sorted dims."""
        a = sf.fit_shape(sf.synthetic_session("box", [0.0478, 0.0475, 0.140], np.random.default_rng(3)), models=("box",))
        b = sf.fit_shape(sf.synthetic_session("box", [0.140, 0.0478, 0.0475], np.random.default_rng(3)), models=("box",))
        np.testing.assert_allclose(sorted(a.dims.values()), sorted(b.dims.values()), atol=0.008)

    def test_short_cylinder_without_end_caps_is_flagged(self):
        g = sf.synthetic_session("cylinder", [0.04, 0.05], np.random.default_rng(1), end_caps=False)
        res = sf.fit_shape(g)
        self.assertTrue(res.ambiguous)

    def test_caliper_distance_is_pose_independent(self):
        g = sf.synthetic_session("box", [0.05, 0.06, 0.14], np.random.default_rng(0), noise_p=0.0,
                                 noise_n_deg=0.0, bias_m=0.0)
        np.testing.assert_allclose(sorted(sf.caliper_distances(g)), [0.05, 0.06, 0.14], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
