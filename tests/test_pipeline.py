"""Run with:  python -m unittest discover -s tests -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from glove_twin import MATERIALS, TapClassifier, TapEvent, estimate_density, extract_features, parse_packet, simulate_tap
from glove_twin.classifier import load_samples, save_samples
from glove_twin.twin import build_properties, write_twin


def clean_ring_tap(freq=2000.0, q=300.0, fs=16000, n=1024, pre=64) -> TapEvent:
    """A single noiseless mode with known frequency and Q."""
    t = np.arange(n - pre) / fs
    tau = q / (np.pi * freq)
    tc = 0.0005
    force = 2.0 * np.where(t < tc, np.sin(np.pi * t / tc), 0.0)
    mic = 0.5 * np.exp(-t / tau) * np.sin(2 * np.pi * freq * t)
    pad = lambda x: np.concatenate([np.zeros(pre), x]) + np.random.default_rng(0).normal(0, 1e-4, n)
    return TapEvent(fs, pad(np.gradient(force) * 0.1), np.maximum(pad(force), 0), pad(mic), pre)


class FeatureTests(unittest.TestCase):
    def test_recovers_frequency_and_q(self):
        f = extract_features(clean_ring_tap(freq=2000.0, q=300.0))
        self.assertAlmostEqual(f.ring_hz, 2000.0, delta=40.0)
        self.assertGreater(f.q_factor, 300.0 * 0.7)
        self.assertLess(f.q_factor, 300.0 * 1.3)

    def test_contact_time_separates_hard_and_soft(self):
        rng = np.random.default_rng(3)
        steel = np.median([extract_features(simulate_tap("steel", rng)).contact_ms for _ in range(10)])
        foam = np.median([extract_features(simulate_tap("foam", rng)).contact_ms for _ in range(10)])
        self.assertGreater(foam, 3 * steel)

    def test_dead_object_has_minimal_q(self):
        f = extract_features(simulate_tap("foam", np.random.default_rng(1)))
        self.assertLess(f.q_factor, 20.0)

    def test_no_mic_means_no_ring_features(self):
        f = extract_features(simulate_tap("wood", np.random.default_rng(1), with_mic=False))
        self.assertFalse(f.has_ring)
        with self.assertRaises(ValueError):
            f.vector(use_ring=True)


class ClassifierTests(unittest.TestCase):
    def test_held_out_accuracy_on_simulated_taps(self):
        clf = TapClassifier.synthetic_prior(taps_per_material=30, seed=0)
        rng = np.random.default_rng(99)
        hits = total = 0
        for name in MATERIALS:
            for _ in range(10):
                probs = clf.predict_proba(extract_features(simulate_tap(name, rng)))
                hits += max(probs, key=probs.get) == name
                total += 1
        self.assertGreater(hits / total, 0.85)

    def test_probabilities_are_normalised_and_never_zero(self):
        clf = TapClassifier.synthetic_prior(taps_per_material=10, seed=0)
        feats = [extract_features(simulate_tap("glass", np.random.default_rng(i))) for i in range(5)]
        probs = clf.predict_session(feats)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertEqual(next(iter(probs)), "glass")

    def test_calibration_round_trip(self):
        rng = np.random.default_rng(0)
        samples = [(extract_features(simulate_tap(m, rng)), m) for m in ("wood", "steel") for _ in range(3)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cal.json"
            save_samples(path, samples)
            loaded = load_samples(path)
        self.assertEqual([m for _, m in loaded], [m for _, m in samples])
        self.assertAlmostEqual(loaded[0][0].contact_ms, samples[0][0].contact_ms)


class DensityTests(unittest.TestCase):
    def test_material_prior_only(self):
        est = estimate_density({"steel": 1.0})
        self.assertEqual(est.source, "material prior")
        self.assertAlmostEqual(est.value, 7850, delta=1)
        self.assertIsNone(est.solidity)

    def test_hollow_mug_detected_from_mass_and_volume(self):
        est = estimate_density({"ceramic": 1.0}, mass_g=350, volume_cm3=810)
        self.assertAlmostEqual(est.value, 432.1, delta=0.5)
        self.assertLess(est.solidity, 0.3)
        self.assertIn("hollow", est.note)

    def test_solid_block(self):
        est = estimate_density({"wood": 1.0}, mass_g=650, volume_cm3=1000)
        self.assertIn("solid", est.note)


class PacketTests(unittest.TestCase):
    def test_parse_firmware_packet(self):
        line = json.dumps({"fs": 16000, "pre": 2, "adc_bits": 12,
                           "piezo": [2047, 2048, 3000, 1000], "fsr": [0, 0, 2000, 1500],
                           "mic": [2048, 2048, 2500, 1600]})
        tap = parse_packet(line)
        self.assertEqual(tap.fs, 16000)
        self.assertEqual(tap.pretrigger, 2)
        self.assertAlmostEqual(tap.piezo[1], 0.0, delta=0.01)
        self.assertLess(tap.force_n[0], 0.01)
        self.assertGreater(tap.force_n[2], tap.force_n[3])
        self.assertIsNotNone(tap.mic)

    def test_status_lines_are_ignored(self):
        self.assertIsNone(parse_packet("tap_glove ready, baseline 2051"))
        self.assertIsNone(parse_packet('{"status": "ok"}'))


class TwinTests(unittest.TestCase):
    def _write(self, d, mass_g=350.0):
        probs = {"ceramic": 0.9, "glass": 0.1}
        feats = [extract_features(simulate_tap("ceramic", np.random.default_rng(i))) for i in range(3)]
        dens = estimate_density(probs, mass_g, 810.0)
        props = build_properties("mug", probs, feats, dens, classifier_source="test",
                                 mass_g=mass_g, dims_m=(0.09, 0.09, 0.10), volume_cm3=810.0)
        return props, write_twin(props, d)

    def test_files_and_hints(self):
        with tempfile.TemporaryDirectory() as d:
            props, paths = self._write(d)
            self.assertTrue(all(p.exists() for p in paths.values()))
            self.assertTrue(props["robot_hints"]["fragile"])
            self.assertEqual(props["robot_hints"]["max_grip_force_n"], 15)  # glass is plausible -> cautious
            self.assertIn('mass="0.35000"', paths["glove"].read_text())
            self.assertIn('density="1000.0"', paths["default"].read_text())

    def test_mujoco_loads_both_twins(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("mujoco not installed")
        with tempfile.TemporaryDirectory() as d:
            _, paths = self._write(d)
            glove = mujoco.MjModel.from_xml_path(str(paths["glove"]))
            default = mujoco.MjModel.from_xml_path(str(paths["default"]))
            self.assertAlmostEqual(glove.body_mass[1], 0.35, places=4)
            self.assertAlmostEqual(default.body_mass[1], 1000 * 0.09 * 0.09 * 0.10, places=4)
            data = mujoco.MjData(glove)
            for _ in range(500):  # 1 s: the object should settle on the floor, not fall through
                mujoco.mj_step(glove, data)
            self.assertGreater(data.qpos[2], 0.04)


if __name__ == "__main__":
    unittest.main()
