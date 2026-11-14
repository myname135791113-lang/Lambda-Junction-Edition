"""Tests for `touchmap/touchmap_grasp.py` and the `glove_twin` robustness fixes."""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from glove_twin.density import estimate_density
from glove_twin.tap import parse_packet
from glove_twin.twin import build_properties, write_twin
from touchmap import touchmap_grasp as tm


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _sample_rokoko_v4_packet(offset_x: float = 0.0) -> dict:
    return {
        "version": 4,
        "scene": {
            "actors": [
                {
                    "name": "actor_1",
                    "body": {
                        "rightThumbTip": {"position": {"x": 0.01 + offset_x, "y": 0.10, "z": 0.30}},
                        "rightIndexTip": {"position": {"x": 0.03 + offset_x, "y": 0.12, "z": 0.32}},
                        "rightMiddleTip": {"position": {"x": 0.05 + offset_x, "y": 0.12, "z": 0.33}},
                        "rightRingTip": {"position": {"x": 0.07 + offset_x, "y": 0.11, "z": 0.32}},
                        "rightLittleTip": {"position": {"x": 0.09 + offset_x, "y": 0.09, "z": 0.30}},
                        # Decoy joint with '1' and 'tip' that used to corrupt thumb via loose '1' matching!
                        "actor1_prop_tip": {"position": {"x": 9.99, "y": 9.99, "z": 9.99}},
                    },
                }
            ]
        },
    }


class TouchmapRobustnessTests(unittest.TestCase):
    def test_nan_and_inf_points_do_not_crash_classify(self):
        rng = np.random.default_rng(42)
        obj = tm.SyntheticObject("sphere", rng=rng)
        pts = np.vstack([obj.grasp() for _ in range(4)])
        bad_rows = np.array([[np.nan, 0.0, 0.0], [np.inf, -np.inf, 10.0]])
        res = tm.classify(np.vstack([pts, bad_rows]), upright=False)
        self.assertEqual(res["n_points"], len(pts))
        self.assertIn(res["best"], tm.MODELS)

        all_nan = np.full((5, 3), np.nan)
        empty_res = tm.classify(all_nan)
        self.assertEqual(empty_res["n_points"], 0)
        self.assertIsNone(empty_res["best"])

    def test_unit_zero_vector_is_finite(self):
        u = tm.unit([0.0, 0.0, 0.0])
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0)

    def test_strict_rokoko_joint_matching_ignores_digit_decoys(self):
        pkt = _sample_rokoko_v4_packet()
        tips = tm.extract_rokoko_tips_mm(pkt, hand="right")
        self.assertEqual(tips.shape, (5, 3))
        # Thumb X should be 10 mm, NOT 9990 mm from `actor1_prop_tip`!
        self.assertAlmostEqual(float(tips[0, 0]), 10.0, places=3)

        # Duplicate conflicting joints must raise RuntimeError rather than silently keeping first match
        pkt_dup = _sample_rokoko_v4_packet()
        pkt_dup["scene"]["actors"][0]["body"]["right_thumb_tip"] = {"position": {"x": 0.2, "y": 0.2, "z": 0.2}}
        with self.assertRaises(RuntimeError):
            tm.extract_rokoko_tips_mm(pkt_dup, hand="right")

        # NaN coordinate in Rokoko packet must raise RuntimeError cleanly
        pkt_nan = _sample_rokoko_v4_packet()
        pkt_nan["scene"]["actors"][0]["body"]["rightIndexTip"]["position"]["x"] = float("nan")
        with self.assertRaises(RuntimeError):
            tm.extract_rokoko_tips_mm(pkt_nan, hand="right")

    def test_rokoko_listener_survives_malformed_udp_and_checks_staleness(self):
        port = _free_udp_port()
        rx = tm.RokokoSource(port=port, hand="right", max_age_s=0.18)
        try:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # 1. Send garbage bytes + fake LZ4 header -> listener thread MUST stay alive!
                sender.sendto(b"not-valid-json{{{", ("127.0.0.1", port))
                sender.sendto(b"\x04\x22\x4d\x18corrupt_lz4_payload", ("127.0.0.1", port))
                time.sleep(0.06)

                # 2. Send a valid Rokoko v4 frame -> should still be received cleanly!
                sender.sendto(json.dumps(_sample_rokoko_v4_packet(0.02)).encode("utf-8"), ("127.0.0.1", port))
                deadline = time.monotonic() + 1.0
                pts = None
                while time.monotonic() < deadline:
                    try:
                        pts = rx.grasp()
                        break
                    except RuntimeError:
                        time.sleep(0.02)
                self.assertIsNotNone(pts)
                assert pts is not None
                self.assertAlmostEqual(float(pts[0, 0]), 30.0, places=2)

                # 3. Wait past `max_age_s` -> stale frame check must raise RuntimeError!
                time.sleep(0.25)
                with self.assertRaises(RuntimeError) as ctx:
                    rx.grasp()
                self.assertIn("stale", str(ctx.exception))
            finally:
                sender.close()
        finally:
            rx.close()

    def test_bench_single_grasp_does_not_crash(self):
        out = tm.bench(trials=2, max_grasps=1, upright=False, tilted=False, seed=0)
        self.assertEqual(out["undecided_final"], 6)  # 1 grasp = 5 pts < 10 needed for free-pose box

    def test_session_dirs_do_not_collide_and_write_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            s1 = tm.Session(tmp, "synthetic")
            s2 = tm.Session(tmp, "synthetic")
            self.assertNotEqual(s1.dir, s2.dir)
            pts = np.zeros((5, 3))
            res = tm.classify(pts)
            s1.write([{"t": 1.0, "p": pts}], res)
            self.assertTrue((s1.dir / "grasps.jsonl").exists())
            self.assertTrue((s1.dir / "result.json").exists())
            self.assertTrue((s1.dir / "cloud.ply").exists())

    def test_serial_button_bad_port_fails_cleanly(self):
        import queue as q_mod
        with self.assertRaises(SystemExit):
            tm.start_serial_button_thread("__NONEXISTENT_SERIAL_PORT_999__", q_mod.Queue())

    def test_synthetic_shape_classification_accuracy(self):
        out = tm.bench(trials=4, max_grasps=6, upright=True, tilted=False, seed=5)
        # All 3 shapes x 4 trials = 12 runs; expect >= 10/12 right at grasp 6
        self.assertGreaterEqual(int(out["correct"][:, -1].sum()), 10)


class GloveTwinFixesTests(unittest.TestCase):
    def test_write_twin_resolves_mesh_path_relative_to_xml_dir(self):
        import xml.etree.ElementTree as ET

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scans_dir = root / "scans"
            scans_dir.mkdir()
            obj_file = scans_dir / "tet.obj"
            obj_file.write_text(
                "v 0 0 0\nv 0.05 0 0\nv 0 0.05 0\nv 0 0 0.05\n"
                "f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n",
                encoding="utf-8",
            )
            dens = estimate_density({"wood": 1.0})
            rel_from_cwd = obj_file.relative_to(root)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(root)
                props = build_properties(
                    "tet", {"wood": 1.0}, [], dens, classifier_source="test", mesh=str(rel_from_cwd)
                )
                paths = write_twin(props, Path("twins"))

                # Verify `<mesh file="..."/>` resolves directly from the XML directory (`twins/`)!
                tree = ET.parse(paths["glove"])
                mesh_elem = tree.find(".//mesh")
                self.assertIsNotNone(mesh_elem)
                assert mesh_elem is not None
                resolved_from_xml = (paths["glove"].parent / mesh_elem.attrib["file"]).resolve()
                self.assertEqual(resolved_from_xml, obj_file.resolve())

                try:
                    import mujoco  # type: ignore
                    model = mujoco.MjModel.from_xml_path(str(paths["glove"]))
                    self.assertGreaterEqual(model.nmesh, 1)
                except ImportError:
                    pass
            finally:
                os.chdir(old_cwd)

    def test_tap_parse_packet_rejects_malformed_and_nan_lines(self):
        self.assertIsNone(parse_packet("not json"))
        self.assertIsNone(parse_packet('{"fs": 0, "piezo": [1, 2], "fsr": [1, 2]}'))
        self.assertIsNone(parse_packet('{"fs": 16000, "piezo": [1, 2], "fsr": [1]}'))
        self.assertIsNone(parse_packet('{"fs": 16000, "piezo": ["bad"], "fsr": [1]}'))
        ok = parse_packet('{"fs": 16000, "pre": 2, "adc_bits": 12, "piezo": [2048, 2200, 2048], "fsr": [0, 500, 0]}')
        self.assertIsNotNone(ok)


if __name__ == "__main__":
    unittest.main()
