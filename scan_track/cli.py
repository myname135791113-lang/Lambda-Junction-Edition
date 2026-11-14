"""Command-line entry points for `scan_track`.

Examples:
  # 1. Instant zero-hardware live 3D demo in browser (synthetic hands + Convex Hull + G1 plan):
  python -m scan_track live --source synthetic

  # 2. Live Rokoko Smartgloves II right now (without Coil Pro — planted-wrist + left-hand gestures):
  python -m scan_track live --source rokoko --mode planted --port 14043

  # 3. Live Rokoko Smartgloves II + Coil Pro (full 6-DoF room-scale scan & relocation):
  python -m scan_track live --source rokoko --mode coil --port 14043 --calibration calibration.json

  # 4. Inspect raw Rokoko Studio / SDK UDP packets & test left-hand gestures in terminal:
  python -m scan_track dump --port 14043

  # 5. Headless end-to-end accuracy & relocation benchmark across all catalog objects:
  python -m scan_track bench
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from glove_twin.objects import OBJECTS

from .calibration import WorkspaceCalibration
from .gestures import GestureController
from .live_server import run_live_dashboard
from .pipeline import ScanTrackSession
from .rokoko_stream import RokokoUDPReceiver


def _parse_udp_target(spec: str | None) -> tuple[str, int] | None:
    if not spec:
        return None
    host, port_s = spec.rsplit(":", 1)
    return host, int(port_s)


def cmd_live(args: argparse.Namespace) -> int:
    session = ScanTrackSession(
        source=args.source,
        mode=args.mode,
        rokoko_port=args.port,
        synthetic_object=args.object,
        catalog_path=args.catalog,
        calibration_path=args.calibration,
        sessions_root=args.sessions_dir,
        g1_udp=_parse_udp_target(args.g1_udp),
        g1_arm=args.arm,
        dump_first_packet=args.dump_first,
    )
    run_live_dashboard(
        session,
        host=args.host,
        port=args.http_port,
        open_browser=not args.no_browser,
    )
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    print(f"Listening for Rokoko UDP packets on 0.0.0.0:{args.port} (press Ctrl+C to stop)...")
    rx = RokokoUDPReceiver(port=args.port, mode=args.mode, dump_first=True)
    gestures = GestureController()
    last_seq = 0
    try:
        while True:
            st = rx.status()
            if st["last_error"]:
                print(f"[UDP error] {st['last_error']}")
            frame = rx.latest()
            if frame is not None and frame.seq != last_seq:
                last_seq = frame.seq
                ev = gestures.update(frame.left, frame.timestamp)
                r_idx = (
                    "none"
                    if frame.right is None
                    else f"({frame.right.tips_pad['index'][0]*1000:6.1f}, "
                         f"{frame.right.tips_pad['index'][1]*1000:6.1f}, "
                         f"{frame.right.tips_pad['index'][2]*1000:6.1f}) mm"
                )
                print(
                    f"\rpkt #{frame.seq:5d} | Right:{'OK' if frame.right else '--'} idx_pad={r_idx} | "
                    f"Left:{'OK' if frame.left else '--'} gesture={ev.active_sign.upper():8s} "
                    f"(pinch={ev.pinch_dist_mm:4.1f} mm)",
                    end="",
                    flush=True,
                )
            time.sleep(0.04)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        rx.close()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    print(f"Connecting to Rokoko UDP on port {args.port} for Coil Pro <-> Unitree G1 calibration...")
    rx = RokokoUDPReceiver(port=args.port, mode="coil")
    try:
        n_pts = max(args.points, 3)
        coil_pts = []
        robot_pts = []
        print(f"\nTouch {n_pts} marked X spots on the flat workspace table with your gloved Right Index fingertip.")
        for i in range(n_pts):
            input(f"\n[{i+1}/{n_pts}] Hold Right Index fingertip on mark #{i+1} and press Enter...")
            samples = []
            t_end = time.time() + 0.35
            while time.time() < t_end:
                fr = rx.latest()
                if fr is not None and fr.right is not None:
                    samples.append(fr.right.tips_pad["index"])
                time.sleep(0.02)
            if not samples:
                print("ERROR: No right-hand Rokoko frames received! Check Rokoko Studio Custom Streaming.")
                return 1
            p_coil = np.mean(samples, axis=0)
            print(f"  Recorded Coil Pro point (m): {np.round(p_coil, 4).tolist()}")
            raw_r = input(
                f"  Enter Unitree G1 base coordinates `x y z` in metres for mark #{i+1} "
                "(or leave blank for table-only calibration): "
            ).strip()
            coil_pts.append(p_coil)
            if raw_r:
                vals = [float(tok) for tok in raw_r.replace(",", " ").split()[:3]]
                robot_pts.append(np.array(vals, dtype=float))

        coil_arr = np.vstack(coil_pts)
        if len(robot_pts) == len(coil_pts):
            cal = WorkspaceCalibration.from_paired_landmarks(
                coil_arr, np.vstack(robot_pts), fit_xy_warp=(len(coil_pts) >= 4 and args.warp)
            )
            print(f"\nCalibrated table + Unitree G1 alignment! Residual RMS = {cal.alignment_rms_mm:.2f} mm")
        else:
            cal = WorkspaceCalibration.from_table_points(coil_arr)
            print("\nCalibrated flat table plane (`z_table = 0.0 m`)!")

        out_p = Path(args.output)
        cal.save(out_p)
        print(f"Saved calibration to: {out_p.resolve()}")
        return 0
    finally:
        rx.close()


def cmd_bench(args: argparse.Namespace) -> int:
    print("Running headless end-to-end scan -> Convex Hull -> Catalog -> Relocation -> G1 benchmark...\n")
    all_ok = True
    for key in OBJECTS:
        sess = ScanTrackSession(source="synthetic", synthetic_object=key, sessions_root=args.sessions_dir)
        assert sess.synthetic is not None
        sess.synthetic.auto_cycle = False
        truth = sess.synthetic.ground_truth()

        # Simulate 14.0 seconds at 30 Hz across the full choreographed scan -> grasp -> carry -> place loop
        dt = 1.0 / 30.0
        snap = {}
        for _ in range(int(14.0 / dt)):
            snap = sess.tick(dt=dt)
        sess.close()

        matched = snap["catalog"]["best_key"]
        conf = snap["catalog"]["probabilities"].get(matched or "", 0.0) * 100.0
        placed = snap["tracked_object"]["pose"] if snap["tracked_object"] else None
        gt_xy = np.array([truth["placed_pose"]["x"], truth["placed_pose"]["y"]])
        est_xy = np.array([placed["x_m"], placed["y_m"]]) if placed else np.array([999.0, 999.0])
        xy_err_mm = float(np.linalg.norm(est_xy - gt_xy)) * 1000.0
        mj_out = snap["mujoco_result"]["outcome"] if snap["mujoco_result"] else "none"
        g1_safe = bool(snap["g1_plan"] and snap["g1_plan"]["safe_to_execute"])

        passed = (matched == key) and (xy_err_mm < 15.0) and (mj_out == "lifted") and g1_safe
        all_ok = all_ok and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {key:12s} | voxels={snap['cloud']['n_voxels']:3d} | "
            f"hull={snap['hull']['volume_cm3'] if snap['hull'] else 0:6.1f} cm^3 | "
            f"match={str(matched):12s} ({conf:5.1f}%) | "
            f"reloc_err={xy_err_mm:4.1f} mm | MuJoCo={mj_out:7s} | G1_safe={g1_safe}"
        )
    print("\nOverall benchmark:", "ALL PASSED" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scan_track",
        description="Rokoko Smartgloves II (+ Coil Pro) 3D Tactile Point Cloud, Catalog & Unitree G1 Tracker",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_live = sub.add_parser("live", help="Launch live 3D web dashboard & tactile tracking server")
    p_live.add_argument("--source", choices=("synthetic", "rokoko"), default="synthetic")
    p_live.add_argument("--mode", choices=("planted", "coil"), default="planted",
                        help="`planted` for Smartgloves-only right now; `coil` once Coil Pro is connected")
    p_live.add_argument("--port", type=int, default=14043, help="Rokoko Studio UDP streaming port")
    p_live.add_argument("--object", choices=tuple(OBJECTS.keys()), default="redbull",
                        help="Initial synthetic target object")
    p_live.add_argument("--catalog", default=None, help="Optional JSON file with extra catalog objects")
    p_live.add_argument("--calibration", default=None, help="Optional saved calibration.json path")
    p_live.add_argument("--sessions-dir", default="sessions", help="Directory for session outputs")
    p_live.add_argument("--g1-udp", default=None, help="Optional `host:port` to UDP-stream Unitree G1 pick JSON")
    p_live.add_argument("--arm", choices=("auto", "right", "left"), default="auto", help="Unitree G1 arm choice")
    p_live.add_argument("--host", default="127.0.0.1", help="Dashboard HTTP bind host")
    p_live.add_argument("--http-port", type=int, default=8765, help="Dashboard HTTP bind port")
    p_live.add_argument("--no-browser", action="store_true", help="Do not auto-open browser window")
    p_live.add_argument("--dump-first", action="store_true", help="Print joint names of first Rokoko packet")
    p_live.set_defaults(func=cmd_live)

    p_dump = sub.add_parser("dump", help="Inspect live Rokoko UDP stream & left-hand gestures in terminal")
    p_dump.add_argument("--port", type=int, default=14043)
    p_dump.add_argument("--mode", choices=("planted", "coil"), default="planted")
    p_dump.set_defaults(func=cmd_dump)

    p_cal = sub.add_parser("calibrate", help="Interactive table plane & Coil Pro <-> Unitree G1 calibration")
    p_cal.add_argument("--port", type=int, default=14043)
    p_cal.add_argument("--points", type=int, default=4)
    p_cal.add_argument("--warp", action="store_true", help="Fit 2D affine magnetic-warp correction across table")
    p_cal.add_argument("--output", default="calibration.json")
    p_cal.set_defaults(func=cmd_calibrate)

    p_bench = sub.add_parser("bench", help="Run headless end-to-end synthetic accuracy & relocation benchmark")
    p_bench.add_argument("--sessions-dir", default="sessions/bench")
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
