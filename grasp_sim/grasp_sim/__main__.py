"""Command line:  python -m grasp_sim {run,suite,gui} ...   (see README.md)"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .run import RUNS, Spec, execute


def _spec_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("object (pick one source)")
    g.add_argument("--shape", choices=["box", "cylinder", "sphere"], help="built-in primitive (default cylinder)")
    g.add_argument("--json", dest="json_path", help="glove-made shape: touchmap result.json (+cloud.ply), "
                   "{vertices,faces}, {points} or {shape,params}")
    g.add_argument("--use-fit", action="store_true", help="touchmap json: use the fitted primitive instead of the point-cloud hull")
    g.add_argument("--card", dest="card_path", help="full object card .json (mesh, pose, mass_kg, mu_pad, max_force_N, ...)")
    d = p.add_argument_group("primitive dimensions in mm (defaults: box 60x60x100, cylinder r33 h115, sphere r20)")
    d.add_argument("--size", type=float, nargs=3, metavar=("X", "Y", "Z"), help="box edge lengths")
    d.add_argument("--radius", type=float, help="cylinder / sphere radius")
    d.add_argument("--height", type=float, help="cylinder height")
    o = p.add_argument_group("physical properties")
    o.add_argument("--mass", type=float, default=0.35, help="kg (default 0.35)")
    o.add_argument("--material", default="aluminium", help="object material, used for pad friction if --mu is not given")
    o.add_argument("--mu", type=float, default=None, help="object vs robot finger pad friction (default 0.5)")
    o.add_argument("--max-force", type=float, default=25.0, help="N; object breaks above this contact force")
    o.add_argument("--hollow", action="store_true", help="hollow: mass in the shell, not uniform")
    o.add_argument("--x", type=float, help="object x in robot frame, m (default 0.35, or where the glove saw it)")
    o.add_argument("--y", type=float, help="object y in robot frame, m (default -0.20)")
    o.add_argument("--yaw", type=float, default=0.0, help="object yaw, degrees")
    p.add_argument("--name")
    p.add_argument("--out", help="output folder (default runs/<time>_<name>)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--check-only", action="store_true", help="only the instant feasibility check (no physics)")
    p.add_argument("--attempts", type=int, default=3, help="different grasps to try before giving up (1 = single try)")


def _to_spec(a) -> Spec:
    dims = {}
    if a.shape == "box" and a.size:
        dims = {"x": a.size[0] / 1000, "y": a.size[1] / 1000, "z": a.size[2] / 1000}
    elif a.shape == "cylinder":
        if a.radius:
            dims["radius"] = a.radius / 1000
        if a.height:
            dims["height"] = a.height / 1000
    elif a.shape == "sphere" and a.radius:
        dims["radius"] = a.radius / 1000
    pos = (a.x if a.x is not None else 0.35, a.y if a.y is not None else -0.20) if (a.x is not None or a.y is not None) else None
    return Spec(shape=a.shape or "cylinder", dims=dims, json_path=a.json_path, use_fit=a.use_fit, card_path=a.card_path,
                pos=pos, yaw_deg=a.yaw, mass_kg=a.mass, material=a.material, mu_pad=a.mu if a.mu is not None else 0.5,
                max_force_N=a.max_force, hollow=a.hollow, name=a.name)


def cmd_run(a):
    r = execute(_to_spec(a), a.out, video=not a.no_video, check_only=a.check_only, attempts=a.attempts)
    print("output:", r["output_dir"])


def cmd_suite(a):
    from .suite import test_set
    from .card import load_card_mesh
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = Path(a.out) if a.out else RUNS / f"{stamp}_suite"
    rows = []
    for name, (card, shape) in test_set().items():
        if a.only and name not in a.only:
            continue
        spec = Spec(card_path=None)
        # reuse the already-built card (mesh in memory)
        from .pipeline import check, public, run_lift
        from .run import annotate, montage
        import imageio
        import numpy as np
        from PIL import Image
        out = base / name
        out.mkdir(parents=True, exist_ok=True)
        v = check(card)
        row = {"object": name, "verdict": v["verdict"], "detail": v["detail"]}
        res = None
        if v["verdict"] == "CAN_PICK":
            from .pipeline import run_best
            res, tried = run_best(card, v, attempts=a.attempts, render=not a.no_video, log=lambda s: print(s, flush=True))
            row.update(res.to_json())
            row["grasp_used"] = res.grasp
            row["attempts"] = [{"grasp": g, "outcome": o} for g, o in tried]
            if res.frames:
                frames = annotate(res.frames, res.log, name, res)
                imageio.mimwrite(out / "video.mp4", frames, fps=30, quality=8)
                Image.fromarray(montage(frames)).save(out / "preview.png")
        (out / "result.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        extra = f"  {res.outcome}  peak {res.peak_force_N:.1f}/{res.max_force_N:g} N  slip {res.slip_mm:.1f} mm" if res else f"  ({v['detail']})"
        print(f"{name:18s} {v['verdict']:14s}{extra}", flush=True)
    (base / "summary.json").write_text(json.dumps(rows, indent=2))
    print("output:", base)


def cmd_showcase(a):
    """Test set + the glove sessions in samples/, into one browsable folder."""
    import glob
    from .gallery import build
    base = Path(a.out) if a.out else RUNS / "showcase"
    a.out = str(base)
    a.only = None
    cmd_suite(a)
    root = Path(__file__).resolve().parent.parent / "samples" / "touchmap_ref"
    for kind in ("cylinder", "box", "sphere"):
        hits = glob.glob(str(root / f"sessions_{kind}" / "*" / "result.json"))
        if not hits:
            continue
        for fit in (False, True):
            spec = Spec(json_path=hits[0], use_fit=fit, mass_kg=0.3, material="aluminium", mu_pad=0.5,
                        max_force_N=25, hollow=True, name=f"glove_{kind}_{'fit' if fit else 'hull'}")
            execute(spec, base / spec.name, video=not a.no_video, attempts=a.attempts)
    print("gallery:", build(base, "Robot hand lift test", "Unitree G1 right arm + Dex3-style hand, MuJoCo. Glove_* rows are real touchmap sessions."))


def cmd_gui(a):
    from .gui import main
    main()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="grasp_sim", description="Can the robot hand pick up this object?")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="check one object and simulate the lift")
    _spec_args(pr)
    pr.set_defaults(fn=cmd_run)
    ps = sub.add_parser("suite", help="run the spec's test set")
    ps.add_argument("--out")
    ps.add_argument("--only", nargs="*", help="names, e.g. full_can tennis_ball")
    ps.add_argument("--no-video", action="store_true")
    ps.add_argument("--attempts", type=int, default=3, help="different grasps to try per object")
    ps.set_defaults(fn=cmd_suite)
    pw = sub.add_parser("showcase", help="test set + glove sessions, with videos and an index.html")
    pw.add_argument("--out")
    pw.add_argument("--no-video", action="store_true")
    pw.add_argument("--attempts", type=int, default=3)
    pw.set_defaults(fn=cmd_showcase)
    pg = sub.add_parser("gui", help="small window to set the shape and parameters, then run")
    pg.set_defaults(fn=cmd_gui)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
