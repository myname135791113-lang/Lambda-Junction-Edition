"""Command line: calibrate on known materials, scan an object, build its twin.

  python -m glove_twin demo
  python -m glove_twin materials
  python -m glove_twin calibrate --material wood --taps 10 --port COM5
  python -m glove_twin scan --object mug --taps 5 --port COM5 --mass-g 350 --dims 0.09 0.09 0.10
  (replace --port COM5 with --simulate to run without hardware)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .classifier import TapClassifier, load_samples, save_samples
from .density import estimate_density
from .features import TapFeatures, extract_features
from .materials import MATERIALS
from .sources import SerialGlove, SimulatedGlove
from .twin import build_properties, write_twin

DEFAULT_CALIBRATION = Path("calibration.json")


def _describe(f: TapFeatures) -> str:
    ring = f"ring {f.ring_hz:6.0f} Hz, Q {f.q_factor:6.0f}" if f.has_ring else "no mic"
    return f"force {f.peak_force_n:4.1f} N, contact {f.contact_ms:4.2f} ms, {ring}"


def _source(args, sim_material: str | None):
    if args.port:
        return SerialGlove(args.port, args.baud)
    if sim_material is None:
        raise SystemExit("Give --port for the real glove, or --simulate MATERIAL")
    return SimulatedGlove(sim_material, seed=args.seed, with_mic=not args.no_mic, finger=args.finger)


def _classifier(args) -> TapClassifier:
    samples = load_samples(args.calibration)
    if samples and len({m for _, m in samples}) >= 2:
        clf = TapClassifier(samples, source=f"calibration ({len(samples)} taps, {args.calibration})")
        if clf.use_ring == args.no_mic:
            print("  note: calibration and this session differ in microphone use; using synthetic prior")
        else:
            return clf
    return TapClassifier.synthetic_prior(with_mic=not args.no_mic, finger=args.finger)


def cmd_materials(_args) -> None:
    print(f"{'material':<13}{'density kg/m3':>15}{'friction':>10}{'max grip N':>12}")
    for m in MATERIALS.values():
        print(f"{m.name:<13}{m.density:>15.0f}{m.friction:>10.2f}{m.max_grip_n:>12.0f}")


def cmd_calibrate(args) -> None:
    if args.material not in MATERIALS:
        raise SystemExit(f"Unknown material. Choose from: {', '.join(MATERIALS)}")
    samples = load_samples(args.calibration)
    glove = _source(args, args.material if args.simulate else None)
    print(f"Tap a {args.material} object lightly {args.taps} times (fingernail works best)...")
    try:
        for i, tap in enumerate(glove.taps(args.taps), 1):
            f = extract_features(tap)
            samples.append((f, args.material))
            print(f"  tap {i}/{args.taps}: {_describe(f)}")
    finally:
        glove.close()
    save_samples(args.calibration, samples)
    counts = {m: sum(1 for _, s in samples if s == m) for m in sorted({s for _, s in samples})}
    print(f"Saved to {args.calibration}: {counts}")


def cmd_scan(args) -> None:
    clf = _classifier(args)
    print(f"Classifier: {clf.source}")
    glove = _source(args, args.simulate)
    feats: list[TapFeatures] = []
    print(f"Tap '{args.object}' lightly {args.taps} times...")
    try:
        for i, tap in enumerate(glove.taps(args.taps), 1):
            f = extract_features(tap)
            feats.append(f)
            running = clf.predict_session(feats)
            best = next(iter(running))
            print(f"  tap {i}/{args.taps}: {_describe(f)}  -> {best} ({running[best]:.0%})")
    finally:
        glove.close()

    probs = clf.predict_session(feats)
    dims = tuple(args.dims) if args.dims else None
    volume = args.volume_cm3 or (float(np.prod(dims)) * 1e6 if dims else None)
    density = estimate_density(probs, args.mass_g, volume)
    props = build_properties(args.object, probs, feats, density, classifier_source=clf.source,
                             mass_g=args.mass_g, dims_m=dims, volume_cm3=volume, mesh=args.mesh)
    paths = write_twin(props, args.out)

    top = ", ".join(f"{m} {p:.0%}" for m, p in list(probs.items())[:3])
    print(f"\nMaterial:  {top}")
    print(f"Density:   {density.value:.0f} kg/m3 ({density.low:.0f}-{density.high:.0f}, {density.source})")
    if density.solidity is not None:
        print(f"Solidity:  {density.solidity:.2f} -> {density.note}")
    else:
        print(f"           {density.note}")
    hints = props["robot_hints"]
    print(f"Friction:  {props['friction_prior']:.2f}   Max safe grip: {hints['max_grip_force_n']:.0f} N"
          f"{'  (FRAGILE)' if hints['fragile'] else ''}")
    for kind, p in paths.items():
        print(f"  wrote {kind:<10} {p}")


def cmd_demo(args) -> None:
    """Simulated end-to-end run: classify every material, then build one twin."""
    clf = TapClassifier.synthetic_prior(with_mic=not args.no_mic, finger=args.finger, seed=0)
    rng_seed = 1
    print(f"Classifier: {clf.source}, microphone {'off' if args.no_mic else 'on'}, {args.finger} taps\n")
    print(f"{'true material':<14}{'predicted':<14}{'conf':>6}{'density est.':>14}{'true':>8}")
    correct = 0
    for name, m in MATERIALS.items():
        glove = SimulatedGlove(name, seed=rng_seed, with_mic=not args.no_mic, finger=args.finger)
        rng_seed += 1
        feats = [extract_features(t) for t in glove.taps(args.taps)]
        probs = clf.predict_session(feats)
        best = next(iter(probs))
        correct += best == name
        est = estimate_density(probs)
        print(f"{name:<14}{best:<14}{probs[best]:>6.0%}{est.value:>12.0f}  {m.density:>8.0f}"
              f"{'' if best == name else '   <- miss'}")
    print(f"\n{correct}/{len(MATERIALS)} materials identified from {args.taps} taps each.\n")

    print("Example twin: a ceramic mug, 350 g, 9 x 9 x 10 cm bounding box")
    scan = argparse.Namespace(**{**vars(args), "object": "demo_mug", "simulate": "ceramic", "port": None,
                                 "mass_g": 350.0, "dims": [0.09, 0.09, 0.10], "volume_cm3": None,
                                 "mesh": None, "calibration": Path("__no_calibration__.json")})
    cmd_scan(scan)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="glove_twin", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, simulate_takes_material: bool):
        src = sp.add_mutually_exclusive_group()
        src.add_argument("--port", help="serial port of the glove, e.g. COM5 or /dev/ttyACM0")
        if simulate_takes_material:
            src.add_argument("--simulate", metavar="MATERIAL", choices=list(MATERIALS),
                             help="use simulated taps on this material instead of hardware")
        else:
            src.add_argument("--simulate", action="store_true", help="use simulated taps")
        sp.add_argument("--baud", type=int, default=921_600)
        sp.add_argument("--taps", type=int, default=5)
        sp.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
        sp.add_argument("--no-mic", action="store_true", help="glove has no microphone channel")
        sp.add_argument("--finger", choices=["nail", "pad"], default="nail",
                        help="simulator only: tap with fingernail (sharper) or finger pad")
        sp.add_argument("--seed", type=int, default=None, help="simulator only: random seed")

    sub.add_parser("materials", help="list material priors").set_defaults(func=cmd_materials)

    sp = sub.add_parser("calibrate", help="record taps on an object of known material")
    sp.add_argument("--material", required=True)
    common(sp, simulate_takes_material=False)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("scan", help="tap an object and build its digital twin")
    sp.add_argument("--object", required=True, help="name for the twin files")
    sp.add_argument("--mass-g", type=float, help="measured mass (FSR lift or a kitchen scale)")
    sp.add_argument("--dims", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="bounding box in metres (from the phone scan)")
    sp.add_argument("--volume-cm3", type=float, help="true volume, e.g. from the scan mesh")
    sp.add_argument("--mesh", help="mesh file from the phone scan (.obj/.stl) for the MuJoCo geom")
    sp.add_argument("--out", type=Path, default=Path("twins"))
    common(sp, simulate_takes_material=True)
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("demo", help="simulated end-to-end run, no hardware needed")
    sp.add_argument("--out", type=Path, default=Path("twins"))
    common(sp, simulate_takes_material=False)
    sp.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)
