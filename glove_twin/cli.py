"""Command line: calibrate on known materials, scan an object, build its twin.

  python -m glove_twin demo
  python -m glove_twin materials
  python -m glove_twin calibrate --material wood --taps 10 --port COM5
  python -m glove_twin scan --object mug --taps 5 --port COM5 --mass-g 350 --dims 0.09 0.09 0.10
  (replace --port COM5 with --simulate to run without hardware)
  python -m glove_twin map --object redbull --simulate --robot
  python -m glove_twin map --frames session.json --mass-g 270 --material filled_can
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


def cmd_map(args) -> None:
    """Touch mapping: shape + size from touch, then the object card (and optionally a robot grasp)."""
    from .mapping import calibrate, make_card, map_object
    from .object_card import build_card, validate_card, write_card
    from .objects import OBJECTS
    from .shape_fit import fit_shape
    from .touch import load_frames, reconstruct

    offset = {"enclosure": args.offset_enclosure_mm / 1000, "caliper": args.offset_caliper_mm / 1000}
    if args.frames:                                       # recorded glove data
        if args.mass_g is None or args.material is None:
            raise SystemExit("With --frames, give --mass-g (scale or lift test) and --material")
        grasps = reconstruct(load_frames(args.frames), pad_offset=offset)
        shape = fit_shape(grasps)
        card = build_card(Path(args.frames).stem, Path(args.frames).stem, shape, mass_kg=args.mass_g / 1000,
                          mass_source="entered by user", material_probs={args.material: 1.0},
                          n_contacts=sum(len(g.points) for g in grasps), n_grasps=len(grasps))
        truth, obj = None, None
    else:
        if not args.simulate or not args.object:
            raise SystemExit("Give --object and --simulate, or --frames FILE")
        obj = OBJECTS[args.object]
        if args.offset_enclosure_mm == 0 and args.offset_caliper_mm == 0:
            print("Calibrating pad offsets on the Red Bull can (simulated) ...")
            offset = calibrate(np.random.default_rng(1000))
        rng = np.random.default_rng(args.seed)
        m = map_object(obj, rng, pad_offset=offset, max_follow_ups=args.follow_ups)
        shape, truth = m.shape, obj.dims_vector()
        card = make_card(obj, m, rng)
        print(f"{len(m.grasps)} grasps, {sum(len(g.points) for g in m.grasps)} contacts, "
              f"{m.follow_ups} follow-up round(s)")
    problems = validate_card(card)
    if problems:
        raise SystemExit(f"Card invalid: {problems}")
    print(f"\nShape:  {shape.best}   (" + ", ".join(f"{k} {v:.0%}" for k, v in shape.weights.items()) + ")")
    for i, (k, v) in enumerate(shape.dims.items()):
        sig = shape.sigma[k]
        line = f"  {k:<7} {v * 1000:6.1f} mm " + (f"+- {sig * 1000:.1f}" if np.isfinite(sig) else "(not measured)")
        if truth is not None and shape.best == obj.shape:
            line += f"   true {truth[i] * 1000:6.1f}   error {(v - truth[i]) * 1000:+5.1f}"
        print(line)
    for r in shape.reasons:
        print(f"  ! {r}")
    g = card["grasp"]
    print(f"Card:   {card['mass_kg'] * 1000:.0f} g, {card['material']['best']}, friction {card['friction']:.2f}, "
          f"max squeeze {card['max_safe_grip_n']:.0f} N")
    print(f"Grasp:  {g['type']} at {g['grip_force_n']:.1f} N  ->  {write_card(card, args.out)}")
    if args.robot and obj is not None:
        from .robot_hand import robot_grasp
        r = robot_grasp(obj, g, g["grip_force_n"])
        print(f"Robot:  {r.outcome} (lift {r.lift_m * 100:.1f} cm, peak squeeze {r.peak_squeeze_n:.1f} N)")


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

    from .objects import OBJECTS as _OBJECTS
    sp = sub.add_parser("map", help="map an object's shape by touch and write its object card")
    sp.add_argument("--object", choices=list(_OBJECTS), help="model object (simulation)")
    sp.add_argument("--simulate", action="store_true", help="simulate the glove session")
    sp.add_argument("--frames", help="recorded glove frames (JSON, see touch.save_frames)")
    sp.add_argument("--mass-g", type=float, help="mass from a scale or the lift test (with --frames)")
    sp.add_argument("--material", choices=list(MATERIALS), help="material from the tap test (with --frames)")
    sp.add_argument("--offset-enclosure-mm", type=float, default=0.0, help="pad offset from calibration")
    sp.add_argument("--offset-caliper-mm", type=float, default=0.0, help="pinch offset from calibration")
    sp.add_argument("--follow-ups", type=int, default=2, help="extra touch rounds while ambiguous")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--robot", action="store_true", help="also run the robot-hand grasp (simulation)")
    sp.add_argument("--out", type=Path, default=Path("cards"))
    sp.set_defaults(func=cmd_map)

    sp = sub.add_parser("demo", help="simulated end-to-end run, no hardware needed")
    sp.add_argument("--out", type=Path, default=Path("twins"))
    common(sp, simulate_takes_material=False)
    sp.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)
