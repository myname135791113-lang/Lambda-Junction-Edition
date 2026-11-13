"""Visual end-to-end test report.

  Part A: touch mapping (box / cylinder / sphere) -> object card -> multi-finger robot grasp
  Part B: Rokoko-glove hand taps -> material -> digital twin -> parallel-gripper A/B

  python -m glove_twin.visual                 # both parts, write reports/visual/index.html, open it
  python -m glove_twin.visual --part mapping  # only part A (--trials N sets the accuracy trials)
  python -m glove_twin.visual --viewer        # also replay the gripper grasps in the MuJoCo viewer

Needs: mujoco, pillow (pip install -r requirements.txt).
"""
from __future__ import annotations

import argparse
import html
import time
import webbrowser
from pathlib import Path

import numpy as np

from .classifier import TapClassifier
from .density import estimate_density
from .features import extract_features
from .materials import DEFAULT_DENSITY, DEFAULT_FRICTION, MATERIALS
from .report import badge, bars, line_chart, page
from .robot_sim import (CALIBRATION_OBJECTS, SCENARIOS, TAP_FS, TAP_PRE, GraspResult, SceneObject,
                        grasp_object, grasp_scene_xml, plan_grip_force, tap_object)
from .twin import build_properties, write_twin

STATUS_RGB = {"lifted": (12, 163, 12), "dropped": (236, 131, 90), "crushed": (208, 59, 59)}


# ---------------------------------------------------------------- GIF helpers

def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _label(img, text: str, xy: tuple[int, int], size: int = 15, fill=(255, 255, 255), bg=(20, 20, 22)):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    font = _font(size)
    box = d.textbbox(xy, text, font=font)
    d.rounded_rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), radius=5, fill=bg)
    d.text(xy, text, font=font, fill=fill)


def save_gif(frames: list, path: Path, ms_per_frame: int) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms_per_frame, loop=0, optimize=True)


def tap_gif(res, obj: SceneObject, path: Path) -> None:
    from PIL import Image
    out = []
    for img, t, f in zip(res.frames, res.frame_t_ms, res.frame_force_n):
        im = Image.fromarray(img)
        _label(im, f"Rokoko-glove hand taps: {obj.label}", (10, 10))
        _label(im, f"t = {t:+5.0f} ms   slow motion 40x", (10, im.height - 30), size=13)
        if f > 0:
            _label(im, f"CONTACT  {f:4.1f} N", (im.width - 150, im.height - 30), size=13, bg=(235, 104, 52))
        out.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    save_gif(out, path, 40)


def grasp_gif(default: GraspResult, glove: GraspResult, obj: SceneObject, path: Path) -> None:
    from PIL import Image
    out = []
    n = min(len(default.frames), len(glove.frames))
    for i in range(n):
        step = i * 20
        panels = []
        for res, title in ((default, "A: default physics twin"), (glove, "B: glove-measured twin")):
            im = Image.fromarray(res.frames[i])
            _label(im, title, (10, 10))
            _label(im, f"grip {res.squeeze_n[min(step, len(res.squeeze_n) - 1)]:5.1f} N", (10, im.height - 30), size=13)
            if i >= n * 0.8:
                _label(im, res.outcome.upper(), (im.width - 110, im.height - 32), size=16, bg=STATUS_RGB[res.outcome])
            panels.append(im)
        canvas = Image.new("RGB", (panels[0].width * 2 + 6, panels[0].height), (250, 250, 248))
        canvas.paste(panels[0], (0, 0))
        canvas.paste(panels[1], (panels[0].width + 6, 0))
        out.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=160))
    save_gif(out, path, 40)


# ---------------------------------------------------------------- the run

def run(out: Path, taps: int, seed: int, render: bool = True, part: str = "all", trials: int = 5) -> dict:
    from .visual_mapping import mapping_body, run_mapping
    out.mkdir(parents=True, exist_ok=True)
    bodies, result = [], {}
    if part in ("all", "mapping"):
        mrun = run_mapping(out, seed, trials, render)
        bodies.append(mapping_body(mrun, trials))
        result["mapping"] = mrun
    if part in ("all", "taps"):
        tap = run_taps(out, taps, seed, render)
        bodies.append(tap["body"])
        result.update(tap)
    bodies.append("""<footer>Everything here is simulated: MuJoCo contact and grasp physics, a simplified acoustic
model of the ring, and the glove's published sensor limits. The hand follows the Rokoko Device SDK skeleton and
sensor layout, with size-M bone lengths. Accuracy on real objects has to be measured with the real glove.</footer>""")
    report = out / "index.html"
    report.write_text(page("Glove Test Report", "".join(bodies)), encoding="utf-8")
    result["report"] = report
    return result


def run_taps(out: Path, taps: int, seed: int, render: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()

    print("1/4  Calibrating: the Rokoko hand taps 8 reference blocks of known material ...")
    samples = []
    for obj in CALIBRATION_OBJECTS:
        for _ in range(6):
            samples.append((extract_features(tap_object(obj, rng).tap), obj.material))
    clf = TapClassifier(samples, source=f"calibration: {len(samples)} Rokoko-hand taps on 8 reference blocks")

    print("2/4  Scanning the test objects (taps -> material -> twin) ...")
    scans = []
    for obj in SCENARIOS:
        results = [tap_object(obj, rng, render=render and i == 0) for i in range(taps)]
        feats = [extract_features(r.tap) for r in results]
        probs = clf.predict_session(feats)
        mass_g = obj.mass_kg * 1000 * rng.normal(1.0, 0.03)       # human lift with FSRs, +-3%
        density = estimate_density(probs, mass_g, obj.volume_cm3)
        props = build_properties(obj.name, probs, feats, density, classifier_source=clf.source,
                                 mass_g=round(mass_g, 1), dims_m=obj.dims_m, volume_cm3=obj.volume_cm3)
        paths = write_twin(props, out / "twins")
        if render:
            tap_gif(results[0], obj, out / f"tap_{obj.name}.gif")
        best = next(iter(probs))
        print(f"     {obj.name:<14} -> {best:<12} {probs[best]:4.0%}   twin: {paths['glove'].name}")
        scans.append({"obj": obj, "taps": results, "feats": feats, "probs": probs, "props": props})

    print("3/4  Robot grasp A/B: default-physics twin vs glove twin ...")
    for s in scans:
        obj, props = s["obj"], s["props"]
        default_mass = DEFAULT_DENSITY * obj.volume_cm3 * 1e-6
        s["plan_default"] = plan_grip_force(default_mass, DEFAULT_FRICTION)
        s["plan_glove"] = plan_grip_force(props["mass_g"] / 1000, props["friction_prior"],
                                          props["robot_hints"]["max_grip_force_n"])
        s["default"] = grasp_object(obj, s["plan_default"], render=render)
        s["glove"] = grasp_object(obj, s["plan_glove"], render=render)
        if render:
            grasp_gif(s["default"], s["glove"], obj, out / f"grasp_{obj.name}.gif")
        print(f"     {obj.name:<14} A {s['plan_default']:5.1f} N -> {s['default'].outcome:<8} "
              f"B {s['plan_glove']:5.1f} N -> {s['glove'].outcome}")

    return {"body": build_report(scans, clf, taps, time.time() - t0, seed), "scans": scans}


# ---------------------------------------------------------------- report

def build_report(scans: list[dict], clf: TapClassifier, taps: int, seconds: float, seed: int) -> str:
    n = len(scans)
    glove_ok = sum(s["glove"].outcome == "lifted" for s in scans)
    default_ok = sum(s["default"].outcome == "lifted" for s in scans)
    mat_ok = sum(next(iter(s["probs"])) == s["obj"].material for s in scans)

    b = [f"""
<h2 class="part">Part B · Tap test: material from light taps, gripper A/B</h2>
<p class="lede">A simulated Rokoko Smartglove hand taps each object lightly with its index finger. The fingertip
FSR, piezo and wrist microphone identify the material, the twin gets density, friction and fragility, and
a robot gripper then grasps the real object using either default physics (as in RialTo) or the glove twin.</p>
<div class="tiles">
  <div class="tile"><div class="v">{glove_ok}/{n}</div><div class="k">safe lifts with the glove twin</div></div>
  <div class="tile"><div class="v">{default_ok}/{n}</div><div class="k">safe lifts with default physics</div></div>
  <div class="tile"><div class="v">{mat_ok}/{n}</div><div class="k">materials identified from {taps} taps each</div></div>
  <div class="tile"><div class="v">{len(clf.labels) + n * taps}</div><div class="k">simulated taps ({seconds:.0f} s run, seed {seed})</div></div>
</div>
<table><thead><tr><th>Object</th><th>Material (tap)</th><th class="num">Mass</th><th class="num">Density</th>
<th>Solidity</th><th class="num">A: default grip</th><th>A outcome</th><th class="num">B: glove grip</th><th>B outcome</th></tr></thead><tbody>"""]
    for s in scans:
        o, p, d = s["obj"], s["props"], s["props"]["density_kg_m3"]
        best = next(iter(s["probs"]))
        b.append(f"<tr><td>{html.escape(o.label)}</td><td>{best.replace('_', ' ')} ({s['probs'][best]:.0%})</td>"
                 f"<td class='num'>{p['mass_g']:.0f} g</td><td class='num'>{d['value']:.0f} kg/m&sup3;</td>"
                 f"<td>{html.escape(d['note'].split(';')[0].split(' (')[0])}</td>"
                 f"<td class='num'>{s['plan_default']:.1f} N</td><td>{badge(s['default'].outcome)}</td>"
                 f"<td class='num'>{s['plan_glove']:.1f} N</td><td>{badge(s['glove'].outcome)}</td></tr>")
    b.append("</tbody></table>")

    b.append("""<h3 style="margin-top:28px">B1 · The Rokoko hand taps</h3>
<p class="lede">The Rokoko glove streams finger pose at 83.33 Hz, one frame every 12 ms. A tap lasts under a
millisecond, so the glove's own frames miss it entirely. The fingertip FSR + piezo (16 kHz) catch the impact, and
the wrist microphone hears the object ring after the finger has left.</p>""")
    for s in scans:
        o, r, f = s["obj"], s["taps"][0], s["feats"][0]
        tap = r.tap
        t_ms = (np.arange(len(tap.force_n)) - TAP_PRE) / TAP_FS * 1000
        contact_end = float(np.count_nonzero(r.force_true > 0) / TAP_FS * 1000)
        charts = [
            line_chart("Fingertip force: true contact vs FSR reading", t_ms,
                       [{"name": "MuJoCo contact", "y": r.force_true, "slot": 1},
                        {"name": "FSR reading", "y": tap.force_n, "slot": 2, "dash": True}],
                       x_label="ms after contact", y_unit="N", x_range=(-1, 6),
                       note="The FSR lags and under-reads short taps; calibration absorbs this."),
            line_chart("Microphone: the object rings", t_ms, [{"name": "mic", "y": tap.mic, "slot": 1}],
                       x_label="ms after contact", y_unit="level", x_range=(-2, 60),
                       note=f"Ring {f.ring_hz:.0f} Hz, Q {f.q_factor:.0f}. Damping (Q) is the main material clue."),
            line_chart("What the Rokoko glove sees: index flexion", r.flex_t_ms,
                       [{"name": "true finger motion", "y": r.flex_deg, "slot": 1, "dash": True}],
                       dots={"name": "Rokoko frame (83 Hz)", "x": r.rokoko_t_ms, "y": r.rokoko_flex_deg, "slot": 2},
                       bands=[{"x0": 0, "x1": max(contact_end, 0.6), "label": "contact"}],
                       x_label="ms after contact", y_unit="degrees", x_range=(-45, 60),
                       y_range=(float(r.flex_deg.min()) - 2, float(r.flex_deg.max()) + 2),
                       note="Dots are the glove's own frames, 12 ms apart. None lands on the sub-millisecond contact."),
        ]
        feat_rows = "".join(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in [
            ("contact time", f"{f.contact_ms:.2f} ms"), ("peak force (FSR)", f"{f.peak_force_n:.1f} N"),
            ("ring frequency", f"{f.ring_hz:.0f} Hz"), ("ring Q", f"{f.q_factor:.0f}"),
            ("tip speed", f"{r.tip_speed_ms:.2f} m/s")])
        b.append(f"""<section class="card"><div class="card-head"><h3>{html.escape(o.label)}</h3>
<span class="sub">true material: {o.material.replace('_', ' ')} · {len(s['taps'])} taps</span></div>
<div class="grid2"><div><img class="anim" src="tap_{o.name}.gif" alt="Rokoko hand tapping {html.escape(o.label)}">
<h3 style="margin-top:14px">Material from {len(s['taps'])} taps</h3>{bars(s['probs'])}
<table><tbody>{feat_rows}</tbody></table></div>
<div>{''.join(charts)}</div></div></section>""")

    b.append("""<h3 style="margin-top:28px">B2 · Parallel-gripper grasp A/B</h3>
<p class="lede">Same gripper, same real object. The grip force is planned from the twin:
safety factor 1.5 × weight ÷ (2 × friction), capped at 80% of the safe force when the twin knows the
object is fragile. A: default physics (density 1000, friction 1, no fragility). B: the glove twin.</p>""")
    for s in scans:
        o = s["obj"]
        A, B = s["default"], s["glove"]
        chart = line_chart("Grip force during the grasp", A.t,
                           [{"name": "A: default twin", "y": A.squeeze_n, "slot": 2},
                            {"name": "B: glove twin", "y": np.interp(A.t, B.t, B.squeeze_n), "slot": 1}],
                           x_label="seconds", y_unit="N",
                           refs=[{"y": o.max_grip_n, "label": f"breaks above {o.max_grip_n:.0f} N"},
                                 {"y": A.needed_force_n, "label": f"slips below {A.needed_force_n:.1f} N"}]
                           if o.max_grip_n < 60 else
                           [{"y": A.needed_force_n, "label": f"slips below {A.needed_force_n:.1f} N"}])
        height = line_chart("Object height", A.t,
                            [{"name": "A: default twin", "y": A.height_m * 100, "slot": 2},
                             {"name": "B: glove twin", "y": np.interp(A.t, B.t, B.height_m) * 100, "slot": 1}],
                            x_label="seconds", y_unit="cm")
        p = s["props"]
        b.append(f"""<section class="card"><div class="card-head"><h3>{html.escape(o.label)}</h3>
<span>{badge(A.outcome)} <span class="sub">A</span> &nbsp; {badge(B.outcome)} <span class="sub">B</span></span></div>
<img class="anim" src="grasp_{o.name}.gif" alt="Grasp comparison for {html.escape(o.label)}">
<div class="ab"><div><b>A</b> assumes {DEFAULT_DENSITY * o.volume_cm3 * 1e-3:.0f} g, friction 1.0 → plans {s['plan_default']:.1f} N</div>
<div><b>B</b> knows {p['mass_g']:.0f} g, friction {p['friction_prior']:.2f}, max {p['robot_hints']['max_grip_force_n']:.0f} N → plans {s['plan_glove']:.1f} N</div></div>
<div class="grid2"><div>{chart}</div><div>{height}</div></div>
<details><summary>Twin files</summary><span class="pill">twins/{o.name}.properties.json</span>
<span class="pill">twins/{o.name}.glove.xml</span><span class="pill">twins/{o.name}.default.xml</span></details></section>""")

    return "".join(b)


def replay_in_viewer(scans: list[dict]) -> None:
    import mujoco
    import mujoco.viewer
    for s in scans:
        for which in ("default", "glove"):
            obj, force = s["obj"], s[f"plan_{which}"]
            model = mujoco.MjModel.from_xml_string(grasp_scene_xml(obj))
            data = mujoco.MjData(model)
            print(f"Viewer: {obj.name} with the {which} twin ({force:.1f} N), plays for 3.5 s ...")
            with mujoco.viewer.launch_passive(model, data) as v:
                t_wall = time.time()
                while v.is_running() and data.time < 3.5:
                    t = data.time
                    data.ctrl[1] = data.ctrl[2] = force * min(1.0, t / 0.3)
                    data.ctrl[0] = 0.0 if t < 0.7 else 0.12 * min(1.0, (t - 0.7) / 1.2)
                    mujoco.mj_step(model, data)
                    v.sync()
                    time.sleep(max(0.0, data.time - (time.time() - t_wall)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="glove_twin.visual", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("reports/visual"))
    ap.add_argument("--taps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-open", action="store_true", help="don't open the report in the browser")
    ap.add_argument("--viewer", action="store_true", help="replay the gripper grasps in the MuJoCo viewer")
    ap.add_argument("--part", choices=["all", "mapping", "taps"], default="all")
    ap.add_argument("--trials", type=int, default=5, help="extra mapping sessions per object for the accuracy table")
    args = ap.parse_args(argv)
    result = run(args.out, args.taps, args.seed, part=args.part, trials=args.trials)
    print(f"\nReport: {result['report'].resolve()}")
    if not args.no_open:
        webbrowser.open(result["report"].resolve().as_uri())
    if args.viewer and "scans" in result:
        replay_in_viewer(result["scans"])


if __name__ == "__main__":
    main()
