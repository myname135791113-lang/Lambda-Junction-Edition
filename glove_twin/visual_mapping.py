"""Part A of the visual report: touch mapping -> object card -> multi-finger robot grasp."""
from __future__ import annotations

import html
import time
from pathlib import Path

import numpy as np

from .mapping import calibrate, make_card, map_object
from .object_card import validate_card, write_card
from .objects import OBJECTS, SOURCES
from .report import badge, bars, line_chart, ortho_views
from .robot_hand import robot_grasp

KIND_LABEL = {"enclosure": "enclosure grasp", "top": "top grasp (palm on top)", "caliper": "caliper pinch"}


def exploration_gif(images: list, label: str, path: Path) -> None:
    from PIL import Image
    from .visual import _label, save_gif
    frames = []
    n = len(images)
    for i, (kind, on, img) in enumerate(images):
        im = Image.fromarray(img)
        _label(im, f"Touch mapping: {label}", (10, 10))
        _label(im, f"grasp {i + 1}/{n}: {KIND_LABEL.get(kind, kind)}, {len(on)} pads pressed",
               (10, im.height - 30), size=13)
        frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    save_gif(frames, path, 1300)


def robot_gif(res, label: str, grasp_type: str, path: Path) -> None:
    from PIL import Image
    from .visual import STATUS_RGB, _label, save_gif
    frames = []
    n = len(res.frames)
    for i, img in enumerate(res.frames):
        im = Image.fromarray(img)
        _label(im, f"Robot hand: {grasp_type.replace('_', ' ')} from the object card", (10, 10))
        k = min(int(i / max(n - 1, 1) * (len(res.squeeze_n) - 1)), len(res.squeeze_n) - 1)
        _label(im, f"squeeze {res.squeeze_n[k]:4.1f} N", (10, im.height - 30), size=13)
        if i >= n * 0.82:
            _label(im, res.outcome.upper(), (im.width - 110, im.height - 32), size=16, bg=STATUS_RGB[res.outcome])
        frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    save_gif(frames, path, 40)


def accuracy_trials(off: dict, trials: int, seed0: int = 100) -> dict:
    out = {}
    for key, o in OBJECTS.items():
        truth = o.dims_vector()
        rows = []
        for k in range(trials):
            m = map_object(o, np.random.default_rng(seed0 + k), pad_offset=off)
            res = m.shape
            err = np.array(list(res.dims.values())) - truth if res.best == o.shape else None
            rows.append({"best": res.best, "ok": res.best == o.shape, "flagged": res.ambiguous,
                         "follow_ups": m.follow_ups, "err": err})
        out[key] = rows
    return out


def run_mapping(out: Path, seed: int, trials: int, render: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("A1  Calibrating pad offsets on the Red Bull can (one-time) ...")
    off = calibrate(np.random.default_rng(1000))
    results = []
    print("A2  Touch-mapping each object, building its card, robot grasp ...")
    for key, o in OBJECTS.items():
        rng = np.random.default_rng(seed)
        m = map_object(o, rng, pad_offset=off, render=render)
        card = make_card(o, m, rng)
        assert not validate_card(card), validate_card(card)
        card_path = write_card(card, out / "cards")
        r = robot_grasp(o, card["grasp"], card["grasp"]["grip_force_n"], render=render)
        if render:
            exploration_gif(m.images, o.label, out / f"map_{key}.gif")
            robot_gif(r, o.label, card["grasp"]["type"], out / f"robot_{key}.gif")
        d = {k: round(v * 1000, 1) for k, v in m.shape.dims.items()}
        print(f"     {key:<12} -> {m.shape.best:<8} {d} follow-ups {m.follow_ups}  robot: {r.outcome}")
        results.append({"obj": o, "map": m, "card": card, "card_path": card_path, "robot": r})
    acc = {}
    if trials:
        print(f"A3  Accuracy trials: {trials} more sessions per object ...")
        acc = accuracy_trials(off, trials)
    return {"results": results, "offset": off, "accuracy": acc, "seconds": time.time() - t0}


def _dims_table(o, shape) -> str:
    truth = o.dims_vector()
    names = list(shape.dims)
    rows = []
    for i, k in enumerate(names):
        est, sig = shape.dims[k], shape.sigma[k]
        t = truth[i] if shape.best == o.shape else None
        err = f"{(est - t) * 1000:+.1f}" if t is not None else "n/a"
        rows.append(f"<tr><td>{k}</td><td class='num'>{t * 1000:.1f}</td>" if t is not None else f"<tr><td>{k}</td><td class='num'>-</td>")
        shown = f"{est * 1000:.1f} &plusmn; {sig * 1000:.1f}" if np.isfinite(sig) else f"&ge; {est * 1000:.1f} (not measured)"
        rows[-1] += f"<td class='num'>{shown}</td><td class='num'>{err if np.isfinite(sig) else 'n/a'}</td></tr>"
    return ("<table><thead><tr><th>dimension</th><th class='num'>true (mm)</th><th class='num'>from touch (mm)</th>"
            "<th class='num'>error</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def mapping_body(run: dict, trials: int) -> str:
    res, acc = run["results"], run["accuracy"]
    n = len(res)
    shape_ok = sum(r["map"].shape.best == r["obj"].shape for r in res)
    lifted = sum(r["robot"].outcome == "lifted" for r in res)
    all_rows = [row for rows in acc.values() for row in rows]
    trial_ok = sum(r["ok"] for r in all_rows)
    errs = [np.abs(r["err"]).max() * 1000 for r in all_rows if r["err"] is not None and not r["flagged"]]
    errs += [np.abs(np.array(list(r["map"].shape.dims.values())) - r["obj"].dims_vector()).max() * 1000
             for r in res if r["map"].shape.best == r["obj"].shape and not r["map"].shape.ambiguous]
    off = run["offset"]
    b = [f"""
<h1>Lambda: Junction Edition, <span>glove test report</span></h1>
<p class="lede">With the Rokoko glove on, a person maps an object <b>by touch alone</b>: its shape (box, cylinder
or sphere), its size and its physical properties. The result goes to a robot hand as an <b>object card</b>, so it
knows how to grab that kind of object. Everything below is simulated, using the published specs of the
three model objects and of the glove.</p>
<h2 class="part">Part A · Touch mapping to robot grasp</h2>
<div class="tiles">
  <div class="tile"><div class="v">{shape_ok}/{n}</div><div class="k">shapes recognised (this run)</div></div>
  <div class="tile"><div class="v">{lifted}/{n}</div><div class="k">robot lifts with the card's grip force</div></div>
  <div class="tile"><div class="v">{trial_ok}/{len(all_rows) or '-'}</div><div class="k">shapes recognised over {trials} extra sessions per object</div></div>
  <div class="tile"><div class="v">{(max(errs) if errs else float('nan')):.1f} mm</div><div class="k">worst dimension error (unflagged sessions)</div></div>
</div>
<div class="note-box"><b>Gloves only, no Coil Pro.</b> The glove knows finger positions relative to the back of the
hand (Rokoko datasheet: 1.60 mm typical) but not where the hand is in the room. Because an object's size doesn't
depend on where the hand is, one joint fit estimates the shape plus an unknown hand pose for every grasp.
Thumb-to-index pinches measure widths and heights directly. Pad offsets were calibrated once on the can:
enclosure {off['enclosure'] * 1000:.1f} mm, pinch {off['caliper'] * 1000:.1f} mm.</div>
<p class="lede"><b>Routine per object (30 to 60 s, object held down):</b> two side enclosures, a top grasp, and
thumb-to-index pinches across width, depth and height. If the result is ambiguous, the person is asked for more
touches (up to two extra rounds).</p>"""]
    for r in res:
        o, m, card, rob = r["obj"], r["map"], r["card"], r["robot"]
        sh = m.shape
        fit = sh.fits[sh.best]
        kinds = [m.grasps[i].kind for i in fit.grasp_index]
        ok = sh.best == o.shape
        verdict = (badge("lifted").replace("Lifted safely", f"{sh.best} (correct)") if ok
                   else badge("crushed").replace("Crushed", f"{sh.best} (wrong)"))
        reasons = ("<div class='note-box'>Flagged: " + html.escape("; ".join(sh.reasons)) + "</div>") if sh.reasons else ""
        g = card["grasp"]
        card_rows = "".join(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in [
            ("mass", f"{card['mass_kg'] * 1000:.0f} g ({card['mass_source'].split(',')[0]})"),
            ("material (taps)", f"{card['material']['best'].replace('_', ' ')} ({card['material']['probabilities'][card['material']['best']]:.0%})"),
            ("bulk density", f"{card['bulk_density_kg_m3']:.0f} kg/m&sup3;"),
            ("friction (slide test)", f"{card['friction']:.2f}"),
            ("max safe squeeze", f"{card['max_safe_grip_n']:.0f} N" + (" (fragile)" if card["fragile"] else "")),
            ("grasp", f"{g['type'].replace('_', ' ')}, {g['grip_width_m'] * 1000:.0f} mm wide"),
            ("grip force", f"{g['grip_force_n']:.1f} N")])
        chart = line_chart("Robot squeeze during the grasp", rob.t, [{"name": "squeeze", "y": rob.squeeze_n, "slot": 1}],
                           x_label="seconds", y_unit="N",
                           refs=[{"y": g["grip_force_n"], "label": f"card target {g['grip_force_n']:.1f} N"}])
        height = line_chart("Object height", rob.t, [{"name": "height", "y": rob.height_m * 100, "slot": 1}],
                            x_label="seconds", y_unit="cm")
        src = " · ".join(f'<a href="{SOURCES[s]}">{s.replace("_", " ")}</a>' for s in o.sources)
        b.append(f"""<section class="card"><div class="card-head"><h3>{html.escape(o.label)}</h3>
<span>{verdict} &nbsp; {badge(rob.outcome)} <span class="sub">{len(m.grasps)} grasps, {sum(len(x.points) for x in m.grasps)} contacts, {m.follow_ups} follow-up round(s)</span></span></div>
<div class="grid2"><div><img class="anim" src="map_{o.key}.gif" alt="Touch mapping of {html.escape(o.label)}">
<h3 style="margin-top:14px">Which shape? (BIC weights)</h3>{bars(sh.weights)}{reasons}
<p class="sub">Next touch if needed: <b>{html.escape(sh.touch_here)}</b>. True dimensions from: {src}.</p></div>
<div><h3>Contacts in the fitted object frame</h3>{ortho_views(fit.points_obj, sh.best, fit.theta, kinds)}
{_dims_table(o, sh)}</div></div>
<div class="grid2" style="margin-top:14px"><div><img class="anim" src="robot_{o.key}.gif" alt="Robot grasp of {html.escape(o.label)}">
<table style="margin-top:8px"><tbody>{card_rows}</tbody></table>
<p class="sub">Card: <code>cards/{o.key}.card.json</code></p></div>
<div>{chart}{height}</div></div></section>""")
    if acc:
        rows = []
        for key, trows in acc.items():
            o = OBJECTS[key]
            ok = sum(t["ok"] for t in trows)
            fl = sum(t["flagged"] for t in trows)
            fu = sum(t["follow_ups"] for t in trows)
            e = np.array([t["err"] for t in trows if t["err"] is not None]) * 1000
            names = {"sphere": ["radius"], "cylinder": ["radius", "height"], "box": ["width", "depth", "height"]}[o.shape]
            errs_txt = ", ".join(f"{nm} {np.abs(e[:, i]).mean():.1f} / {np.abs(e[:, i]).max():.1f}" for i, nm in enumerate(names)) if len(e) else "-"
            rows.append(f"<tr><td>{html.escape(o.label)}</td><td class='num'>{ok}/{len(trows)}</td><td class='num'>{fl}</td>"
                        f"<td class='num'>{fu}</td><td>{errs_txt}</td></tr>")
        b.append(f"""<h3 style="margin-top:22px">Accuracy over {trials} more sessions per object (new hand size and noise each time)</h3>
<table><thead><tr><th>Object</th><th class='num'>shape correct</th><th class='num'>still flagged</th><th class='num'>follow-up rounds</th>
<th>dimension error, mean / max (mm)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="sub">A flagged session tells the person what is missing (for example "pinch across the height") instead of
returning a confident wrong answer.</p>""")
    return "".join(b)
