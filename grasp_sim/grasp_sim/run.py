"""Build an object (primitive, glove JSON, or card), check it, simulate the lift, write the outputs."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .card import Card, card_from_dict, load_card, load_card_mesh
from .pipeline import check, public, run_best, run_lift
from .shapes import Shape, load_shape_json, make_shape, save_obj

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _font(size):
    for f in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class Spec:
    """Everything needed to define one object. Dimensions in metres."""
    shape: str | None = "cylinder"          # box | cylinder | sphere (ignored when json/card is given)
    dims: dict | None = None                # e.g. {"x":.05,"y":.05,"z":.14} / {"radius":.03,"height":.1}
    json_path: str | None = None            # glove-made shape (touchmap result.json, mesh or point cloud)
    use_fit: bool = False                   # touchmap json: rebuild the fitted primitive instead of the hull
    card_path: str | None = None            # a full object card .json (overrides the physical fields below)
    pos: tuple | None = None                # x, y (m); default: 0.35, -0.20 (or where the glove saw it)
    yaw_deg: float = 0.0
    mass_kg: float = 0.35
    material: str = "aluminium"
    mu_pad: float | None = 0.5
    max_force_N: float = 25.0
    hollow: bool = False
    name: str | None = None


def build_card(spec: Spec) -> tuple[Card, Shape | None, str]:
    """Returns (card with in-memory mesh, shape, human description)."""
    if spec.card_path:
        c = load_card(spec.card_path)
        desc = f"card {Path(spec.card_path).name}"
        return c, None, desc
    note = ""
    centre = None
    if spec.json_path:
        shape, centre = load_shape_json(spec.json_path, "fit" if spec.use_fit else "cloud")
        name = spec.name or Path(spec.json_path).parent.name or Path(spec.json_path).stem
        note = f"{shape.source}: {shape.note}"
    else:
        shape = make_shape(spec.shape or "cylinder", **(spec.dims or {}))
        name = spec.name or shape.kind
        note = "parametric " + shape.kind + " " + ", ".join(f"{k}={v*1000:.0f}mm" for k, v in shape.params.items())
    if spec.pos is not None:
        x, y = spec.pos
    elif centre is not None:
        x, y = float(centre[0]), float(centre[1])
        note += f"; placed where the glove saw it ({x:.3f}, {y:.3f})"
    else:
        x, y = 0.35, -0.20
    h = np.radians(spec.yaw_deg) / 2
    c = card_from_dict({"mesh": f"{name}.obj",
                        "pose": {"pos_m": [x, y, 0.0], "quat_wxyz": [float(np.cos(h)), 0, 0, float(np.sin(h))]},
                        "mass_kg": spec.mass_kg, "material": spec.material, "mu_pad": spec.mu_pad,
                        "max_force_N": spec.max_force_N, "hollow": spec.hollow}, None, name)
    c.mesh = shape.mesh
    return c, shape, note


def annotate(frames, log, title, result, fps=30):
    """Burn a caption (live grip force, time, final outcome) into the video frames."""
    t = np.array(log["t"]) if log["t"] else np.zeros(1)
    f = np.array(log["f"]) if log["f"] else np.zeros((1, 3))
    big, small = _font(22), _font(15)
    color = {"SUCCESS": (60, 200, 90), "SLIPPED": (240, 180, 40), "DROPPED": (230, 70, 60), "BROKEN": (230, 70, 60)}
    out = []
    n = len(frames)
    for i, fr in enumerate(frames):
        im = Image.fromarray(fr)
        dr = ImageDraw.Draw(im, "RGBA")
        k = int(np.clip(np.searchsorted(t, i / fps), 0, len(t) - 1))
        th, fi, pk = f[k]
        grip = 0.5 * (th + fi)
        dr.rectangle([0, 0, im.width, 54], fill=(0, 0, 0, 150))
        dr.text((10, 6), title, fill=(255, 255, 255), font=big)
        dr.text((10, 32), f"t = {i / fps:4.1f} s   grip {grip:5.1f} N (target {result.target_force_N:.1f}, limit {result.max_force_N:g})   "
                f"peak contact {pk:5.1f} N", fill=(220, 220, 220), font=small)
        if i > n - int(2.0 * fps):
            dr.rectangle([0, im.height - 44, im.width, im.height], fill=(0, 0, 0, 170))
            dr.text((10, im.height - 38), result.outcome, fill=color.get(result.outcome, (255, 255, 255)), font=big)
            dr.text((150, im.height - 34), f"peak {result.peak_force_N:.1f}/{result.max_force_N:g} N   slip {result.slip_mm:.1f} mm   "
                    f"lifted {result.lift_height_mm:.0f} mm", fill=(230, 230, 230), font=small)
        out.append(np.asarray(im))
    return out


def montage(frames, n=6, cols=3):
    idx = np.linspace(int(len(frames) * 0.12), len(frames) - 1, n).astype(int)
    rows = [np.hstack([frames[i] for i in idx[r:r + cols]]) for r in range(0, n, cols)]
    return np.vstack(rows)


def execute(spec: Spec, out_dir: Path | None = None, video: bool = True, check_only: bool = False,
            log=print, attempts: int = 3) -> dict:
    card, shape, desc = build_card(spec)
    name = card.name
    out = Path(out_dir) if out_dir else RUNS / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    mesh = load_card_mesh(card) if not card.missing or card.mesh is not None else None
    result = {"name": name, "source": desc}
    if mesh is not None:
        ext = (mesh.bounds[1] - mesh.bounds[0]) * 1000
        result["object_extents_mm"] = [round(float(e), 1) for e in ext]
        shape_for_obj = Shape(mesh, "mesh", {})
        save_obj(shape_for_obj, out / f"{name}.obj")
        cj = card.to_json()
        cj["mesh"] = f"{name}.obj"
        (out / "card.json").write_text(json.dumps(cj, indent=2))
    log(f"[{name}] {desc}")
    t0 = time.time()
    v = check(card)
    result["feasibility"] = public(v)
    log(f"[{name}] feasibility: {v['verdict']} - {v['detail']}" + (
        f" | {v['grasp']['kind']} grasp, target {v['target_force_N']} N" if v["verdict"] == "CAN_PICK" else ""))
    if v["verdict"] == "CAN_PICK" and not check_only:
        res, tried = run_best(card, v, attempts=attempts, render=video, log=log)
        result["simulation"] = res.to_json()
        result["simulation"]["grasp_used"] = res.grasp
        result["simulation"]["attempts"] = [{"grasp": g, "outcome": o} for g, o in tried]
        log(f"[{name}] lift: {res.outcome}  peak {res.peak_force_N:.1f}/{res.max_force_N:g} N  slip {res.slip_mm:.1f} mm  "
            f"lifted {res.lift_height_mm:.0f} mm  ({time.time() - t0:.0f} s)")
        for n in res.notes:
            log(f"      note: {n}")
        if video and res.frames:
            frames = annotate(res.frames, res.log, name, res)
            imageio.mimwrite(out / "video.mp4", frames, fps=30, quality=8)
            Image.fromarray(montage(frames)).save(out / "preview.png")
    result["output_dir"] = str(out)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result
