"""Object card: everything the robot hand needs to grab this kind of object, as JSON.

Shape and size come from touch mapping, material from taps, mass from the lift-and-hold test
(fingertip FSRs) or a scale. Mass is never derived from the material: a can classed as
"aluminium" would otherwise weigh ~790 g instead of ~270 g.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .density import estimate_density
from .materials import CONTAINER_MATERIALS, MATERIALS
from .robot_sim import plan_grip_force
from .shape_fit import ShapeResult

CARD_VERSION = 1
HAND_MAX_APERTURE_M = 0.12   # widest object the robot hand can wrap (Rokoko-size hand)
REFERENCES = {
    "glove_accuracy": "Rokoko Smartgloves II datasheet: finger position accuracy 1.60 mm typ, 83.33 Hz",
    "skin_friction": "Derler & Gerhardt 2012, Tribology Letters 45: dry skin friction ~0.5",
    "exploration": "Lederman & Klatzky 1987: enclosure -> global shape, contour following -> exact shape",
}


def fitted_volume_m3(shape: ShapeResult) -> float:
    d = shape.dims
    if shape.best == "sphere":
        return 4 / 3 * np.pi * d["radius"] ** 3
    if shape.best == "cylinder":
        return np.pi * d["radius"] ** 2 * d["height"]
    return d["width"] * d["depth"] * d["height"]


def choose_grasp(shape: ShapeResult, mass_kg: float, friction: float, max_grip_n: float) -> dict:
    d = shape.dims
    if shape.best == "cylinder":
        g = {"type": "power_wrap", "approach": "side", "wrap_axis": "object long axis",
             "grip_width_m": 2 * d["radius"], "preshape": "open"}
    elif shape.best == "box":
        g = {"type": "box_wrap", "approach": "side", "wrap_axis": "object long axis",
             "grip_width_m": d["width"], "palm_on": "a face across the narrowest width", "preshape": "open"}
    else:
        g = {"type": "spherical_power", "approach": "side", "grip_width_m": 2 * d["radius"],
             "preshape": "sphere"}
    g["feasible"] = bool(g["grip_width_m"] <= HAND_MAX_APERTURE_M)
    g["grip_force_n"] = round(plan_grip_force(mass_kg, friction, max_grip_n), 2)
    g["grip_force_rule"] = "1.5 x weight / (2 x friction), capped at 80% of the safe squeeze"
    return g


def build_card(name: str, label: str, shape: ShapeResult, *, mass_kg: float, mass_source: str,
               material_probs: dict, friction: float | None = None, n_contacts: int = 0,
               n_grasps: int = 0, follow_ups: int = 0) -> dict:
    best_mat = max(material_probs, key=material_probs.get)
    plausible = [m for m, p in material_probs.items() if p >= 0.1] or [best_mat]
    mu = friction if friction is not None else float(sum(p * MATERIALS[m].friction for m, p in material_probs.items()))
    max_grip = float(min(MATERIALS[m].max_grip_n for m in plausible))
    volume = fitted_volume_m3(shape)
    density = estimate_density(material_probs, mass_kg * 1000, volume * 1e6)
    return {
        "card_version": CARD_VERSION,
        "object": name,
        "label": label,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shape": {
            "primitive": shape.best,
            "dims_m": {k: round(v, 4) for k, v in shape.dims.items()},
            "sigma_m": {k: (round(v, 4) if np.isfinite(v) else None) for k, v in shape.sigma.items()},
            "unmeasured": [k for k, v in shape.sigma.items() if not np.isfinite(v)],
            "model_weights": {k: round(v, 3) for k, v in shape.weights.items()},
            "ambiguous": shape.ambiguous,
            "reasons": shape.reasons,
            "touch_here_next": shape.touch_here,
            "contacts": n_contacts, "grasps": n_grasps, "follow_up_rounds": follow_ups,
        },
        "mass_kg": round(mass_kg, 4),
        "mass_source": mass_source,
        "volume_m3": round(volume, 7),
        "bulk_density_kg_m3": round(density.value, 1),
        "container": best_mat in CONTAINER_MATERIALS,
        "material": {"best": best_mat, "probabilities": {k: round(v, 3) for k, v in material_probs.items()}},
        "friction": round(mu, 3),
        "friction_source": "slide-until-slip test" if friction is not None else "material prior",
        "max_safe_grip_n": max_grip,
        "fragile": max_grip <= 20,
        "grasp": choose_grasp(shape, mass_kg, mu, max_grip),
        "references": REFERENCES,
    }


def validate_card(card: dict) -> list[str]:
    """Light schema check; returns a list of problems (empty = valid)."""
    problems = []
    for key in ("card_version", "object", "shape", "mass_kg", "material", "friction", "max_safe_grip_n", "grasp"):
        if key not in card:
            problems.append(f"missing {key}")
    if "shape" in card:
        prim = card["shape"].get("primitive")
        need = {"sphere": {"radius"}, "cylinder": {"radius", "height"}, "box": {"width", "depth", "height"}}
        if prim not in need:
            problems.append(f"unknown primitive {prim}")
        elif set(card["shape"].get("dims_m", {})) != need[prim]:
            problems.append("dims do not match the primitive")
    if card.get("mass_kg", 0) <= 0:
        problems.append("mass must be positive")
    if "grasp" in card and card["grasp"].get("grip_force_n", 0) <= 0:
        problems.append("grip force must be positive")
    return problems


def write_card(card: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{card['object']}.card.json"
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return path
