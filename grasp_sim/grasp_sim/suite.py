"""The spec's test set. Sizes and masses are typical real-world values (assumptions, change freely)."""
from __future__ import annotations

import numpy as np
import trimesh

from .card import Card, card_from_dict
from .shapes import Shape, ground, make_shape


def _cup() -> Shape:
    """Paper cup: truncated cone, 80 mm top, 56 mm bottom, 95 mm tall."""
    m = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=48)
    v = m.vertices.copy()
    top = v[:, 2] > 0
    v[:, :2] *= np.where(top, 0.040, 0.028)[:, None]
    v[:, 2] = np.where(top, 0.095, 0.0)
    m = trimesh.Trimesh(v, m.faces, process=True)
    m, _ = ground(m)
    return Shape(m, "mesh", {"top_radius": 0.040, "bottom_radius": 0.028, "height": 0.095}, "suite", "truncated cone")


def make_card(shape: Shape, mass_kg: float, material: str, mu_pad: float, max_force_N: float, hollow: bool,
              x=0.35, y=-0.20, z=0.0, name="object", yaw_deg=0.0) -> Card:
    h = np.radians(yaw_deg) / 2
    c = card_from_dict({"mesh": f"{name}.obj",
                        "pose": {"pos_m": [x, y, z], "quat_wxyz": [float(np.cos(h)), 0.0, 0.0, float(np.sin(h))]},
                        "mass_kg": mass_kg, "material": material, "mu_pad": mu_pad,
                        "max_force_N": max_force_N, "hollow": hollow}, None, name)
    c.mesh = shape.mesh
    return c


def test_set() -> dict[str, tuple[Card, Shape]]:
    S = make_shape
    can = S("cylinder", radius=0.033, height=0.115)
    out = {
        "ping_pong_ball": (S("sphere", radius=0.020), dict(mass_kg=0.0027, material="plastic", mu_pad=0.5, max_force_N=8, hollow=True)),
        "tennis_ball": (S("sphere", radius=0.0335), dict(mass_kg=0.057, material="rubber", mu_pad=0.8, max_force_N=40, hollow=True)),
        "empty_can": (can, dict(mass_kg=0.015, material="aluminium", mu_pad=0.45, max_force_N=12, hollow=True)),
        "full_can": (can, dict(mass_kg=0.35, material="aluminium", mu_pad=0.5, max_force_N=25, hollow=False)),
        "tissue_box": (S("box", x=0.065, y=0.065, z=0.09), dict(mass_kg=0.07, material="cardboard", mu_pad=0.6, max_force_N=25, hollow=True, yaw_deg=-30)),
        "paper_cup": (_cup(), dict(mass_kg=0.012, material="paper", mu_pad=0.6, max_force_N=6, hollow=True)),
        "out_of_reach_can": (can, dict(mass_kg=0.35, material="aluminium", mu_pad=0.5, max_force_N=25, hollow=False, x=0.95, y=-0.20)),
        "heavy_block": (S("box", x=0.06, y=0.06, z=0.10), dict(mass_kg=0.8, material="wood", mu_pad=0.55, max_force_N=200, hollow=False)),
    }
    return {k: (make_card(s, name=k, **p), s) for k, (s, p) in out.items()}
