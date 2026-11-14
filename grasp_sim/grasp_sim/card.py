"""The object card (input of both the feasibility check and the lift simulation)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

# object-vs-robot-finger-pad friction, by object material (rubber-ish pad on the real hand).
# The glove measures object-vs-skin; the robot needs object-vs-pad, so this is a stand-in for a
# measured value. Used only when the card gives a material but no mu_pad.
PAD_FRICTION = {
    "aluminium": 0.45, "steel": 0.40, "glass": 0.45, "plastic": 0.50, "cardboard": 0.60, "paper": 0.60,
    "wood": 0.55, "rubber": 0.90, "felt": 0.70, "fabric": 0.60, "ceramic": 0.45,
}

REQUIRED = ("mesh", "pose", "mass_kg", "max_force_N")


@dataclass
class Card:
    mesh_path: str | None
    pos_m: np.ndarray
    quat_wxyz: np.ndarray
    mass_kg: float | None
    material: str | None
    mu_pad: float | None
    max_force_N: float | None
    hollow: bool = False
    name: str = "object"
    mesh: trimesh.Trimesh | None = None       # in-memory mesh (metres, min z = 0, xy-centred) overrides mesh_path
    raw: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    @property
    def friction(self) -> float | None:
        if self.mu_pad is not None:
            return float(self.mu_pad)
        if self.material and self.material.lower() in PAD_FRICTION:
            return PAD_FRICTION[self.material.lower()]
        return None

    def to_json(self) -> dict:
        d = {"mesh": self.mesh_path or f"{self.name}.obj",
             "pose": {"pos_m": [float(x) for x in self.pos_m], "quat_wxyz": [float(x) for x in self.quat_wxyz]},
             "mass_kg": self.mass_kg, "material": self.material, "mu_pad": self.mu_pad,
             "max_force_N": self.max_force_N, "hollow": self.hollow}
        return d


def card_from_dict(doc: dict, base: Path | None = None, name: str = "object") -> Card:
    pose = doc.get("pose") or {}
    pos = pose.get("pos_m")
    quat = pose.get("quat_wxyz", [1, 0, 0, 0])
    missing = []
    if not doc.get("mesh"):
        missing.append("mesh")
    if pos is None or len(pos) != 3:
        missing.append("pose.pos_m")
    for k in ("mass_kg", "max_force_N"):
        if doc.get(k) is None:
            missing.append(k)
    if doc.get("mu_pad") is None and not (doc.get("material") and doc["material"].lower() in PAD_FRICTION):
        missing.append("mu_pad (or a known material)")
    c = Card(
        mesh_path=doc.get("mesh"),
        pos_m=np.array(pos if pos is not None else [np.nan] * 3, float),
        quat_wxyz=np.array(quat, float),
        mass_kg=None if doc.get("mass_kg") is None else float(doc["mass_kg"]),
        material=doc.get("material"), mu_pad=None if doc.get("mu_pad") is None else float(doc["mu_pad"]),
        max_force_N=None if doc.get("max_force_N") is None else float(doc["max_force_N"]),
        hollow=bool(doc.get("hollow", False)), name=name, raw=doc, missing=missing)
    if c.mesh_path and base is not None:
        p = Path(c.mesh_path)
        c.mesh_path = str(p if p.is_absolute() else base / p)
    return c


def load_card(path: str | Path) -> Card:
    path = Path(path)
    return card_from_dict(json.loads(path.read_text()), path.parent, path.stem)


def load_card_mesh(card: Card) -> trimesh.Trimesh:
    """Mesh in metres, with the pose rotation applied, resting on z = 0 (before adding pose z)."""
    if card.mesh is not None:
        m = card.mesh.copy()
    else:
        m = trimesh.load(card.mesh_path, force="mesh")
        # card meshes are given in metres, origin anywhere; re-centre horizontally, rest on the table
        lo, hi = m.bounds
        m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0])
    q = card.quat_wxyz / np.linalg.norm(card.quat_wxyz)
    R = trimesh.transformations.quaternion_matrix(q)
    m.apply_transform(R)
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m
