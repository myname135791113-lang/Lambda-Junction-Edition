"""Density estimate from tapping, refined with mass and volume when available.

Tap only  -> material-class density (a prior; a hollow mug is "ceramic", not "ceramic-dense").
+ mass and volume -> measured bulk density, and solidity = bulk / material density, which tells
                     a solid object from a hollow shell or a container.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .materials import MATERIALS


@dataclass
class DensityEstimate:
    value: float                 # kg/m^3, what the twin uses
    low: float
    high: float
    source: str                  # "material prior" or "mass / volume"
    material_density: float      # kg/m^3 implied by the material probabilities
    solidity: float | None       # bulk / material density, when mass and volume are known
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def material_density(probs: dict[str, float]) -> tuple[float, float, float]:
    """Probability-weighted geometric mean, plus a range over the plausible classes."""
    value = float(np.exp(sum(p * np.log(MATERIALS[m].density) for m, p in probs.items())))
    likely = [m for m, p in probs.items() if p >= 0.1] or [max(probs, key=probs.get)]
    low = min(MATERIALS[m].density_range[0] for m in likely)
    high = max(MATERIALS[m].density_range[1] for m in likely)
    return value, float(low), float(high)


def _solidity_note(s: float) -> str:
    if s > 1.3:
        return "denser than the material guess: filled with something heavy, or the material guess is wrong"
    if s >= 0.75:
        return "solid object"
    if s >= 0.3:
        return "partly hollow or porous (or the volume is a loose bounding box)"
    return "hollow shell or container (cup, bottle, box); use the measured mass, not the material density"


def estimate_density(probs: dict[str, float], mass_g: float | None = None,
                     volume_cm3: float | None = None) -> DensityEstimate:
    mat, low, high = material_density(probs)
    if mass_g and volume_cm3:
        bulk = mass_g / volume_cm3 * 1000.0
        solidity = bulk / mat
        return DensityEstimate(bulk, bulk * 0.9, bulk * 1.1, "mass / volume", mat, solidity,
                               _solidity_note(solidity))
    return DensityEstimate(mat, low, high, "material prior", mat, None,
                           "tap only: assumes a solid object. Add --mass-g and a volume for hollow objects")
