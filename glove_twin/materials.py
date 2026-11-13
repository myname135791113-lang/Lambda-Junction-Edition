"""Material classes and the physical priors attached to them.

Tapping does not measure density directly. It identifies the *material class*
(steel rings for a long time, foam is dead, glass is sharp and bright), and each
class carries a typical density, friction and fragility. True bulk density needs
mass and volume as well (lift with the FSRs + a phone scan); see density.py.

All numbers are rough, order-of-magnitude priors for hand-sized household objects.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    density: float                       # kg/m^3, typical for the solid material
    density_range: tuple[float, float]   # kg/m^3, plausible spread
    friction: float                      # sliding friction vs. a rubber robot fingertip
    max_grip_n: float                    # fragility prior: rough max safe grip force (N)
    # Tap-response priors (used by the simulator, and so by the uncalibrated classifier)
    contact_ms: float    # contact duration for a hard (fingernail) tap
    loss_factor: float   # damping eta; quality factor Q ~ 1 / eta (geometry-independent)
    ring_hz: float       # typical first resonance of a hand-sized object (geometry-dependent)
    coupling: float      # how loudly the object rings per newton of tap force (relative)


MATERIALS: dict[str, Material] = {m.name: m for m in [
    Material("steel",        7850, (7600, 8100), 0.45, 200, 0.25, 0.001, 2600, 1.00),
    Material("aluminum",     2700, (2600, 2850), 0.45, 120, 0.30, 0.002, 2200, 0.90),
    Material("glass",        2500, (2200, 2800), 0.55,  15, 0.25, 0.004, 3200, 0.80),
    Material("ceramic",      2300, (1900, 2600), 0.60,  20, 0.30, 0.008, 1900, 0.70),
    Material("hard_plastic", 1100, ( 900, 1400), 0.40,  30, 0.60, 0.030, 1100, 0.45),
    Material("wood",          650, ( 400,  900), 0.50,  80, 0.50, 0.015,  800, 0.40),
    Material("cardboard",     300, ( 100,  700), 0.50,  10, 1.50, 0.080,  350, 0.20),
    Material("foam",           40, (  15,  100), 0.80,   5, 6.00, 0.300,  150, 0.05),
]}

# Values MuJoCo uses when nothing is known (the "default physics" twin, as in RialTo).
DEFAULT_DENSITY = 1000.0
DEFAULT_FRICTION = 1.0


def get_material(name: str | Material) -> Material:
    if isinstance(name, Material):
        return name
    try:
        return MATERIALS[name]
    except KeyError:
        raise ValueError(f"Unknown material {name!r}. Known: {', '.join(MATERIALS)}") from None
