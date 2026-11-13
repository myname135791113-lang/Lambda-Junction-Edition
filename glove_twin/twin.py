"""Write the digital twin: properties JSON plus two MuJoCo models for the A/B test.

  <name>.properties.json   everything the glove measured or inferred
  <name>.glove.xml         twin with glove-measured physics
  <name>.default.xml       same geometry with MuJoCo default physics (density 1000, friction 1),
                           the baseline used by RialTo
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .density import DensityEstimate
from .features import TapFeatures
from .materials import DEFAULT_DENSITY, DEFAULT_FRICTION, MATERIALS


def _mean_features(feats: list[TapFeatures]) -> dict:
    out = {}
    for key in TapFeatures.__dataclass_fields__:
        vals = [getattr(f, key) for f in feats if getattr(f, key) is not None]
        out[key] = float(np.median(vals)) if vals else None
    return out


def build_properties(name: str, probs: dict[str, float], feats: list[TapFeatures],
                     density: DensityEstimate, *, classifier_source: str,
                     mass_g: float | None = None, dims_m: tuple[float, float, float] | None = None,
                     volume_cm3: float | None = None, mesh: str | None = None) -> dict:
    best = max(probs, key=probs.get)
    # Friction and fragility: probability-weighted, but fragility takes the cautious end.
    friction = float(sum(p * MATERIALS[m].friction for m, p in probs.items()))
    plausible = [m for m, p in probs.items() if p >= 0.1] or [best]
    max_grip = float(min(MATERIALS[m].max_grip_n for m in plausible))
    return {
        "object": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "taps": len(feats),
        "classifier": classifier_source,
        "material": {"best": best, "probabilities": {m: round(p, 4) for m, p in probs.items()}},
        "density_kg_m3": density.to_dict(),
        "mass_g": mass_g,
        "volume_cm3": volume_cm3,
        "geometry": {"dims_m": list(dims_m) if dims_m else None, "mesh": mesh},
        "friction_prior": round(friction, 3),
        "robot_hints": {"max_grip_force_n": max_grip, "fragile": max_grip <= 20},
        "tap_features_median": _mean_features(feats),
    }


def mjcf(name: str, *, friction: float, density: float | None = None, mass_kg: float | None = None,
         dims_m: tuple[float, float, float] | None = None, mesh: str | None = None, variant: str) -> str:
    if mesh:
        asset = f'  <asset>\n    <mesh name="{name}_mesh" file={quoteattr(mesh)}/>\n  </asset>\n'
        shape = f'type="mesh" mesh="{name}_mesh"'
    else:
        dims_m = dims_m or (0.05, 0.05, 0.05)
        asset = ""
        shape = 'type="box" size="{:.4f} {:.4f} {:.4f}"'.format(*(d / 2 for d in dims_m))
    z = (dims_m[2] / 2 if dims_m else 0.1) + 0.002
    inertial = f'mass="{mass_kg:.5f}"' if mass_kg else f'density="{density:.1f}"'
    return (
        f'<mujoco model="{name}_{variant}">\n'
        f'  <option timestep="0.002"/>\n{asset}'
        f'  <worldbody>\n'
        f'    <light pos="0 0 1.5" dir="0 0 -1"/>\n'
        f'    <geom name="floor" type="plane" size="1 1 0.05"/>\n'
        f'    <body name="{name}" pos="0 0 {z:.4f}">\n'
        f'      <freejoint/>\n'
        f'      <geom name="{name}_geom" {shape} {inertial} friction="{friction:.3f} 0.005 0.0001"/>\n'
        f'    </body>\n'
        f'  </worldbody>\n'
        f'</mujoco>\n'
    )


def write_twin(props: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = props["object"]
    dims = tuple(props["geometry"]["dims_m"]) if props["geometry"]["dims_m"] else None
    mesh = props["geometry"]["mesh"]
    mass_kg = props["mass_g"] / 1000.0 if props["mass_g"] else None

    paths = {
        "properties": out / f"{name}.properties.json",
        "glove": out / f"{name}.glove.xml",
        "default": out / f"{name}.default.xml",
    }
    paths["properties"].write_text(json.dumps(props, indent=2), encoding="utf-8")
    paths["glove"].write_text(mjcf(name, friction=props["friction_prior"], mass_kg=mass_kg,
                                   density=props["density_kg_m3"]["value"], dims_m=dims, mesh=mesh,
                                   variant="glove"), encoding="utf-8")
    paths["default"].write_text(mjcf(name, friction=DEFAULT_FRICTION, density=DEFAULT_DENSITY,
                                     dims_m=dims, mesh=mesh, variant="default"), encoding="utf-8")
    return paths
