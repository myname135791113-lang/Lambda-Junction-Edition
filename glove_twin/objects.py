"""Reference catalog of the model objects, with sources.

These are the ground truth for the simulator and the accuracy checks. The mapping pipeline never
reads them; it has to recover shape and size from touch alone.

All lengths in metres, masses in kg. Each figure names its source. Values marked "estimate" are
derived (the arithmetic is written out) and should be replaced by measuring the real object.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SOURCES = {
    "crown_250_slim": "https://crowncork.com/beverage-packaging/products/beverage-cans/83oz-250ml-slim",
    "ball_design_guide": "https://printguides.ball.com/whats-available/Europe",
    "redbull": "https://www.redbull.com/int-en/energydrink/red-bull-energy-drink",
    "prisma_profeel": "https://www.prisma.fi/tuotteet/111163612/valio-profeel-proteiinijuoma-salted-caramel-250-ml-uht-laktoositon-111163612",
    "valio_tetra_edge": "https://www.packaginginsights.com/news/valio-revitalises-dairy-products-with-tetra-brik-aseptic-edge.html",
    "usda_wood_handbook": "https://research.fs.usda.gov/treesearch/37427",
    "rokoko_sg2_datasheet": "https://sdk.rokoko.com/datasheets/Smartgloves_II_Technical_Specifications_v1.0.0_External.pdf",
    "skin_friction_review": "https://doc.rero.ch/record/315296",
}

# USDA Wood Handbook (FPL-GTR-190) Table 5-3: specific gravity at 12 % moisture content, based on
# oven-dry weight and volume at 12 % MC. Density at 12 % MC = SG x (1 + 0.12) x 1000 kg/m^3.
WOOD_SG_12 = {"beech (American)": 0.64, "birch (paper)": 0.55, "pine (red)": 0.46}
WOOD_DENSITY = {k: sg * 1.12 * 1000.0 for k, sg in WOOD_SG_12.items()}
WOOD_DENSITY_AVG = float(np.mean(list(WOOD_DENSITY.values())))  # the user asked for the average


@dataclass(frozen=True)
class RefObject:
    key: str
    label: str
    shape: str                       # "box" | "cylinder" | "sphere"
    dims: dict                       # cylinder: radius, height; box: x, y, z; sphere: radius
    mass_kg: float
    mass_note: str
    material: str                    # key into materials.MATERIALS
    friction: float                  # assumed true friction vs. skin / robot pads (dry skin ~0.5)
    max_grip_n: float                # safe squeeze; assumption until a squeeze test is done
    container: bool = False          # full sealed container: bulk density, not material density
    sources: tuple = ()
    notes: tuple = ()
    sloped_top_m: float = 0.0        # box only: height drop of the sloped "Edge" top panel

    @property
    def volume_m3(self) -> float:
        d = self.dims
        if self.shape == "sphere":
            return 4.0 / 3.0 * np.pi * d["radius"] ** 3
        if self.shape == "cylinder":
            return np.pi * d["radius"] ** 2 * d["height"]
        return d["x"] * d["y"] * d["z"]

    @property
    def bulk_density(self) -> float:
        return self.mass_kg / self.volume_m3

    def dims_vector(self) -> np.ndarray:
        """Canonical shape parameters used by shape_fit (sorted box extents)."""
        d = self.dims
        if self.shape == "sphere":
            return np.array([d["radius"]])
        if self.shape == "cylinder":
            return np.array([d["radius"], d["height"]])
        return np.sort([d["x"], d["y"], d["z"]])

    def extents(self) -> np.ndarray:
        """Full bounding-box extents x, y, z with the object standing upright."""
        d = self.dims
        if self.shape == "sphere":
            return np.full(3, 2 * d["radius"])
        if self.shape == "cylinder":
            return np.array([2 * d["radius"], 2 * d["radius"], d["height"]])
        return np.array([d["x"], d["y"], d["z"]])


RED_BULL_LIQUID_DENSITY = 1040.0   # estimate: ~11 % sugar solution (27 g sugar / 250 ml, redbull.com)
RED_BULL_CAN_MASS = 0.010          # estimate: empty 250 ml slim aluminium can, ~10 g

OBJECTS: dict[str, RefObject] = {o.key: o for o in [
    RefObject(
        key="redbull", label="Red Bull 250 ml can (full)", shape="cylinder",
        dims={"radius": 0.053 / 2, "height": 0.133},
        mass_kg=0.250e-3 * RED_BULL_LIQUID_DENSITY + RED_BULL_CAN_MASS,
        mass_note="estimate: 250 ml x 1.04 g/ml + ~10 g can = ~270 g. Weigh the real can.",
        material="filled_can", friction=0.40, max_grip_n=120.0, container=True,
        sources=("crown_250_slim", "ball_design_guide", "redbull"),
        notes=("Crown 250 ml slim: diameter 53 mm, height 133 mm (202 diameter, 200 end).",
               "Necked top and domed base: the fitted height is the overall height."),
    ),
    RefObject(
        key="profeel", label="Valio PROfeel 250 ml carton (full)", shape="box",
        dims={"x": 0.0478, "y": 0.0475, "z": 0.140},
        mass_kg=0.2658,
        mass_note="Prisma.fi: product weight 0.2658 kg (listed packaging weight 0.2783 kg).",
        material="filled_carton", friction=0.50, max_grip_n=20.0, container=True,
        sources=("prisma_profeel", "valio_tetra_edge"),
        notes=("Prisma.fi package size: width 4.78 cm, height 14 cm, depth 4.75 cm.",
               "Tetra Brik Aseptic 250 Edge: the top panel is sloped, so height = maximum extent.",
               "max_grip 20 N is an assumption for a full sealed carton; measure with a squeeze test."),
        sloped_top_m=0.010,
    ),
    RefObject(
        key="wood_sphere", label="Wooden sphere, r = 50 mm", shape="sphere",
        dims={"radius": 0.050},
        mass_kg=4.0 / 3.0 * np.pi * 0.050 ** 3 * WOOD_DENSITY_AVG,
        mass_note=("average of beech, paper birch and red pine (USDA Wood Handbook, SG at 12 % MC "
                   "x 1.12): ~323 g. Range ~270-375 g depending on species."),
        material="wood", friction=0.50, max_grip_n=80.0,
        sources=("usda_wood_handbook",),
        notes=("Radius given by the user: 50 mm, so V = 523.6 cm^3.",),
    ),
]}


def get_object(key: str) -> RefObject:
    try:
        return OBJECTS[key]
    except KeyError:
        raise ValueError(f"Unknown object {key!r}. Known: {', '.join(OBJECTS)}") from None
