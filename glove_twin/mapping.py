"""Touch mapping session: explore -> reconstruct -> fit, asking for more touches while unclear."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .explore_sim import Session
from .objects import OBJECTS, RefObject
from .shape_fit import Grasp, ShapeResult, fit_shape
from .touch import Frame, calibrate_pad_offset, reconstruct


@dataclass
class MappingResult:
    obj_key: str
    shape: ShapeResult
    grasps: list[Grasp]
    frames: list[Frame]
    follow_ups: int
    history: list = field(default_factory=list)   # (n_grasps, best, ambiguous) per fit round
    images: list = field(default_factory=list)
    records: list = field(default_factory=list)


def calibrate(rng: np.random.Generator, reference: RefObject | None = None) -> dict:
    """One-time pad-offset calibration on a known cylinder (default: the Red Bull can)."""
    ref = reference or OBJECTS["redbull"]
    s = Session(ref, rng)
    s.routine()
    return calibrate_pad_offset(s.frames, ref.dims["radius"])


def map_object(obj: RefObject, rng: np.random.Generator, *, pad_offset: dict | float = 0.0,
               max_follow_ups: int = 2, render: bool = False) -> MappingResult:
    """Standard routine, then up to `max_follow_ups` rounds of extra touches while ambiguous."""
    s = Session(obj, rng, render=render)
    s.routine()
    history = []
    k = 0
    while True:
        grasps = reconstruct(s.frames, pad_offset=pad_offset)
        res = fit_shape(grasps)
        history.append((len(grasps), res.best, res.ambiguous))
        if not res.ambiguous or k >= max_follow_ups:
            break
        s.follow_up(res.reasons)
        k += 1
    s.close()
    return MappingResult(obj.key, res, grasps, s.frames, k, history, s.images, s.records)


_TAP_CLASSIFIER = None


def physical_tests(obj: RefObject, rng: np.random.Generator, taps: int = 5) -> dict:
    """The other glove tests, simulated: lift and hold (mass), slide until slip (friction), taps.

    Error levels: FSR lift +-10 % (FSRs drift and have hysteresis), slide test +-10 %.
    """
    global _TAP_CLASSIFIER
    from .classifier import TapClassifier
    from .features import extract_features
    from .simulate import simulate_tap
    if _TAP_CLASSIFIER is None:
        _TAP_CLASSIFIER = TapClassifier.synthetic_prior(seed=0)
    feats = [extract_features(simulate_tap(obj.material, rng)) for _ in range(taps)]
    return {
        "mass_kg": float(obj.mass_kg * np.clip(rng.normal(1.0, 0.05), 0.9, 1.1)),
        "mass_source": "lift and hold, fingertip FSRs (simulated, +-10 %)",
        "friction": float(obj.friction * np.clip(rng.normal(1.0, 0.05), 0.9, 1.1)),
        "material_probs": _TAP_CLASSIFIER.predict_session(feats),
    }


def make_card(obj: RefObject, m: MappingResult, rng: np.random.Generator) -> dict:
    from .object_card import build_card
    tests = physical_tests(obj, rng)
    return build_card(obj.key, obj.label, m.shape, mass_kg=tests["mass_kg"], mass_source=tests["mass_source"],
                      material_probs=tests["material_probs"], friction=tests["friction"],
                      n_contacts=sum(len(g.points) for g in m.grasps), n_grasps=len(m.grasps),
                      follow_ups=m.follow_ups)
