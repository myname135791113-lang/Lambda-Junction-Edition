"""Material classifier trained on *your own* taps (k-nearest neighbours).

Calibrate by tapping objects whose material you know. Until then, a synthetic prior built
from the simulator is used. It works for the simulator, but real hardware needs calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import TapFeatures, extract_features
from .materials import MATERIALS
from .simulate import simulate_tap

# Relative importance of [contact, impact centroid, amp/N, (Q, ring Hz)]
IMPACT_WEIGHTS = np.array([1.0, 0.7, 0.7])
RING_WEIGHTS = np.array([1.2, 0.4])
FLOOR = 0.03  # probability mass spread over all classes so no tap is ever "impossible"


class TapClassifier:
    def __init__(self, samples: list[tuple[TapFeatures, str]], k: int = 5, source: str = "calibration"):
        if not samples:
            raise ValueError("Need at least one labelled tap")
        self.use_ring = all(f.has_ring for f, _ in samples)
        X = np.array([f.vector(self.use_ring) for f, _ in samples])
        self.labels = np.array([label for _, label in samples])
        self.classes = sorted(set(self.labels))
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-6
        self.weights = np.concatenate([IMPACT_WEIGHTS, RING_WEIGHTS]) if self.use_ring else IMPACT_WEIGHTS
        self.X = self._scale(X)
        self.k = min(k, len(samples))
        self.source = source

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std * self.weights

    @classmethod
    def synthetic_prior(cls, taps_per_material: int = 40, with_mic: bool = True,
                        finger: str = "nail", seed: int = 0) -> "TapClassifier":
        rng = np.random.default_rng(seed)
        samples = [(extract_features(simulate_tap(name, rng, with_mic=with_mic, finger=finger)), name)
                   for name in MATERIALS for _ in range(taps_per_material)]
        return cls(samples, source="synthetic prior (simulator)")

    def predict_proba(self, feat: TapFeatures) -> dict[str, float]:
        d = np.linalg.norm(self.X - self._scale(feat.vector(self.use_ring)[None, :]), axis=1)
        idx = np.argsort(d)[:self.k]
        w = 1.0 / (d[idx] + 1e-3)
        p = {c: 0.0 for c in self.classes}
        for label, weight in zip(self.labels[idx], w):
            p[label] += weight
        total = sum(p.values())
        return {c: (1 - FLOOR) * v / total + FLOOR / len(self.classes) for c, v in p.items()}

    def predict_session(self, feats: list[TapFeatures]) -> dict[str, float]:
        """Combine several taps on the same object (naive-Bayes product of per-tap probabilities)."""
        logp = {c: 0.0 for c in self.classes}
        for f in feats:
            for c, v in self.predict_proba(f).items():
                logp[c] += np.log(v)
        top = max(logp.values())
        p = {c: np.exp(v - top) for c, v in logp.items()}
        total = sum(p.values())
        return dict(sorted(((c, float(v / total)) for c, v in p.items()), key=lambda kv: -kv[1]))


def load_samples(path: str | Path) -> list[tuple[TapFeatures, str]]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(TapFeatures.from_dict(s["features"]), s["material"]) for s in data["samples"]]


def save_samples(path: str | Path, samples: list[tuple[TapFeatures, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"samples": [{"material": m, "features": f.to_dict()} for f, m in samples]}
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
