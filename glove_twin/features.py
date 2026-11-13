"""Turn one tap into a small feature vector.

Impact features (piezo + FSR, always available):
  contact_ms          how long the finger stays in contact; hard objects bounce back fast
  impact_centroid_hz  spectral centre of the impact transient; sharper for harder objects
  amp_per_n           piezo transient amplitude per newton of tap force

Ring features (microphone, optional):
  q_factor   damping of the object's ringing, Q = pi * f * tau. Mostly a property of the
             material, not the object's size, so it is the strongest single clue.
  ring_hz    dominant resonance. Depends on size and shape too, so it is weighted low.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .tap import TapEvent

MIN_HZ = 50.0
DEAD_Q = 1.0           # encoding for "no audible ring" (foam, sponge, hand on a cushion)


@dataclass
class TapFeatures:
    peak_force_n: float
    contact_ms: float
    impact_centroid_hz: float
    amp_per_n: float
    ring_hz: float | None = None
    q_factor: float | None = None

    @property
    def has_ring(self) -> bool:
        return self.q_factor is not None and self.ring_hz is not None

    def vector(self, use_ring: bool) -> np.ndarray:
        v = [np.log(self.contact_ms), np.log(self.impact_centroid_hz), np.log(self.amp_per_n)]
        if use_ring:
            if not self.has_ring:
                raise ValueError("This tap has no microphone channel, but the classifier needs one")
            v += [np.log(self.q_factor), np.log(self.ring_hz)]
        return np.asarray(v, float)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TapFeatures":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


def _spectrum(x: np.ndarray, fs: int, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    nfft = max(nfft, 1 << int(np.ceil(np.log2(len(x)))))
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = freqs >= MIN_HZ
    return freqs[keep], mag[keep]


def _peak_hz(freqs: np.ndarray, mag: np.ndarray) -> float:
    k = int(np.argmax(mag))
    if 0 < k < len(mag) - 1:  # parabolic interpolation on log magnitude
        a, b, c = np.log(mag[k - 1:k + 2] + 1e-12)
        denom = a - 2 * b + c
        offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float(freqs[k] + offset * (freqs[1] - freqs[0]))
    return float(freqs[k])


def _ring(mic: np.ndarray, fs: int, start: int, noise: float) -> tuple[float, float]:
    seg = mic[start:]
    if len(seg) < 128 or np.max(np.abs(seg)) < 6 * noise:
        return MIN_HZ, DEAD_Q
    freqs, mag = _spectrum(seg, fs, 8192)
    ring_hz = max(_peak_hz(freqs, mag), MIN_HZ)

    win = max(4, int(1.5 * fs / ring_hz))
    if len(seg) <= win:
        return ring_hz, DEAD_Q
    env = sliding_window_view(np.abs(seg), win).max(axis=1)
    valid = env > 4 * noise
    if not valid[0]:
        return ring_hz, DEAD_Q
    run = int(np.argmin(valid)) if not valid.all() else len(valid)  # first contiguous run
    if run < max(3 * win, 32):
        return ring_hz, DEAD_Q
    t = np.arange(run) / fs
    slope = np.polyfit(t, np.log(env[:run]), 1)[0]
    tau = -1.0 / slope if slope < 0 else 20.0 * len(seg) / fs
    return ring_hz, float(np.clip(np.pi * ring_hz * tau, DEAD_Q, 1e5))


def extract_features(tap: TapEvent) -> TapFeatures:
    fs, pre = tap.fs, max(8, tap.pretrigger)
    force = tap.force_n - np.median(tap.force_n[:pre])
    piezo = tap.piezo - np.mean(tap.piezo[:pre])
    piezo_noise = max(float(np.std(piezo[:pre])), 1e-5)

    peak_force = float(max(force.max(), 1e-3))
    above = np.flatnonzero(force > 0.2 * peak_force)
    loud = np.flatnonzero(np.abs(piezo) > 6 * piezo_noise)
    if above.size:
        gaps = np.flatnonzero(np.diff(above) > 1)
        f_onset, f_end = int(above[0]), int(above[gaps[0]] if gaps.size else above[-1])
    else:
        f_onset = f_end = int(np.argmax(np.abs(piezo)))
    onset = min(f_onset, int(loud[0])) if loud.size else f_onset
    contact_ms = (f_end - f_onset + 1) / fs * 1000.0

    impact = piezo[max(onset - 4, 0):f_end + int(0.001 * fs) + 1]
    freqs, mag = _spectrum(impact, fs, 2048)
    centroid = float(np.sum(freqs * mag) / max(np.sum(mag), 1e-12))
    amp_per_n = max(float(np.max(np.abs(impact))), piezo_noise) / peak_force

    ring_hz = q = None
    if tap.mic is not None:
        mic = tap.mic - np.mean(tap.mic[:pre])
        mic_noise = max(float(np.std(mic[:pre])), 1e-5)
        ring_hz, q = _ring(mic, fs, f_end + int(0.0005 * fs), mic_noise)

    return TapFeatures(peak_force_n=peak_force, contact_ms=contact_ms,
                       impact_centroid_hz=max(centroid, MIN_HZ), amp_per_n=amp_per_n,
                       ring_hz=ring_hz, q_factor=q)
