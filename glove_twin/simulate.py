"""Synthetic taps, so the whole pipeline runs without hardware.

The model is deliberately simple: a contact force pulse, a fingertip piezo that responds to
the rate of change of force, a slow FSR, and an object that rings with two damped modes heard
by a microphone. Per-tap randomness covers tap strength, object size (resonance frequency) and
damping spread.

`simulate_tap` uses an analytic half-sine force pulse. `synthesize_sensors` takes any contact
force trace instead, e.g. one computed by MuJoCo for the Rokoko hand (see robot_sim.py).
"""
from __future__ import annotations

import numpy as np

from .materials import Material, get_material
from .tap import TapEvent

FINGER_CONTACT_MS = {"nail": 0.3, "pad": 3.0}  # fingertip compliance added to the contact
PIEZO_GAIN = 1e-5      # normalised piezo units per (N/s)
MIC_GAIN = 0.2
FSR_TAU_S = 5e-4       # FSRs respond with roughly a millisecond of lag
NOISE = {"piezo": 1e-3, "mic": 1e-3, "force": 0.01}


def _lowpass(x: np.ndarray, fs: int, tau: float) -> np.ndarray:
    a = 1.0 - np.exp(-1.0 / (fs * tau))
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    return y


def synthesize_sensors(material: str | Material, force: np.ndarray, fs: int, pre: int,
                       rng: np.random.Generator, *, with_mic: bool = True,
                       piezo_gain: float = PIEZO_GAIN, mic_gain: float = MIC_GAIN) -> TapEvent:
    """Glove sensor readings for a true fingertip contact force trace (newtons).

    `force` must start at the trigger point; `pre` samples of quiet baseline are prepended.
    The gains are the hardware gain settings: lower them for harder taps so nothing clips.
    """
    m = get_material(material)
    force = np.maximum(np.asarray(force, float), 0.0)
    n = len(force) + pre
    peak = float(force.max()) if force.size else 0.0
    contact_ms = max(np.count_nonzero(force > 0.05 * peak) / fs * 1000.0, 0.05) if peak > 0 else 0.05

    piezo = piezo_gain * np.gradient(force, 1.0 / fs)
    force_meas = _lowpass(force, fs, FSR_TAU_S)

    mic = None
    if with_mic:
        ring_hz = m.ring_hz * rng.lognormal(0.0, 0.15)
        eta = m.loss_factor * rng.lognormal(0.0, 0.2)
        t = np.arange(len(force)) / fs
        excitation = m.coupling * peak * min(1.0, 0.5 / contact_ms)  # soft taps excite less
        ring = np.zeros_like(t)
        for f_mult, amp, damp_mult in ((1.0, 1.0, 1.0), (2.76, 0.4, 1.5)):
            f = ring_hz * f_mult
            if f < 0.45 * fs:
                tau = 1.0 / (np.pi * f * eta * damp_mult)
                ring += amp * np.exp(-t / tau) * np.sin(2 * np.pi * f * t)
        mic = mic_gain * excitation * ring

    def finish(x: np.ndarray, noise: float, clip: bool = True) -> np.ndarray:
        x = np.concatenate([np.zeros(pre), x]) + rng.normal(0.0, noise, n)
        return np.clip(x, -1.0, 1.0) if clip else x

    return TapEvent(
        fs=fs,
        piezo=finish(piezo, NOISE["piezo"]),
        force_n=np.maximum(finish(force_meas, NOISE["force"], clip=False), 0.0),
        mic=finish(mic, NOISE["mic"]) if mic is not None else None,
        pretrigger=pre,
    )


def simulate_tap(material: str | Material, rng: np.random.Generator | None = None, *,
                 fs: int = 16_000, n: int = 1024, pre: int = 64, finger: str = "nail",
                 force_n: float | None = None, with_mic: bool = True) -> TapEvent:
    m = get_material(material)
    rng = rng if rng is not None else np.random.default_rng()
    peak = force_n if force_n is not None else rng.uniform(0.5, 3.0)
    contact_ms = float(np.hypot(m.contact_ms, FINGER_CONTACT_MS[finger]) * rng.uniform(0.85, 1.15))
    t = np.arange(n - pre) / fs
    tc = contact_ms / 1000.0
    force = peak * np.where(t < tc, np.sin(np.pi * t / tc), 0.0)
    return synthesize_sensors(m, force, fs, pre, rng, with_mic=with_mic)
