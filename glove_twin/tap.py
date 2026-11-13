"""One captured tap, and parsing of the glove's serial packets."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class TapEvent:
    """Synchronised sensor windows around one tap.

    piezo:   fingertip piezo, normalised to -1..1 (0 = rest). Hears the impact itself.
    force_n: fingertip FSR, converted to newtons. Gives tap force and contact duration.
    mic:     optional microphone, normalised to -1..1. Hears the object *ringing* after the
             finger has left it, which the fingertip piezo cannot.
    pretrigger: number of samples recorded before the trigger (used as the noise baseline).
    """
    fs: int
    piezo: np.ndarray
    force_n: np.ndarray
    mic: np.ndarray | None = None
    pretrigger: int = 64


@dataclass(frozen=True)
class FsrCalibration:
    """FSR wired as: Vcc -- FSR -- ADC pin -- R_fixed -- GND.

    FSR conductance is roughly proportional to force. The default slope fits an
    Interlink FSR 402 at low forces; calibrate it by pressing on a kitchen scale.
    """
    vcc: float = 3.3
    r_fixed_ohm: float = 10_000.0
    newtons_per_microsiemens: float = 0.01

    def counts_to_newtons(self, counts: np.ndarray, adc_max: int) -> np.ndarray:
        v = np.clip(np.asarray(counts, float) / adc_max * self.vcc, 1e-6, self.vcc - 1e-6)
        r_fsr = self.r_fixed_ohm * (self.vcc - v) / v
        return np.maximum(1e6 / r_fsr * self.newtons_per_microsiemens, 0.0)


def parse_packet(line: str, cal: FsrCalibration = FsrCalibration()) -> TapEvent | None:
    """Parse one JSON line from firmware/tap_glove. Returns None for non-tap lines."""
    try:
        d = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict) or "piezo" not in d or "fsr" not in d:
        return None
    adc_max = (1 << int(d.get("adc_bits", 12))) - 1
    mid = adc_max / 2
    piezo = (np.asarray(d["piezo"], float) - mid) / mid
    force = cal.counts_to_newtons(np.asarray(d["fsr"], float), adc_max)
    mic = (np.asarray(d["mic"], float) - mid) / mid if d.get("mic") else None
    return TapEvent(fs=int(d["fs"]), piezo=piezo, force_n=force, mic=mic,
                    pretrigger=int(d.get("pre", 64)))
