"""Where taps come from: the real glove over USB serial, or the simulator."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .materials import Material
from .simulate import simulate_tap
from .tap import FsrCalibration, TapEvent, parse_packet


class SerialGlove:
    """Reads JSON tap packets sent by firmware/tap_glove/tap_glove.ino."""

    def __init__(self, port: str, baud: int = 921_600, cal: FsrCalibration = FsrCalibration()):
        try:
            import serial  # pyserial
        except ImportError as e:
            raise SystemExit("Reading the glove needs pyserial: pip install pyserial") from e
        try:
            self._serial = serial.Serial(port, baud, timeout=1.0)
        except Exception as e:
            raise SystemExit(f"Could not open serial port {port!r}: {e}") from e
        self._cal = cal

    def taps(self, n: int) -> Iterator[TapEvent]:
        got = 0
        while got < n:
            try:
                raw = self._serial.readline()
            except Exception as e:
                raise SystemExit(f"Serial read failed: {e}") from e
            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                continue
            tap = parse_packet(line, self._cal)
            if tap is None:  # status messages from the firmware
                print(f"  [glove] {line[:80]}")
                continue
            got += 1
            yield tap

    def close(self) -> None:
        self._serial.close()


class SimulatedGlove:
    def __init__(self, material: str | Material, *, seed: int | None = None,
                 with_mic: bool = True, finger: str = "nail"):
        self.material, self.with_mic, self.finger = material, with_mic, finger
        self._rng = np.random.default_rng(seed)

    def taps(self, n: int) -> Iterator[TapEvent]:
        for _ in range(n):
            yield simulate_tap(self.material, self._rng, with_mic=self.with_mic, finger=self.finger)

    def close(self) -> None:
        pass
