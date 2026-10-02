from __future__ import annotations

from typing import Optional

import numpy as np

from synesthesia.audio.base import AudioSource


class Framer:
    """
    Sliding Hann-windowed framer over an AudioSource
    """

    def __init__(self, source: AudioSource, window_size: int, hop: int) -> None:
        if hop <= 0 or window_size <= 0:
            raise ValueError("window_size and hop must be positive")
        if hop > window_size:
            raise ValueError("hop must not exceed window_size")

        self.source = source
        self.window_size = window_size
        self.hop = hop

        # Persistent window; older samples on the left, newest on the right.
        self._window = np.zeros(window_size, dtype=np.float32)
        self._hann = np.hanning(window_size).astype(np.float32)

        # Number of hops needed before the window is first fully populated.
        self._priming = window_size // hop
        self._exhausted = False

    def read(self) -> Optional[np.ndarray]:
        """
        Advance by one hop and return the latest Hann-windowed frame.

        """
        if self._exhausted:
            return None

        needed = self._priming if self._priming > 0 else 1
        for _ in range(needed):
            chunk = self.source.read(self.hop)
            if chunk is None:
                self._exhausted = True
                return None
            self._push(chunk)
        self._priming = 0

        return self._window * self._hann

    def _push(self, chunk: np.ndarray) -> None:
        self._window[: -self.hop] = self._window[self.hop :]
        self._window[-self.hop :] = chunk

    def frames(self):
        """Iterate windowed frames until the source is exhausted."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
