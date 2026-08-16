from __future__ import annotations

from typing import Optional

import numpy as np

from synesthesia.detect.chroma import chroma as compute_chroma
from synesthesia.detect.templates import build_templates, match
from synesthesia.types import NO_CHORD, ChordEstimate

ANALYSIS_SR = 44100


class ChordDetector:
    """
    Uses an exponential moving average over each chrome to discount transient chords.

    For example;

    raw per-frame chord
    frame:   1    2    3    4    5    6    7    8
    chord:   C    C    C    Am   C    C    F    C

    We don't want to render frames 4 & 7 as they are one-frame flickers,
    instead we want to simulate a smooth transition between chords over a sustained period.
    """

    def __init__(
        self,
        sample_rate: int = ANALYSIS_SR,
        silence_rms: float = 0.01,
        smoothing: float = 0.6,
    ) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0.0, 1.0)")

        self.sample_rate = sample_rate
        self.silence_rms = float(silence_rms)
        self.smoothing = float(smoothing)

        self._templates, self._labels = build_templates()

        self._ema: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> ChordEstimate:
        """
        Process one windowed frame and return a smoothed ChordEstimate.

        Call once per frame; the EMA state carries between calls. Silence
        (RMS < threshold) short-circuits to NO_CHORD and decays the smoothed
        chroma toward zero, so sound resuming after a pause fades back in rather
        than snapping to the pre-silence chord.
        """
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))

        if rms < self.silence_rms:
            if self._ema is not None:
                self._ema *= self.smoothing
            return NO_CHORD

        chroma = compute_chroma(frame, self.sample_rate)

        if self._ema is None:
            self._ema = chroma
        else:
            self._ema = self.smoothing * self._ema + (1.0 - self.smoothing) * chroma

        root, quality, confidence = match(self._ema, self._templates, self._labels)
        return ChordEstimate(
            root=root,
            quality=quality,
            confidence=confidence,
            rms=rms,
            chroma=self._ema.copy(),
        )
