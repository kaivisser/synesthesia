from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class Quality(str, Enum):
    MAJ = "maj"
    MIN = "min"


@dataclass(frozen=True)
class ChordEstimate:
    """
    root:
        Pitch class 0-11 (0=C .. 11=B). Meaningless when is_chord is False.
    quality:
        MAJ or MIN etc. Meaningless when is_chord is False.
    confidence:
        Cosine-similarity match strength in [0, 1].
    rms:
        Root-mean-square loudness of the source frame.
    is_chord:
        False only for the silence sentinel.
    chroma:
        The 12-element pitch-class vector (C..B) that produced this estimate,
        after smoothing. ``None`` for the silence sentinel.
    onsets:
        Pitch classes (0-11) whose energy *newly* rose this frame -- i.e. note
        attacks detected since the previous frame. Instantaneous events (not
        smoothed), used to trigger per-note visual effects (e.g. rain drops).
        Empty when nothing new struck.
    """

    root: int
    quality: Quality
    confidence: float
    rms: float
    is_chord: bool = True
    chroma: Optional[np.ndarray] = None
    onsets: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        """Human-readable label, e.g. 'C maj', or '-' for silence."""
        if not self.is_chord:
            return "-"
        return f"{_NOTE_NAMES[self.root]} {self.quality.value}"


NO_CHORD = ChordEstimate(
    root=0, quality=Quality.MAJ, confidence=0.0, rms=0.0, is_chord=False
)


@dataclass(frozen=True)
class ColourState:
    """
    hue:
        Dominant hue in [0, 1) (0=red, 1/3=green, 2/3=blue), from root + scheme.
    saturation:
        Colour purity in [0, 1]; driven by quality/confidence.
    value:
        Brightness in [0, 1]; driven mainly by loudness (RMS).
    intensity:
        Overall energy in [0, 1] the renderer can use to scale motion amplitude
        (e.g. stronger/faster movement when the music is louder).
    onsets:
        Note attacks for *this frame* as ``(pitch_class, hue)`` pairs -- the
        pitch classes that newly struck (from ``ChordEstimate.onsets``) paired
        with each note's hue under the active colour scheme. Motion styles use
        these to spawn per-note effects (e.g. a rain drop tinted by the note).
        Instantaneous: empty on frames with no new attack, and never smoothed by
        the interpolator (unlike hue/sat/value/intensity).
    """

    hue: float
    saturation: float
    value: float
    intensity: float
    onsets: tuple[tuple[int, float], ...] = ()

    def as_rgb(self) -> tuple[int, int, int]:
        """Convert to an 8-bit RGB triple for renderers that need raw colour."""
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(self.hue, self.saturation, self.value)
        return round(r * 255), round(g * 255), round(b * 255)


BLACK = ColourState(hue=0.01, saturation=0.01, value=0.01, intensity=0.01)
