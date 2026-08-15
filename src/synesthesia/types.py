from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
    """

    root: int
    quality: Quality
    confidence: float
    rms: float
    is_chord: bool = True

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
    """

    hue: float
    saturation: float
    value: float
    intensity: float

    def as_rgb(self) -> tuple[int, int, int]:
        """Convert to an 8-bit RGB triple for renderers that need raw colour."""
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(self.hue, self.saturation, self.value)
        return round(r * 255), round(g * 255), round(b * 255)


# Fully black state: what silence/fade-to-black settles on (§3.4).
BLACK = ColourState(hue=0.0, saturation=0.0, value=0.0, intensity=0.0)
