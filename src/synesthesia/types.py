from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Pitch-class names, index 0-11 = C..B (shared spelling across the project).
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class Quality(str, Enum):
    """
    Chord quality (the "flavour" of a triad), independent of its root.

    v1 vocabulary is major/minor only (PLAN.md §3.3); dim/aug/7ths are planned
    later and would slot in as new members here. Inheriting from ``str`` makes
    members compare/serialise as their value ("maj"/"min") -- handy for logging
    and for interop with the matcher, which already speaks those strings.
    """

    MAJ = "maj"
    MIN = "min"


@dataclass(frozen=True)
class ChordEstimate:
    """
    One detector output: the chord heard in a frame, with dynamics (PLAN.md §2).

    An immutable value object passed down the pipeline (detector -> colour
    mapper). ``root``/``quality`` say *which* chord; ``confidence`` and ``rms``
    carry the dynamics the colour mapper needs (confidence -> purity, rms ->
    brightness, per §3.4).

    Silence is represented by the ``NO_CHORD`` sentinel below rather than a magic
    root value: check ``is_chord`` to decide chord-colour vs fade-to-black.

    Attributes
    ----------
    root:
        Pitch class 0-11 (0=C .. 11=B). Meaningless when ``is_chord`` is False.
    quality:
        MAJ or MIN (v1). Meaningless when ``is_chord`` is False.
    confidence:
        Cosine-similarity match strength in [0, 1] (§3.3).
    rms:
        Root-mean-square loudness of the source frame (roughly [0, 1]).
    is_chord:
        False only for the silence sentinel; True for a real detected chord.
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


# Silence sentinel: root/quality are placeholders (not read when is_chord=False).
NO_CHORD = ChordEstimate(
    root=0, quality=Quality.MAJ, confidence=0.0, rms=0.0, is_chord=False
)
