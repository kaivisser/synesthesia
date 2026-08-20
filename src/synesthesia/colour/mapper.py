from __future__ import annotations

from synesthesia.colour.schemes import SCHEMES
from synesthesia.types import BLACK, ChordEstimate, ColourState, Quality


def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


_QUALITY_BIAS: dict[Quality, tuple[float, float]] = {
    Quality.MAJ: (0.95, 1.0),  # (saturation multiplier, brightness multiplier)
    Quality.MIN: (0.80, 0.75),
}


class ColourMapper:
    """
    Root -> hue: Chords root value deterimes background colour.
    Quality -> saturation/value:
      major chords brighter and more saturated,
      minor chords darker and less saturated.
    Loudness (RMS) -> value/intensity:
      louder chords are brighter and drive faster motion (if motion != solid)
    Confidence -> purity: Weak mapper matches result in desaturated background colour.
    Silence (NO_CHORD) -> black
    """

    def __init__(self, scheme: str = "scriabin") -> None:
        if scheme not in SCHEMES:
            valid = ", ".join(sorted(SCHEMES))
            raise ValueError(f"unknown scheme {scheme!r}; choose one of: {valid}")

        self.scheme_name = scheme
        self.hues = SCHEMES[scheme]

    def map(self, estimate: ChordEstimate) -> ColourState:
        """
        Maps a ChordEstimate to a ColourState
        """
        if not estimate.is_chord:
            return BLACK

        hue = self.hues[estimate.root]
        base_sat, base_val = _QUALITY_BIAS[estimate.quality]

        confidence = _clamp(estimate.confidence)
        purity = 0.4 + 0.6 * confidence
        saturation = _clamp(base_sat * purity)

        loudness = _clamp(estimate.rms * 4.0)
        value = _clamp(base_val * (0.25 + 0.75 * loudness))
        intensity = loudness

        onsets = tuple((pc, self.hues[pc]) for pc in estimate.onsets)

        return ColourState(
            hue=hue,
            saturation=saturation,
            value=value,
            intensity=intensity,
            onsets=onsets,
        )
