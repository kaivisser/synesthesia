from __future__ import annotations

from synesthesia.colour.schemes import SCHEMES
from synesthesia.types import BLACK, ChordEstimate, ColourState, Quality


def _clamp01(x: float) -> float:
    """Clamp a float into [0.0, 1.0]."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


_QUALITY_BIAS: dict[Quality, tuple[float, float]] = {
    Quality.MAJ: (0.95, 1.0),  # bright & saturated
    Quality.MIN: (0.80, 0.75),  # darker & softer
}


class ColourMapper:
    """
    Turn a ChordEstimate into a target ColourState (PLAN.md §3.4 -- the
    "synesthesia core").

    This is a *pure, stateless* mapping: chord in -> colour out, one call per
    frame. It does **not** do interpolation or fade-to-black over time -- that
    smoothing lives separately in ``interp.py`` so the two concerns stay
    swappable (§4b). The mapper only answers "what colour *should* this chord
    be, right now?"; the interpolator decides how fast we move toward it.

    The mapping follows the §3.4 rules, all expressed on the HSV axes of
    ``ColourState``:

    * **Root -> hue** via the selected scheme table (default Scriabin), so the
      dominant colour is chosen by the chord's root pitch class.
    * **Quality -> saturation/value**: major reads brighter/more saturated,
      minor darker -- a small, fixed adjustment per quality.
    * **Loudness (RMS) -> value/intensity**: louder chords glow brighter and
      drive stronger motion.
    * **Confidence -> purity**: a shaky match desaturates toward muddy grey so
      uncertain detections don't produce a confidently "wrong" vivid colour.
    * **Silence (NO_CHORD) -> black**: the sentinel maps straight to ``BLACK``.

    Attributes
    ----------
    hues:
        The active 12-entry hue table (root -> hue in [0, 1)), taken from
        ``schemes.SCHEMES`` by name at construction.
    scheme_name:
        Name of the active scheme (for logging / display).
    """

    def __init__(self, scheme: str = "scriabin") -> None:
        if scheme not in SCHEMES:
            valid = ", ".join(sorted(SCHEMES))
            raise ValueError(f"unknown scheme {scheme!r}; choose one of: {valid}")

        self.scheme_name = scheme
        self.hues = SCHEMES[scheme]

    def map(self, estimate: ChordEstimate) -> ColourState:
        """
        Map one ChordEstimate to its target ColourState (§3.4 rules).

        Pure and stateless: same estimate always yields the same colour. Silence
        short-circuits to BLACK; otherwise HSV is built from root (hue), quality
        (sat/value bias), loudness (value/intensity) and confidence (purity).
        """
        if not estimate.is_chord:
            return BLACK

        hue = self.hues[estimate.root]
        base_sat, base_val = _QUALITY_BIAS[estimate.quality]

        confidence = _clamp01(estimate.confidence)
        purity = 0.4 + 0.6 * confidence
        saturation = _clamp01(base_sat * purity)

        # Loudness -> brightness/intensity. RMS is roughly [0, 1] but usually
        # small; scale it up and clamp so normal playing reaches full brightness.
        loudness = _clamp01(estimate.rms * 4.0)
        value = _clamp01(base_val * (0.25 + 0.75 * loudness))
        intensity = loudness

        return ColourState(
            hue=hue,
            saturation=saturation,
            value=value,
            intensity=intensity,
        )
