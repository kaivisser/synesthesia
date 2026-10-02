from __future__ import annotations

import math
from dataclasses import replace

from synesthesia.types import BLACK, ColourState


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate from a to b by fraction t in [0, 1]."""
    return a + (b - a) * t


def _lerp_hue(a: float, b: float, t: float) -> float:
    """
    Interpolate hue the short way around the wheel (hue is circular, [0, 1)).
    """
    delta = (b - a + 0.5) % 1.0 - 0.5
    return (a + delta * t) % 1.0


class ColourInterpolator:
    """
    Smoothly interpolates between chords / colours.
    For saturation, value and intensity, we use linear interpolation.
    Since hue is circular, we want to head in the direction that gives us
    the smallest distnace between consecutive chords / colours.
    """

    def __init__(self, tau: float = 0.6) -> None:
        if tau <= 0.0:
            raise ValueError("tau must be positive")

        self.tau = float(tau)
        self.current = BLACK

    def update(self, target: ColourState, dt: float) -> ColourState:
        if dt <= 0.0:
            return replace(self.current, onsets=target.onsets)

        alpha = 1.0 - math.exp(-dt / self.tau)

        cur = self.current
        self.current = ColourState(
            hue=_lerp_hue(cur.hue, target.hue, alpha),
            saturation=_lerp(cur.saturation, target.saturation, alpha),
            value=_lerp(cur.value, target.value, alpha),
            intensity=_lerp(cur.intensity, target.intensity, alpha),
        )
        return replace(self.current, onsets=target.onsets)
