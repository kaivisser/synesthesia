from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from opensimplex import OpenSimplex

from synesthesia.types import ColourState


class Motion(ABC):
    """
    Abstract base for the motion styles selected by --motion

    The chord provides the **base colour**; motion only *modulates* it slightly
    over time and space so the field feels alive rather than flat. Every style
    shares this one cheap model and swaps behind the CLI flag.

    A motion is a pure function of *where* (normalised screen coordinates) and
    *when* (an accumulated clock). It samples the **whole grid at once**: given
    coordinate arrays it returns an HSV image, so all per-pixel work happens in
    NumPy rather than a Python loop (keeps ~30 fps affordable even for the
    space-varying styles like ``plasma``/``perlin``).

    Concrete styles (``solid``, ``breathe``, ``plasma``, ``perlin`` …) implement
    ``sample`` and, if they need to advance internal state, ``advance``.
    """

    def advance(self, dt: float) -> None:
        """
        Advance the style's internal clock by ``dt`` seconds. Called once per
        frame before sampling. Static styles (e.g. ``solid``) can ignore this;
        the default is a no-op so trivial styles need no boilerplate.
        """

    @abstractmethod
    def sample(self, base: ColourState, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Return the frame's HSV image given the base colour and coordinate grids.

        ``xs`` and ``ys`` are float arrays of identical shape ``(H, W)`` holding
        each pixel's normalised position in [0, 1). Returns a float array of
        shape ``(H, W, 3)`` with h, s, v channels in [0, 1] -- the colour of
        every pixel this frame. Implementations are fully vectorized (no Python
        per-pixel loop).
        """
        pass


class Solid(Motion):
    """
    Flat colour field: every pixel shows the base colour unchanged (PLAN.md §3.5,
    "solid — baseline/debug"). No time or space dependence, so it inherits the
    no-op ``advance`` and is the cheapest possible style — handy for verifying the
    render/colour pipeline in isolation before layering motion on top.
    """

    def sample(self, base: ColourState, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        hsv = np.empty((*xs.shape, 3), dtype=np.float32)
        hsv[..., 0] = base.hue
        hsv[..., 1] = base.saturation
        hsv[..., 2] = base.value
        return hsv


class Perlin(Motion):
    """
    Drifting simplex-noise field (PLAN.md §3.5, the **default** motion).

    A smooth 3-D simplex field (2 space axes + 1 time axis) slowly drifts through
    the screen, giving an organic, cloud-like shimmer. Per §3.5 the noise only
    *modulates* the chord's base colour -- it nudges **brightness** up/down and
    **hue** very slightly, so the chord still clearly owns the colour while the
    field breathes. Uses the pure-Python ``opensimplex`` (§4) via its vectorized
    ``noise3array``, so a whole frame is one C-level call rather than a loop.

    Parameters
    ----------
    scale:
        Spatial frequency of the noise: higher = finer, busier texture; lower =
        broad, slow blobs. Applied to the normalised [0, 1) coordinates.
    drift_speed:
        How fast the time axis advances (screen-units/second); larger = quicker
        churn.
    value_depth:
        Max brightness swing the noise adds/removes around ``base.value`` (in
        [0, 1] value units). Small, so the chord stays dominant.
    hue_depth:
        Max hue shift the noise adds around ``base.hue`` (in [0, 1) hue units).
        Kept tiny so the colour doesn't wander off the chord's hue.
    seed:
        Simplex seed; fixed by default for reproducible visuals.
    """

    def __init__(
        self,
        scale: float = 3.0,
        drift_speed: float = 0.15,
        value_depth: float = 0.25,
        hue_depth: float = 0.03,
        seed: int = 0,
    ) -> None:
        self.scale = float(scale)
        self.drift_speed = float(drift_speed)
        self.value_depth = float(value_depth)
        self.hue_depth = float(hue_depth)
        self._noise = OpenSimplex(seed=seed)
        self._clock = 0.0

    def advance(self, dt: float) -> None:
        self._clock += dt

    def sample(self, base: ColourState, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        # noise3array wants 1-D axes and returns (len(z), len(y), len(x)); our
        # xs/ys come from meshgrid, so recover the axes from row0 / col0.
        x_axis = np.ascontiguousarray(xs[0, :], dtype=np.float64) * self.scale
        y_axis = np.ascontiguousarray(ys[:, 0], dtype=np.float64) * self.scale
        z_axis = np.array([self._clock * self.drift_speed], dtype=np.float64)

        # (1, H, W) -> (H, W); simplex output is ~[-1, 1], remap to [-0.5, 0.5].
        field = self._noise.noise3array(x_axis, y_axis, z_axis)[0]
        field = np.clip(field, -1.0, 1.0) * 0.5

        hsv = np.empty((*xs.shape, 3), dtype=np.float32)
        hsv[..., 0] = (base.hue + field * (2.0 * self.hue_depth)) % 1.0
        hsv[..., 1] = base.saturation
        hsv[..., 2] = np.clip(base.value + field * (2.0 * self.value_depth), 0.0, 1.0)
        return hsv


# Name -> style registry. The `--motion` flag (config.MOTIONS) is validated at
# the CLI, but this is the single source of truth for which names are actually
# *implemented*; the remaining §3.5 styles slot in here as they're built.
_MOTIONS: dict[str, type[Motion]] = {
    "solid": Solid,
    "perlin": Perlin,
}


def make_motion(name: str) -> Motion:
    """
    Construct the motion style registered under ``name``.

    Raises ``ValueError`` (listing what *is* implemented) if the name is unknown
    -- this covers valid ``--motion`` choices that simply aren't built yet, so
    the failure is explicit rather than a silent fall-through to ``solid``.
    """
    try:
        cls = _MOTIONS[name]
    except KeyError:
        valid = ", ".join(sorted(_MOTIONS))
        raise ValueError(
            f"motion style {name!r} not implemented; available: {valid}"
        ) from None
    return cls()
