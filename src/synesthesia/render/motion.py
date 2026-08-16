from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

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


class Plasma(Motion):
    """
    Sine-interference plasma (PLAN.md §3.5, "plasma — smooth, hypnotic, cheap").

    Classic demoscene plasma: a few sine waves at different angles/frequencies
    are summed into a scalar field that drifts over time. The field only
    *modulates* the chord's base colour -- it swings **brightness** and nudges
    **hue** slightly, so the field roils hypnotically while the chord still owns
    the colour (§3.5). Fully vectorized over the coordinate grid.

    Parameters
    ----------
    scale:
        Spatial frequency of the sine waves; higher = finer, busier ripples.
    speed:
        Drift speed of the animation (radians/second of phase).
    value_depth:
        Max brightness swing added around ``base.value`` (in [0, 1] units).
    hue_depth:
        Max hue shift added around ``base.hue`` (in [0, 1) hue units); tiny.
    """

    def __init__(
        self,
        scale: float = 6.0,
        speed: float = 1.2,
        value_depth: float = 0.3,
        hue_depth: float = 0.04,
    ) -> None:
        self.scale = float(scale)
        self.speed = float(speed)
        self.value_depth = float(value_depth)
        self.hue_depth = float(hue_depth)
        self._clock = 0.0

    def advance(self, dt: float) -> None:
        self._clock += dt

    def sample(self, base: ColourState, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        t = self._clock * self.speed
        sx = xs * self.scale
        sy = ys * self.scale

        # Sum of a few sines at different orientations -> interference pattern.
        field = (
            np.sin(sx + t)
            + np.sin(sy - t * 0.8)
            + np.sin((sx + sy) * 0.5 + t * 1.3)
            + np.sin(np.sqrt((sx - self.scale * 0.5) ** 2
                             + (sy - self.scale * 0.5) ** 2) - t)
        )
        # 4 sines -> [-4, 4]; normalise to [-0.5, 0.5].
        field = field * (0.5 / 4.0)

        hsv = np.empty((*xs.shape, 3), dtype=np.float32)
        hsv[..., 0] = (base.hue + field * (2.0 * self.hue_depth)) % 1.0
        hsv[..., 1] = base.saturation
        hsv[..., 2] = np.clip(base.value + field * (2.0 * self.value_depth), 0.0, 1.0)
        return hsv


class Aurora(Motion):
    """
    Undulating vertical bands (PLAN.md §3.5, "aurora -- northern-lights; pairs
    well with half-blocks").

    Soft vertical curtains of light shimmer and sway across the screen, brightest
    partway up and drifting sideways over time -- an approximation of the aurora.
    Bands come from summed sines in x (swaying with time); a vertical envelope
    concentrates the glow into curtains. Only **brightness** is modulated.

    Parameters
    ----------
    bands:
        Approximate number of vertical curtains across the width.
    speed:
        Sideways drift/sway speed (radians/second of phase).
    depth:
        Peak brightness added at a curtain's crest (in [0, 1] value units).
    """

    def __init__(
        self,
        bands: float = 3.0,
        speed: float = 0.6,
        depth: float = 0.45,
    ) -> None:
        self.bands = float(bands)
        self.speed = float(speed)
        self.depth = float(depth)
        self._clock = 0.0

    def advance(self, dt: float) -> None:
        self._clock += dt

    def sample(self, base: ColourState, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        t = self._clock * self.speed
        # Two swaying sine trains in x -> shifting curtains in [0, 1].
        curtains = (
            np.sin(2.0 * np.pi * self.bands * xs + t)
            + 0.5 * np.sin(2.0 * np.pi * self.bands * 0.5 * xs - t * 1.4)
        )
        curtains = (curtains + 1.5) / 3.0  # -> ~[0, 1]
        # Vertical envelope: brightest partway up, dimmer at top/bottom edges.
        envelope = np.sin(np.pi * ys)
        field = curtains * envelope

        hsv = np.empty((*xs.shape, 3), dtype=np.float32)
        hsv[..., 0] = base.hue
        hsv[..., 1] = base.saturation
        hsv[..., 2] = np.clip(base.value + field * self.depth, 0.0, 1.0)
        return hsv


@dataclass
class _Drop:
    """
    One live rain drop: a note-tinted blob spreading from where it landed.

    x, y:
        Normalised screen position in [0, 1) where the drop landed (random per
        drop, so notes rain across the whole field).
    hue:
        The striking note's hue (from the colour scheme) -- the drop's tint,
        distinct from the chord's base colour.
    age:
        Seconds since the drop landed; drives its expanding radius and fade.
    """

    x: float
    y: float
    hue: float
    age: float = 0.0


class Rain(Motion):
    """
    Note-triggered rain drops over the chord field (raindrop synesthesia).

    The chord still owns the **base colour** of the whole field; on top of it,
    every detected **note onset** (``base.onsets`` -- see ColourState) drops a
    soft blob at a random position, tinted by *that note's* hue and expanding +
    fading like a raindrop hitting water. A busy passage sprinkles many drops;
    silence lets the field settle back to the plain chord colour.

    Unlike the other styles this one is **event-driven**: it reads the per-frame
    ``onsets`` the detector/mapper produced, rather than deriving motion purely
    from a clock. Drops are composited over the base by pulling hue toward the
    note and lifting brightness where a blob is present; all blob math is
    vectorized over the coordinate grid.

    Parameters
    ----------
    speed:
        How fast a drop's radius grows (screen-units/second).
    max_radius:
        Radius at which a drop has fully spread and is culled (screen-units).
    depth:
        Peak brightness a fresh drop adds at its centre (in [0, 1] value units).
    hue_pull:
        How strongly a drop tints toward the note's hue at its centre, in [0, 1]
        (0 = brightness only; 1 = fully the note's hue at the core).
    max_drops:
        Cap on simultaneously live drops (oldest culled first) to bound cost.
    seed:
        Optional RNG seed for reproducible drop positions.
    """

    def __init__(
        self,
        speed: float = 0.4,
        max_radius: float = 0.5,
        depth: float = 0.6,
        hue_pull: float = 0.6,
        max_drops: int = 24,
        seed: int | None = None,
    ) -> None:
        self.speed = float(speed)
        self.max_radius = float(max_radius)
        self.depth = float(depth)
        self.hue_pull = float(hue_pull)
        self.max_drops = int(max_drops)
        self._rng = random.Random(seed)
        self._drops: list[_Drop] = []

    def advance(self, dt: float) -> None:
        """
        Age every live drop by ``dt`` and cull ones that have fully spread.

        Spawning happens in ``sample`` (which is where ``base.onsets`` is
        available); ``advance`` only moves time forward and drops any blob whose
        radius has passed ``max_radius`` so the list stays bounded.
        """
        for drop in self._drops:
            drop.age += dt
        self._drops = [
            d for d in self._drops if self.speed * d.age < self.max_radius
        ]

    def sample(self, base: ColourState, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        self._spawn(base.onsets)

        hsv = np.empty((*xs.shape, 3), dtype=np.float32)
        hsv[..., 0] = base.hue
        hsv[..., 1] = base.saturation
        hsv[..., 2] = base.value

        for drop in self._drops:
            radius = self.speed * drop.age
            # Expanding soft ring: a Gaussian shell at the drop's current radius,
            # fading as it grows so older drops gently disappear.
            dist = np.sqrt((xs - drop.x) ** 2 + (ys - drop.y) ** 2)
            width = 0.12 * self.max_radius
            shell = np.exp(-((dist - radius) ** 2) / (2.0 * width ** 2))
            fade = 1.0 - radius / self.max_radius  # 1 -> 0 over the drop's life
            weight = (shell * fade).astype(np.float32)

            # Tint hue toward the note (short way round the wheel) where the blob
            # is present, and lift brightness. Saturation is left to the chord.
            delta = (drop.hue - hsv[..., 0] + 0.5) % 1.0 - 0.5
            hsv[..., 0] = (hsv[..., 0] + delta * (weight * self.hue_pull)) % 1.0
            hsv[..., 2] = np.clip(hsv[..., 2] + weight * self.depth, 0.0, 1.0)

        return hsv

    def _spawn(self, onsets: tuple[tuple[int, float], ...]) -> None:
        """
        Drop a fresh blob for each note onset this frame at a random position.

        Each onset is a ``(pitch_class, hue)`` pair (from the mapper); we keep
        only the hue for the tint and scatter the drop uniformly across the
        field. Enforces ``max_drops`` by discarding the oldest when over budget.
        """
        for _pc, hue in onsets:
            self._drops.append(
                _Drop(
                    x=self._rng.random(),
                    y=self._rng.random(),
                    hue=float(hue),
                )
            )
        if len(self._drops) > self.max_drops:
            self._drops = self._drops[-self.max_drops:]


# Name -> style registry. The `--motion` flag (config.MOTIONS) is validated at
# the CLI, but this is the single source of truth for which names are actually
# *implemented*; the remaining §3.5 styles slot in here as they're built.
_MOTIONS: dict[str, type[Motion]] = {
    "solid": Solid,
    "plasma": Plasma,
    "perlin": Perlin,
    "aurora": Aurora,
    "rain": Rain,
}


def make_motion(name: str, config: object | None = None) -> Motion:
    """
    Construct the motion style registered under ``name``.

    ``config`` is the app ``Config`` (optional); when the selected style exposes
    tunables (currently only ``rain``) they're read from it, so ``--rain-*``
    flags reach the style. Other styles ignore it and use their defaults.

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

    if cls is Rain and config is not None:
        return Rain(
            speed=config.rain_speed,
            max_radius=config.rain_max_radius,
            depth=config.rain_depth,
            hue_pull=config.rain_hue_pull,
            max_drops=config.rain_max_drops,
        )
    return cls()
