from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from opensimplex import OpenSimplex

from synesthesia.config import Config
from synesthesia.types import ColourState


class Motion(ABC):
    def advance(self, dt: float) -> None:
        """
        Advance the style's internal clock by dt seconds. Called once per
        frame before sampling.
        """

    @abstractmethod
    def sample(self, base: ColourState, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Compute the frame's HSV image given the base colour and coordinate grids.
        """
        pass


class Solid(Motion):
    """
    Single colour background
    """

    def sample(self, base: ColourState, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        hsv = np.empty((*xs.shape, 3), dtype=np.float32)
        hsv[..., 0] = base.hue
        hsv[..., 1] = base.saturation
        hsv[..., 2] = base.value
        return hsv


class Perlin(Motion):
    """
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
            + np.sin(
                np.sqrt((sx - self.scale * 0.5) ** 2 + (sy - self.scale * 0.5) ** 2) - t
            )
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
        curtains = np.sin(2.0 * np.pi * self.bands * xs + t) + 0.5 * np.sin(
            2.0 * np.pi * self.bands * 0.5 * xs - t * 1.4
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
    x: float
    y: float
    hue: float
    age: float = 0.0


class Rain(Motion):
    """
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
        for drop in self._drops:
            drop.age += dt
        self._drops = [d for d in self._drops if self.speed * d.age < self.max_radius]

    def sample(self, base: ColourState, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        self._spawn(base.onsets)

        hsv = np.empty((*xs.shape, 3), dtype=np.float32)
        hsv[..., 0] = base.hue
        hsv[..., 1] = base.saturation
        hsv[..., 2] = base.value

        for drop in self._drops:
            radius = self.speed * drop.age
            dist = np.sqrt((xs - drop.x) ** 2 + (ys - drop.y) ** 2)
            width = 0.12 * self.max_radius
            # shell = exp(-(d - r)² / 2σ²)
            shell = np.exp(-((dist - radius) ** 2) / (2.0 * width**2))
            fade = 1.0 - radius / self.max_radius  # 1 -> 0 over the drop's life
            weight = (shell * fade).astype(np.float32)

            # Tint hue toward the note (short way round the wheel) where the blob
            # is present, and lift brightness. Saturation is left to the chord.
            delta = (drop.hue - hsv[..., 0] + 0.5) % 1.0 - 0.5
            hsv[..., 0] = (hsv[..., 0] + delta * (weight * self.hue_pull)) % 1.0
            hsv[..., 2] = np.clip(hsv[..., 2] + weight * self.depth, 0.0, 1.0)

        return hsv

    def _spawn(self, onsets: tuple[tuple[int, float], ...]) -> None:
        for _pc, hue in onsets:
            self._drops.append(
                _Drop(
                    x=self._rng.random(),
                    y=self._rng.random(),
                    hue=float(hue),
                )
            )
        if len(self._drops) > self.max_drops:
            self._drops = self._drops[-self.max_drops :]


_MOTIONS: dict[str, type[Motion]] = {
    "solid": Solid,
    "plasma": Plasma,
    "perlin": Perlin,
    "aurora": Aurora,
    "rain": Rain,
}


def make_motion(name: str, config: Config | None = None) -> Motion:
    """
    Construct the motion style for the given name.
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
