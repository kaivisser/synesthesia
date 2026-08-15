from __future__ import annotations

import time

from synesthesia.audio.base import AudioSource
from synesthesia.audio.framer import Framer
from synesthesia.audio.mic import MicSource
from synesthesia.colour.interpolator import ColourInterpolator
from synesthesia.colour.mapper import ColourMapper
from synesthesia.config import Config
from synesthesia.detect.detector import ChordDetector


def build_source(config: Config) -> AudioSource:
    if config.source == "mic":
        return MicSource(device=config.device, gain=config.gain)

    if config.source in ("file", "monitor"):
        raise NotImplementedError(f"source {config.source!r} not implemented yet")

    raise ValueError(f"unknown source {config.source!r}")


def run(config: Config) -> None:
    framer = Framer(
        build_source(config),
        window_size=config.window_size,
        hop=config.hop,
    )
    detector = ChordDetector(
        sample_rate=config.sample_rate,
        silence_rms=config.silence_rms,
        smoothing=config.smoothing,
    )
    mapper = ColourMapper(scheme=config.scheme)
    interp = ColourInterpolator(tau=config.fade_tau)

    frame_dt = 1.0 / config.fps
    last = time.monotonic()

    with framer.source:
        try:
            for frame in framer.frames():
                estimate = detector.update(frame)
                target = mapper.map(estimate)

                now = time.monotonic()
                dt = now - last
                last = now
                colour = interp.update(target, dt)

                _print_swatch(estimate.name, colour)

                # Pace the render side toward the target fps.
                sleep = frame_dt - (time.monotonic() - now)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass

    print()  # leave the cursor on a fresh line after the swatch


def _print_swatch(name: str, colour) -> None:
    """Temporary stand-in for the renderer: chord label + truecolor block."""
    r, g, b = colour.as_rgb()
    block = f"\x1b[48;2;{r};{g};{b}m      \x1b[0m"
    print(f"\r{name:<8} {block} rgb({r:3d},{g:3d},{b:3d})", end="", flush=True)
