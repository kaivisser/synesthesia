from __future__ import annotations

import time

from synesthesia.audio.base import AudioSource
from synesthesia.audio.framer import Framer
from synesthesia.audio.mic import MicSource
from synesthesia.colour.interpolator import ColourInterpolator
from synesthesia.colour.mapper import ColourMapper
from synesthesia.config import Config
from synesthesia.detect.detector import ChordDetector
from synesthesia.render.motion import make_motion
from synesthesia.render.terminal import TerminalRenderer


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

    if config.debug:
        _run_debug(config, framer, detector, mapper, interp)
    else:
        _run_render(config, framer, detector, mapper, interp)


def _run_render(config, framer, detector, mapper, interp) -> None:
    """Normal path: full-screen colour field via the terminal renderer."""
    renderer = TerminalRenderer(make_motion(config.motion))

    frame_dt = 1.0 / config.fps
    last = time.monotonic()

    with framer.source, renderer:
        try:
            for frame in framer.frames():
                estimate = detector.update(frame)

                now = time.monotonic()
                dt = now - last
                last = now
                colour = interp.update(mapper.map(estimate), dt)

                renderer.render(colour, dt)

                sleep = frame_dt - (time.monotonic() - now)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass


def _run_debug(config, framer, detector, mapper, interp) -> None:
    """
    Debug path: no full-screen colour field. Prints a single, updating status
    line with the live pipeline state plus a tiny colour swatch, so detection
    and audio can be diagnosed without any terminal rendering.
    """
    frame_dt = 1.0 / config.fps
    last = time.monotonic()
    fps_ema = config.fps

    with framer.source:
        try:
            for frame in framer.frames():
                estimate = detector.update(frame)

                now = time.monotonic()
                dt = now - last
                last = now
                colour = interp.update(mapper.map(estimate), dt)

                if dt > 0:
                    fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

                _print_debug_line(estimate, colour, fps_ema)

                sleep = frame_dt - (time.monotonic() - now)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass

    print()  # leave the cursor on a fresh line after the status line


def _print_debug_line(estimate, colour, fps: float) -> None:
    """Render one in-place status line: swatch + chord/conf/rms/rgb/fps."""
    r, g, b = colour.as_rgb()
    swatch = f"\x1b[48;2;{r};{g};{b}m   \x1b[0m"
    line = (
        f"{swatch} chord={estimate.name:<6} "
        f"conf={estimate.confidence:4.2f} "
        f"rms={estimate.rms:6.3f} "
        f"rgb=({r:3d},{g:3d},{b:3d}) "
        f"fps={fps:4.1f}"
    )
    print(f"\r\x1b[2K{line}", end="", flush=True)
