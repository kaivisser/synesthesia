from __future__ import annotations

import time

from synesthesia.audio.base import AudioSource
from synesthesia.audio.framer import Framer
from synesthesia.audio.mic import MicSource
from synesthesia.audio.monitor import MonitorSource
from synesthesia.colour.interpolator import ColourInterpolator
from synesthesia.colour.mapper import ColourMapper
from synesthesia.config import Config
from synesthesia.detect.detector import ChordDetector
from synesthesia.render.motion import make_motion
from synesthesia.render.terminal import TerminalRenderer


def build_source(config: Config) -> AudioSource:
    if config.source == "mic":
        return MicSource(device=config.device, gain=config.gain)

    if config.source == "monitor":
        return MonitorSource(sink=config.monitor_sink, gain=config.gain)

    if config.source == "file":
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
        onset_rise=config.onset_rise,
        onset_floor=config.onset_floor,
        onset_cooldown_frames=config.onset_cooldown_frames,
        onset_min_rms=config.onset_min_rms,
    )
    mapper = ColourMapper(scheme=config.scheme)
    interp = ColourInterpolator(tau=config.fade_tau)

    if config.debug:
        _run_debug(config, framer, detector, mapper, interp)
    else:
        _run_render(config, framer, detector, mapper, interp)


def _run_render(config, framer, detector, mapper, interp) -> None:
    renderer = TerminalRenderer(make_motion(config.motion, config))

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

    print()


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Peak-normalised chroma level a note must exceed to be shown as "firing" in the
# debug readout. Matches the spirit of the detector's onset gate.
_FIRING_THRESHOLD = 0.5


def _format_chroma(chroma) -> str:
    """
    Show only the notes currently "firing" (peak-normalised level above
    ``_FIRING_THRESHOLD``), strongest first, instead of all 12 pitch classes.
    """
    if chroma is None:
        return "notes[ - ]"
    peak = chroma.max()
    if peak <= 0.0:
        return "notes[ - ]"

    norm = chroma / peak
    firing = [
        (_NOTE_NAMES[i], float(norm[i]))
        for i in range(len(norm))
        if norm[i] >= _FIRING_THRESHOLD
    ]
    firing.sort(key=lambda nv: nv[1], reverse=True)
    if not firing:
        return "notes[ - ]"
    cells = " ".join(f"{name}={val:4.2f}" for name, val in firing)
    return f"notes[ {cells} ]"


def _print_debug_line(estimate, colour, fps: float) -> None:
    r, g, b = colour.as_rgb()
    swatch = f"\x1b[48;2;{r};{g};{b}m   \x1b[0m"
    line = (
        f"{swatch} chord={estimate.name:<6} "
        f"conf={estimate.confidence:4.2f} "
        f"rms={estimate.rms:6.3f} "
        f"rgb=({r:3d},{g:3d},{b:3d}) "
        f"fps={fps:4.1f} "
        f"{_format_chroma(estimate.chroma)}"
    )
    print(f"\r\x1b[2K{line}", end="", flush=True)
