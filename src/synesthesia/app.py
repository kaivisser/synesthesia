from __future__ import annotations

import threading
import time
from dataclasses import replace

from synesthesia.app_state import LatestTarget, OnsetQueue
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


def _analysis_worker(
    framer: Framer,
    detector: ChordDetector,
    mapper: ColourMapper,
    latest: LatestTarget,
    onsets: OnsetQueue,
    finished: threading.Event,
) -> None:
    """
    Runs AudioSource, Framer, Mapper and places output in to shared thread-safe LatestTarget / OnsetQueue  buffers
    """
    try:
        for frame in framer.frames():
            if finished.is_set():
                break
            estimate = detector.update(frame)
            target = mapper.map(estimate)
            onsets.push(target.onsets)
            latest.publish(replace(target, onsets=()), estimate)
    finally:
        finished.set()


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

    latest = LatestTarget()
    onsets = OnsetQueue()
    finished = threading.Event()

    analysis = threading.Thread(
        target=_analysis_worker,
        args=(framer, detector, mapper, latest, onsets, finished),
        name="analysis",
        daemon=True,
    )

    consumer = _run_debug if config.debug else _run_render

    framer.source.start()
    analysis.start()
    try:
        consumer(config, interp, latest, onsets, finished)
    finally:
        finished.set()
        framer.source.stop()
        analysis.join(timeout=2.0)


def _run_render(config, interp, latest, onsets, finished) -> None:
    """
    Render to terminal. Run on dedicated thread.
    """
    renderer = TerminalRenderer(make_motion(config.motion, config))

    frame_dt = 1.0 / config.fps
    last = time.monotonic()

    with renderer:
        try:
            while not finished.is_set():
                now = time.monotonic()
                dt = now - last
                last = now

                colour = interp.update(latest.get(), dt)
                colour = replace(colour, onsets=onsets.drain())
                renderer.render(colour, dt)

                sleep = frame_dt - (time.monotonic() - now)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass


def _run_debug(config, interp, latest, onsets, finished) -> None:
    """
    Render debug info to terminal.
    """

    frame_dt = 1.0 / config.fps
    last = time.monotonic()
    fps_ema = config.fps
    first_debug = True

    try:
        while not finished.is_set():
            now = time.monotonic()
            dt = now - last
            last = now

            target, estimate = latest.get_with_estimate()
            colour = interp.update(target, dt)
            colour = replace(colour, onsets=onsets.drain())

            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

            _print_debug_line(estimate, colour, fps_ema, first_debug)
            first_debug = False

            sleep = frame_dt - (time.monotonic() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    print("\x1b[?7h")


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _format_chroma(chroma) -> str:
    """
    Show all 12 pitch classes in fixed C..B order, peak-normalised, each
    defaulting to 0.00 when there is no energy (silence / no chroma).
    """
    if chroma is None:
        cells = " ".join(f"{name}=0.00" for name in _NOTE_NAMES)
        return f"notes[ {cells} ]"

    peak = float(chroma.max())
    norm = chroma / peak if peak > 0.0 else chroma
    cells = " ".join(
        f"{name}={float(norm[i]):4.2f}" for i, name in enumerate(_NOTE_NAMES)
    )
    return f"notes[ {cells} ]"


_DEBUG_LINES = 6

_first_debug_print = True


def _print_debug_line(estimate, colour, fps: float, first_debug: bool) -> None:
    if first_debug:
        time.sleep(2)
    r, g, b = colour.as_rgb()
    swatch = f"\x1b[48;2;{r};{g};{b}m   \x1b[0m"

    # Move cursor back up to overwrite previous block (skip on first call)
    prefix = "" if first_debug else f"\x1b[{_DEBUG_LINES}A"

    block = (
        f"{prefix}"
        f"\x1b[2K{swatch} chord={estimate.name}\n"
        f"\x1b[2K  confidence     = {estimate.confidence:4.2f}\n"
        f"\x1b[2K  loudness (rms) ={estimate.rms:6.3f}\n"
        f"\x1b[2K  rgb            = ({r:3d},{g:3d},{b:3d})\n"
        f"\x1b[2K  fps            = {fps:4.1f}\n"
        f"\x1b[2K  {_format_chroma(estimate.chroma)}"
    )
    print(block, flush=True)


def _print_debug_line2(estimate, colour, fps: float) -> None:
    r, g, b = colour.as_rgb()
    swatch = f"\x1b[48;2;{r};{g};{b}m   \x1b[0m"
    line = (
        f"{swatch} chord={estimate.name:<6} "
        f"confidence={estimate.confidence:4.2f} "
        f"loudness (rms) ={estimate.rms:6.3f} "
        f"rgb=({r:3d},{g:3d},{b:3d}) "
        f"fps={fps:4.1f} "
        f"{_format_chroma(estimate.chroma)}"
    )
    print(f"\r\x1b[?7l\x1b[0K{line}", end="", flush=True)
