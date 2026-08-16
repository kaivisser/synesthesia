from __future__ import annotations

import argparse
from dataclasses import dataclass

SOURCES = ("mic", "file", "monitor")
SCHEMES = ("scriabin", "chromatic", "fifths")
MOTIONS = (
    "solid",
    "breathe",
    "plasma",
    "perlin",
    "radial",
    "gradient-drift",
    "aurora",
    "ripple",
)


@dataclass(frozen=True)
class Config:
    """
    source:
        Which audio input to build: "mic", "file", or "monitor".
    file_path:
        Path to the audio file if source == file.
    device:
        Optional sounddevice device index/name for mic/monitor.
    gain:
        Digital gain applied post-downmix.

    sample_rate:
        Fixed analysis rate; every source resamples to this (default = 44.1 kHz).
    window_size:
        Framer window length in samples. Bigger = better pitch accuracy.
    hop:
        Framer hop in samples. Sets the analysis frame rate (sr/hop).

    silence_rms:
        RMS below this gates to NO_CHORD -> fade to black.
    smoothing:
        Detector chroma EMA factor in [0, 1); higher = smoother/slower.

    scheme:
        Colour scheme name: "scriabin" | "chromatic" | "fifths".
    fade_tau:
        Colour interpolation time constant in seconds; larger = dreamier.

    motion:
        Renderer motion style (default = perlin).
    fps:
        Target render frames per second (decisions log: ~30 fps).
    debug:
        When True, overlay live pipeline info (chord, confidence, RMS, RGB,
        fps, scheme/motion) on top of the colour field.
    """

    # source controls
    source: str = "mic"
    file_path: str | None = None
    device: int | str | None = None
    gain: float = 1.0

    # framer controls
    sample_rate: int = 44100
    window_size: int = 4096
    hop: int = 2048

    # detector controls
    silence_rms: float = 0.05
    smoothing: float = 0.6

    # colour controls
    scheme: str = "scriabin"
    fade_tau: float = 0.6

    # renderer controls
    motion: str = "perlin"
    fps: float = 60.0
    debug: bool = False


def parse_args(argv: list[str] | None = None) -> Config:
    d = Config()
    p = argparse.ArgumentParser(
        prog="synesthesia",
        description="A Python music visualizer.",
    )

    # source args
    p.add_argument(
        "--source", choices=SOURCES, default=d.source, help="audio input to use"
    )
    p.add_argument(
        "--file",
        dest="file_path",
        default=d.file_path,
        help="audio file path (required for --source file)",
    )
    p.add_argument(
        "--device",
        default=d.device,
        help="sounddevice device index or name (mic/monitor)",
    )
    p.add_argument(
        "--gain",
        type=float,
        default=d.gain,
        help="digital gain applied after mono downmix",
    )

    # framer args
    p.add_argument(
        "--window-size",
        type=int,
        default=d.window_size,
        help="framer window length in samples",
    )
    p.add_argument(
        "--hop", type=int, default=d.hop, help="framer hop length in samples"
    )

    # detector args
    p.add_argument(
        "--silence-rms",
        type=float,
        default=d.silence_rms,
        help="RMS below this is treated as silence (fade to black)",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=d.smoothing,
        help="detector chroma EMA factor in [0, 1)",
    )

    # colour args
    p.add_argument(
        "--scheme", choices=SCHEMES, default=d.scheme, help="pitch-to-hue colour scheme"
    )
    p.add_argument(
        "--fade-tau",
        type=float,
        default=d.fade_tau,
        help="colour fade time constant in seconds (larger = dreamier)",
    )

    # render args
    p.add_argument(
        "--motion",
        choices=MOTIONS,
        default=d.motion,
        help="motion style for the colour field",
    )
    p.add_argument(
        "--fps", type=float, default=d.fps, help="target render frames per second"
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=d.debug,
        help="overlay live pipeline debug info (chord, confidence, RMS, fps)",
    )

    ns = p.parse_args(argv)

    device = ns.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    if ns.source == "file" and not ns.file_path:
        p.error("--file is required when --source file")

    return Config(
        source=ns.source,
        file_path=ns.file_path,
        device=device,
        gain=ns.gain,
        sample_rate=d.sample_rate,
        window_size=ns.window_size,
        hop=ns.hop,
        silence_rms=ns.silence_rms,
        smoothing=ns.smoothing,
        scheme=ns.scheme,
        fade_tau=ns.fade_tau,
        motion=ns.motion,
        fps=ns.fps,
        debug=ns.debug,
    )
