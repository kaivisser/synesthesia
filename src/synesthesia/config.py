from __future__ import annotations

import argparse
from dataclasses import dataclass

SOURCES = ("mic", "file", "monitor")
SCHEMES = ("scriabin", "chromatic", "fifths")
MOTIONS = (
    "solid",
    "plasma",
    "perlin",
    "aurora",
    "rain",
)


@dataclass(frozen=True)
class Config:
    """
    source:
        Which audio input to build: "mic", "file", or "monitor".
    file_path:
        Path to the audio file if source == file.
    device:
        Optional sounddevice device index/name for mic.
    monitor_sink:
        PipeWire sink node name whose monitor to capture (source == monitor).
        None -> auto-detect the current default sink.
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

    onset_rise:
        Min per-frame energy jump (peak-normalised) for a note attack to fire an
        onset (drives rain drops). Higher = only sharp attacks trigger.
    onset_floor:
        Absolute energy gate a note must exceed to fire an onset. Higher =
        only prominent notes drop rain (suppresses leakage/quiet notes).
    onset_cooldown_frames:
        Per-note lockout (frames) after an onset fires, so a held note strikes
        once. Higher = ignores fast re-strikes (sparser rain).
    onset_min_rms:
        Absolute raw-loudness gate for onsets: notes quieter than this never
        drop rain (even if loudest this frame). Distinct from silence_rms.

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

    rain_speed:
        (motion=rain) How fast a drop's ring expands (screen-units/sec).
    rain_max_radius:
        (motion=rain) Radius at which a drop fully fades and is culled.
    rain_depth:
        (motion=rain) Peak brightness a fresh drop adds at its crest [0, 1].
    rain_hue_pull:
        (motion=rain) How strongly a drop tints toward its note's hue [0, 1].
    rain_max_drops:
        (motion=rain) Cap on simultaneously live drops (oldest culled).
    """

    # source controls
    source: str = "mic"
    file_path: str | None = None
    device: int | str | None = None
    monitor_sink: str | None = None
    gain: float = 45.0

    # framer controls
    sample_rate: int = 11000  # 44100
    window_size: int = 4096 * 2
    hop: int = 2048 * 2

    # detector controls
    silence_rms: float = 0.1
    smoothing: float = 0.7

    # colour controls
    scheme: str = "scriabin"
    fade_tau: float = 0.9

    # renderer controls
    motion: str = "perlin"
    fps: float = 120.0
    debug: bool = False

    # rain motion controls
    onset_rise: float = 0.3
    onset_floor: float = 0.3
    onset_cooldown_frames: int = 5
    onset_min_rms: float = 0.8
    rain_speed: float = 0.3
    rain_max_radius: float = 0.5
    rain_depth: float = 0.6
    rain_hue_pull: float = 0.6
    rain_max_drops: int = 12


def parse_args(argv: list[str] | None = None) -> Config:
    d = Config()
    p = argparse.ArgumentParser(
        prog="synesthesia",
        description="A Python music visualizer.",
    )

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
        help="sounddevice device index or name (mic)",
    )
    p.add_argument(
        "--monitor-sink",
        dest="monitor_sink",
        default=d.monitor_sink,
        help=(
            "PipeWire sink node name to capture (source=monitor); "
            "default = current default sink (see `pw-link -o`)"
        ),
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
    p.add_argument(
        "--onset-rise",
        type=float,
        default=d.onset_rise,
        help="min per-frame energy jump for a note attack (rain drops)",
    )
    p.add_argument(
        "--onset-floor",
        type=float,
        default=d.onset_floor,
        help="absolute energy gate a note must exceed to trigger a rain drop",
    )
    p.add_argument(
        "--onset-cooldown-frames",
        type=int,
        default=d.onset_cooldown_frames,
        help="per-note lockout in frames after an onset (higher = sparser rain)",
    )
    p.add_argument(
        "--onset-min-rms",
        type=float,
        default=d.onset_min_rms,
        help="absolute loudness gate for onsets (quiet notes never drop rain)",
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

    # rain motion args (only used when --motion rain)
    p.add_argument(
        "--rain-speed",
        type=float,
        default=d.rain_speed,
        help="(rain) drop ring expansion speed (screen-units/sec)",
    )
    p.add_argument(
        "--rain-max-radius",
        type=float,
        default=d.rain_max_radius,
        help="(rain) radius at which a drop fully fades and is culled",
    )
    p.add_argument(
        "--rain-depth",
        type=float,
        default=d.rain_depth,
        help="(rain) peak brightness a fresh drop adds at its crest [0, 1]",
    )
    p.add_argument(
        "--rain-hue-pull",
        type=float,
        default=d.rain_hue_pull,
        help="(rain) how strongly a drop tints toward its note's hue [0, 1]",
    )
    p.add_argument(
        "--rain-max-drops",
        type=int,
        default=d.rain_max_drops,
        help="(rain) cap on simultaneously live drops (oldest culled)",
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
        monitor_sink=ns.monitor_sink,
        gain=ns.gain,
        sample_rate=d.sample_rate,
        window_size=ns.window_size,
        hop=ns.hop,
        silence_rms=ns.silence_rms,
        smoothing=ns.smoothing,
        onset_rise=ns.onset_rise,
        onset_floor=ns.onset_floor,
        onset_cooldown_frames=ns.onset_cooldown_frames,
        onset_min_rms=ns.onset_min_rms,
        scheme=ns.scheme,
        fade_tau=ns.fade_tau,
        motion=ns.motion,
        fps=ns.fps,
        debug=ns.debug,
        rain_speed=ns.rain_speed,
        rain_max_radius=ns.rain_max_radius,
        rain_depth=ns.rain_depth,
        rain_hue_pull=ns.rain_hue_pull,
        rain_max_drops=ns.rain_max_drops,
    )
