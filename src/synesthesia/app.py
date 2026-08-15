from __future__ import annotations

from synesthesia.audio.base import AudioSource
from synesthesia.audio.mic import MicSource
from synesthesia.config import Config


def build_source(config: Config) -> AudioSource:
    """
    Construct the AudioSource named by ``config.source`` (PLAN.md §3.1).

    The one place that knows how to turn config into a concrete source, so the
    main loop stays source-agnostic (it only ever sees the ``AudioSource``
    interface). ``file`` and ``monitor`` are planned (§5 step 6) and raise a
    clear error until their classes land, rather than failing obscurely later.
    """
    if config.source == "mic":
        return MicSource(device=config.device, gain=config.gain)

    if config.source in ("file", "monitor"):
        raise NotImplementedError(
            f"source {config.source!r} not implemented yet (see PLAN.md §5 step 6)"
        )

    raise ValueError(f"unknown source {config.source!r}")
