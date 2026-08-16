from __future__ import annotations

from typing import Optional

import sounddevice as sd

from synesthesia.audio.mic import MicSource


class MonitorSource(MicSource):
    """
    Live system-audio capture (PLAN.md §3.1).

    On Linux, PulseAudio/PipeWire exposes a ".monitor" input device for every
    audio sink (e.g. "alsa_output...analog-stereo.monitor"). Recording from a
    monitor device captures whatever the system is *playing* -- so pointing a
    mic-style capture stream at it turns any music player into an input for the
    visualizer. This is why the plan calls it "nearly free once mic works":
    it is a MicSource whose device just happens to be a monitor.

    The only added responsibility over MicSource is *finding* that monitor
    device. If the caller passes an explicit ``device``, we trust it verbatim;
    otherwise we auto-detect the monitor of the default output sink.
    """

    def __init__(
        self,
        device: Optional[int | str] = None,
        buffer_seconds: float = 1.0,
        gain: float = 1.0,
        max_channels: int = 2,
    ) -> None:
        if device is None:
            device = _find_default_monitor()
        super().__init__(
            device=device,
            buffer_seconds=buffer_seconds,
            gain=gain,
            max_channels=max_channels,
        )
