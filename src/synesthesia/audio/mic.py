from __future__ import annotations

from typing import Optional

import numpy as np
import sounddevice as sd

from synesthesia.audio.base import AudioSource
from synesthesia.audio.ringbuffer import RingBuffer


class MicSource(AudioSource):
    """
    Live microphone input.

    Opens a sounddevice.InputStream at the fixed analysis rate (44.1 kHz);
    if the device runs natively at another rate the driver resamples for us.
    """

    def __init__(
        self,
        device: Optional[int | str] = None,
        buffer_seconds: float = 1.0,
        gain: float = 1.0,
    ) -> None:
        self.device = device

        # Digital gain applied post-downmix.
        self.gain = float(gain)

        capacity = int(buffer_seconds * self.sample_rate)
        self._ring = RingBuffer(capacity=capacity)

        self._stream: Optional[sd.InputStream] = None
        # Accumulates PortAudio status flags seen in the callback for debugging
        self._status: sd.CallbackFlags = sd.CallbackFlags()

    def start(self) -> None:
        if self._stream is not None:
            return

        info = sd.query_devices(self.device, "input")
        channels = int(info["max_input_channels"]) or 1

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,  # driver resamples if device differs
            channels=channels,
            dtype="float32",
            blocksize=0,  # let PortAudio choose an optimal block size
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def read(self, n: int) -> Optional[np.ndarray]:
        return self._ring.read(n)

    def stop(self) -> None:
        # Alwasy silence the producer before closing the buffer.
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._ring.close()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # number of sample-frames in this block. RingBuffer decides how many samples to keep
        time,  # PortAudio CData timestamp struct; unused
        status: sd.CallbackFlags,
    ) -> None:
        # NEVER raise an exception here as it can tear the stream down messily.
        if status:
            self._status |= status

        # indata is (frames, channels) float32. Average channels -> mono (frames,).
        mono = indata.mean(axis=1).astype(np.float32, copy=False)
        if self.gain != 1.0:
            mono *= self.gain
        self._ring.write(mono)

    @property
    def dropped(self) -> int:
        return self._ring.dropped

    @property
    def status(self) -> sd.CallbackFlags:
        return self._status
