from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AudioSource(ABC):
    """
    Abstract base class for all audio inputs
    Every source produces a stream of mono float32
    samples at a fixed sample rate of 44100hz.
    """

    sample_rate: int = 44100

    @abstractmethod
    def start(self) -> None:
        """
        Begin producing audio.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop producing audio and release resources.
        """
        pass

    @abstractmethod
    def read(self, n: int) -> Optional[np.ndarray]:
        """
        Return exactly n mono float32 samples.
        Blocks until n samples are available and returns None at end of stream.
        """
        pass

    def __enter__(self) -> AudioSource:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
