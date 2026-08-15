from __future__ import annotations

from abc import ABC, abstractmethod

from synesthesia.types import ColourState


class Renderer(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def render(self, colour: ColourState, dt: float) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    def __enter__(self) -> Renderer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
