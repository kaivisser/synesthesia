from __future__ import annotations

from abc import ABC, abstractmethod

from synesthesia.types import ColourState


class Renderer(ABC):
    """
    Abstract base class for all render backends (PLAN.md §3.5).

    A renderer is the final pipeline stage: it turns the smoothed ``ColourState``
    produced each frame into something the human sees. v1 targets the terminal
    (``TerminalRenderer``); a graphical backend (e.g. Pygame) can be dropped in
    later by implementing this same interface, with no upstream changes.

    Like ``AudioSource``, this is an ABC rather than a Protocol: every backend
    shares the *exact* same context-manager teardown (enter alt-screen / hide
    cursor on start, restore on stop), so ``__enter__``/``__exit__`` live once
    here and only ``start``/``render``/``stop`` are abstract.
    """

    @abstractmethod
    def start(self) -> None:
        """
        Acquire the display and prepare to draw (e.g. enter the alternate
        screen buffer, hide the cursor, clear).
        """
        pass

    @abstractmethod
    def render(self, colour: ColourState, dt: float) -> None:
        """
        Draw a single frame for ``colour``. ``dt`` is the seconds elapsed since
        the previous frame, so time-based motion can advance at real speed.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Release the display and restore the terminal to its original state
        (leave the alternate screen, show the cursor).
        """
        pass

    def __enter__(self) -> Renderer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
