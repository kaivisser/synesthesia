from __future__ import annotations

import threading

from synesthesia.types import BLACK, NO_CHORD, ChordEstimate, ColourState


class LatestTarget:
    """
    Single-producer/single-consumer latest value wins handoff
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target = BLACK
        self._estimate = NO_CHORD

    def publish(self, target: ColourState, estimate: ChordEstimate) -> None:
        """Store the newest target + estimate (analysis thread). Never blocks."""
        with self._lock:
            self._target = target
            self._estimate = estimate

    def get(self) -> ColourState:
        """Return the last published target"""
        with self._lock:
            return self._target

    def get_with_estimate(self) -> tuple[ColourState, ChordEstimate]:
        """Return the last (target, estimate) pair for debug."""
        with self._lock:
            return self._target, self._estimate


class OnsetQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[tuple[int, float]] = []

    def push(self, onsets: tuple[tuple[int, float], ...]) -> None:
        """Append this analysis frame's onsets (analysis thread). No-op if empty."""
        if not onsets:
            return
        with self._lock:
            self._pending.extend(onsets)

    def drain(self) -> tuple[tuple[int, float], ...]:
        """Remove and return all pending onsets (render thread), once each."""
        with self._lock:
            if not self._pending:
                return ()
            out = tuple(self._pending)
            self._pending.clear()
            return out
