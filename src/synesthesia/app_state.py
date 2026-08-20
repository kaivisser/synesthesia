from __future__ import annotations

import threading

from synesthesia.types import BLACK, NO_CHORD, ChordEstimate, ColourState


class LatestTarget:
    """
    Single-producer/single-consumer "latest value wins" handoff for the
    concurrency split (PLAN §2).

    The analysis thread computes a *target* ``ColourState`` (the raw
    ``ColourMapper`` output for the current chord -- NOT yet interpolated) and
    ``publish()``es it. The render thread ``get()``s the freshest target every
    frame and does its own smoothing/motion at render fps. Only the newest
    target matters: if analysis publishes several times between reads, older
    targets are harmlessly overwritten (mirrors the drop-oldest audio policy,
    §3.1).

    Alongside the target colour it carries the ``ChordEstimate`` that produced
    it, purely so the debug consumer can show chord/confidence/RMS/notes without
    reaching across to the analysis thread's locals. The render (non-debug)
    consumer just ignores it.

    Note this deliberately carries only the *smoothed* colour fields
    (hue/sat/value/intensity). Instantaneous ``onsets`` must NOT ride here --
    latest-wins would drop attacks the render thread hasn't seen yet; they get a
    separate queue in a later chunk.

    Starts at ``BLACK`` / ``NO_CHORD`` so consumers have a valid value before
    the first chord is detected.
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
        """Return the freshest published target (render thread)."""
        with self._lock:
            return self._target

    def get_with_estimate(self) -> tuple[ColourState, ChordEstimate]:
        """Return the freshest (target, estimate) pair (debug consumer)."""
        with self._lock:
            return self._target, self._estimate


class OnsetQueue:
    """
    Thread-safe drain-once queue for note-attack onsets (PLAN §2 concurrency
    split; the event counterpart to ``LatestTarget``).

    Onsets are *instantaneous events*, not a retained state, so they must NOT
    ride the latest-wins colour slot: a faster render loop re-reads that slot
    several times per analysis frame and would spawn a rain drop each time (one
    key press -> multiple drops). Instead the analysis thread ``push()``es each
    frame's onsets and the render thread ``drain()``s them exactly once per
    frame -- so every attack fires exactly one drop regardless of the
    analysis/render fps ratio (render faster -> most frames drain empty; render
    slower -> a frame drains several accumulated onsets, none lost).

    Each onset is a ``(pitch_class, hue)`` pair (from ``ColourMapper``). The
    queue is unbounded; Rain's ``max_drops`` bounds the *visual* cost, so extreme
    bursts self-limit on screen rather than here.
    """

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
