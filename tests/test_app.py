from __future__ import annotations

import threading
from dataclasses import replace
from typing import Optional

import numpy as np

from synesthesia.app import _analysis_worker
from synesthesia.app_state import LatestTarget, OnsetQueue
from synesthesia.audio.base import AudioSource
from synesthesia.audio.framer import Framer
from synesthesia.colour.interpolator import ColourInterpolator
from synesthesia.types import ChordEstimate, ColourState, Quality


class FakeSource(AudioSource):
    """Serves a fixed array in hop-sized chunks, then EOF (mirrors test_framer)."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._pos = 0
        self.stopped = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def read(self, n: int) -> Optional[np.ndarray]:
        if self._pos + n > self._data.size:
            return None
        out = self._data[self._pos : self._pos + n]
        self._pos += n
        return out


class CountingDetector:
    """Returns a distinct estimate per frame; frame index 0 carries an onset."""

    def __init__(self) -> None:
        self.calls = 0

    def update(self, frame) -> ChordEstimate:
        onsets = (4,) if self.calls == 0 else ()  # exactly one attack, on frame 0
        est = ChordEstimate(
            root=self.calls % 12,
            quality=Quality.MAJ,
            confidence=0.9,
            rms=0.5,
            onsets=onsets,
        )
        self.calls += 1
        return est


class PassthroughMapper:
    """Maps root->hue trivially and forwards onsets as (pc, hue) pairs."""

    def map(self, estimate: ChordEstimate) -> ColourState:
        onsets = tuple((pc, pc / 12.0) for pc in estimate.onsets)
        return ColourState(
            hue=estimate.root / 12.0,
            saturation=1.0,
            value=1.0,
            intensity=1.0,
            onsets=onsets,
        )


def _make_framer(num_frames: int, win: int = 8, hop: int = 4) -> Framer:
    # Priming needs win/hop hops; each further frame needs one more hop.
    hops = (win // hop) + num_frames
    data = np.arange(hops * hop, dtype=np.float32)
    return Framer(FakeSource(data), window_size=win, hop=hop)


def test_worker_publishes_target_without_onsets():
    """Worker strips onsets off the colour slot and pushes them to the queue."""
    framer = _make_framer(num_frames=3)
    latest = LatestTarget()
    onsets = OnsetQueue()
    finished = threading.Event()

    _analysis_worker(
        framer, CountingDetector(), PassthroughMapper(), latest, onsets, finished
    )

    # Colour slot never carries onsets...
    assert latest.get().onsets == ()
    # ...but the single attack landed in the queue exactly once.
    assert onsets.drain() == ((4, 4 / 12.0),)
    assert onsets.drain() == ()
    # EOF sets finished so consumers can shut down.
    assert finished.is_set()


def test_worker_sets_finished_even_on_empty_stream():
    framer = _make_framer(num_frames=0)
    finished = threading.Event()
    _analysis_worker(
        framer, CountingDetector(), PassthroughMapper(),
        LatestTarget(), OnsetQueue(), finished,
    )
    assert finished.is_set()


def test_one_attack_one_drop_across_faster_render():
    """End-to-end regression: a single onset fires exactly one drop even when the
    render consumer reads several times per analysis frame."""
    framer = _make_framer(num_frames=1)
    latest = LatestTarget()
    onsets = OnsetQueue()
    finished = threading.Event()

    # Run analysis to completion (1 frame with an onset), populating the handoffs.
    _analysis_worker(
        framer, CountingDetector(), PassthroughMapper(), latest, onsets, finished
    )

    # Simulate the render loop consuming 4x within one analysis frame's lifetime.
    interp = ColourInterpolator(tau=1.0)
    spawned = 0
    for _ in range(4):
        colour = interp.update(latest.get(), dt=1 / 60)
        colour = replace(colour, onsets=onsets.drain())
        spawned += len(colour.onsets)

    assert spawned == 1  # was 3-4 before the queue fix


def test_worker_stops_early_when_finished_preset():
    """If the supervisor sets finished before/at start, the worker publishes nothing."""
    framer = _make_framer(num_frames=5)
    latest = LatestTarget()
    onsets = OnsetQueue()
    finished = threading.Event()
    finished.set()

    _analysis_worker(
        framer, CountingDetector(), PassthroughMapper(), latest, onsets, finished
    )

    # Nothing consumed; colour still at the initial BLACK and no onsets queued.
    assert latest.get().value == 0.0
    assert onsets.drain() == ()


def test_worker_thread_shutdown_joins_cleanly():
    """Worker running in a real thread exits and joins after EOF."""
    framer = _make_framer(num_frames=10)
    finished = threading.Event()
    t = threading.Thread(
        target=_analysis_worker,
        args=(framer, CountingDetector(), PassthroughMapper(),
              LatestTarget(), OnsetQueue(), finished),
        daemon=True,
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert finished.is_set()
