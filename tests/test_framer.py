from __future__ import annotations

from typing import Optional

import numpy as np

from synesthesia.audio.base import AudioSource
from synesthesia.audio.framer import Framer


class FakeSource(AudioSource):
    """Deterministic source that serves a fixed array in hop-sized chunks."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._pos = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self, n: int) -> Optional[np.ndarray]:
        if self._pos + n > self._data.size:
            return None
        out = self._data[self._pos : self._pos + n]
        self._pos += n
        return out


def test_first_frame_is_full_window():
    win, hop = 8, 4
    src = FakeSource(np.arange(64))
    f = Framer(src, window_size=win, hop=hop)

    frame = f.read()
    assert frame is not None
    assert frame.shape == (win,)
    assert frame.dtype == np.float32


def test_hann_applied():
    win, hop = 8, 4
    src = FakeSource(np.ones(64))
    f = Framer(src, window_size=win, hop=hop)

    frame = f.read()
    assert frame is not None
    hann = np.hanning(win).astype(np.float32)
    assert np.allclose(frame, hann)


def test_overlap_advances_by_hop():
    win, hop = 8, 4
    src = FakeSource(np.arange(64, dtype=np.float32))
    f = Framer(src, window_size=win, hop=hop)
    hann = np.hanning(win).astype(np.float32)

    # After priming (win/hop = 2 hops), the window holds samples [0..7].
    frame1 = f.read()
    assert frame1 is not None
    assert np.allclose(frame1, np.arange(0, 8) * hann)

    # One more hop advances the window to [4..11].
    frame2 = f.read()
    assert frame2 is not None
    assert np.allclose(frame2, np.arange(4, 12) * hann)


def test_returns_none_at_eof_and_is_idempotent():
    win, hop = 8, 4
    # Exactly enough for priming (2 hops), nothing left for a further frame.
    src = FakeSource(np.arange(8, dtype=np.float32))
    f = Framer(src, window_size=win, hop=hop)

    assert f.read() is not None
    assert f.read() is None
    assert f.read() is None


def test_frames_generator_yields_expected_count():
    win, hop = 8, 4
    # 4 hops of data: 2 prime the window, leaving 2 more frames -> 3 total.
    src = FakeSource(np.arange(16, dtype=np.float32))
    f = Framer(src, window_size=win, hop=hop)

    frames = list(f.frames())
    assert len(frames) == 3
    assert all(fr.shape == (win,) for fr in frames)


def test_no_overlap_when_hop_equals_window():
    win = hop = 4
    src = FakeSource(np.arange(16, dtype=np.float32))
    f = Framer(src, window_size=win, hop=hop)
    hann = np.hanning(win).astype(np.float32)

    frame1 = f.read()
    frame2 = f.read()
    assert frame1 is not None and frame2 is not None
    assert np.allclose(frame1, np.arange(0, 4) * hann)
    assert np.allclose(frame2, np.arange(4, 8) * hann)


def test_invalid_params():
    src = FakeSource(np.arange(8))
    for kwargs in ({"window_size": 0}, {"hop": 0}, {"window_size": 4, "hop": 8}):
        try:
            Framer(src, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
