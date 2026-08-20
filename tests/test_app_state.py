from __future__ import annotations

import threading

from synesthesia.app_state import LatestTarget, OnsetQueue
from synesthesia.types import BLACK, NO_CHORD, ChordEstimate, ColourState, Quality


def _colour(hue: float) -> ColourState:
    return ColourState(hue=hue, saturation=1.0, value=1.0, intensity=1.0)


def _estimate(root: int) -> ChordEstimate:
    return ChordEstimate(root=root, quality=Quality.MAJ, confidence=0.9, rms=0.5)


# --------------------------------------------------------------------------- #
# LatestTarget
# --------------------------------------------------------------------------- #
def test_latest_starts_black_and_no_chord():
    lt = LatestTarget()
    assert lt.get() == BLACK
    target, estimate = lt.get_with_estimate()
    assert target == BLACK
    assert estimate is NO_CHORD


def test_latest_returns_most_recent_publish():
    lt = LatestTarget()
    lt.publish(_colour(0.1), _estimate(1))
    lt.publish(_colour(0.2), _estimate(2))

    target, estimate = lt.get_with_estimate()
    assert target.hue == 0.2
    assert estimate.root == 2


def test_latest_wins_overwrites_silently():
    """Several publishes between reads keep only the newest (drop-oldest spirit)."""
    lt = LatestTarget()
    for i in range(10):
        lt.publish(_colour(i / 10.0), _estimate(i))
    assert lt.get().hue == 0.9


def test_latest_get_is_repeatable_without_publish():
    lt = LatestTarget()
    lt.publish(_colour(0.5), _estimate(5))
    assert lt.get().hue == 0.5
    assert lt.get().hue == 0.5  # retained state: same value until next publish


# --------------------------------------------------------------------------- #
# OnsetQueue
# --------------------------------------------------------------------------- #
def test_onset_queue_starts_empty():
    q = OnsetQueue()
    assert q.drain() == ()


def test_onset_push_then_drain_once():
    q = OnsetQueue()
    q.push(((4, 0.1),))
    assert q.drain() == ((4, 0.1),)
    # Drained exactly once: a second drain sees nothing.
    assert q.drain() == ()


def test_onset_push_empty_is_noop():
    q = OnsetQueue()
    q.push(())
    assert q.drain() == ()


def test_onset_accumulates_across_pushes():
    """Render slower than analysis: multiple analysis frames drain together, none lost."""
    q = OnsetQueue()
    q.push(((4, 0.1),))
    q.push(((7, 0.2), (0, 0.3)))
    assert q.drain() == ((4, 0.1), (7, 0.2), (0, 0.3))
    assert q.drain() == ()


def test_onset_single_attack_drains_once_over_many_reads():
    """Render faster than analysis: one push -> exactly one non-empty drain.

    This is the regression guard for the multi-raindrop bug: onsets must NOT be
    re-delivered on each faster render frame.
    """
    q = OnsetQueue()
    q.push(((4, 0.1),))

    drains = [q.drain() for _ in range(5)]
    non_empty = [d for d in drains if d]
    assert non_empty == [((4, 0.1),)]  # fired exactly once
    assert sum(len(d) for d in drains) == 1


def test_onset_queue_thread_safe_no_loss():
    """Concurrent producer pushes and consumer drains: every onset seen once."""
    q = OnsetQueue()
    total = 5000
    done = threading.Event()

    def producer():
        for i in range(total):
            q.push(((i % 12, (i % 12) / 12.0),))
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    seen = 0
    while not (done.is_set() and seen >= total):
        seen += len(q.drain())
    seen += len(q.drain())  # final sweep for anything pushed after the check
    t.join(timeout=5.0)

    assert seen == total
