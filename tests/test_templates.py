from __future__ import annotations

import numpy as np

from synesthesia.detect.templates import build_templates, match
from synesthesia.types import Quality

_MAJOR_INTERVALS = (0, 4, 7)
_MINOR_INTERVALS = (0, 3, 7)


def _chroma_for(root: int, intervals: tuple[int, ...]) -> np.ndarray:
    """A clean chroma with energy only on one triad's chord tones."""
    c = np.zeros(12, dtype=np.float32)
    for iv in intervals:
        c[(root + iv) % 12] = 1.0
    return c


def test_shape_and_labels():
    templates, labels = build_templates()

    assert templates.shape == (24, 12)
    assert len(labels) == 24
    # First 12 rows are majors C..B, next 12 are minors C..B.
    assert labels[:12] == [(r, Quality.MAJ) for r in range(12)]
    assert labels[12:] == [(r, Quality.MIN) for r in range(12)]


def test_each_template_has_three_tones():
    templates, _ = build_templates()
    assert np.all(templates.sum(axis=1) == 3.0)
    assert set(np.unique(templates)) <= {0.0, 1.0}


def test_major_template_intervals():
    templates, labels = build_templates()
    # C major (row 0) should light C, E, G.
    expected = _chroma_for(0, _MAJOR_INTERVALS)
    assert np.array_equal(templates[0], expected)


def test_minor_template_intervals():
    templates, labels = build_templates()
    # A minor (root 9) lights A, C, E.
    row = labels.index((9, Quality.MIN))
    expected = _chroma_for(9, _MINOR_INTERVALS)
    assert np.array_equal(templates[row], expected)


def test_match_identifies_all_major_triads():
    templates, labels = build_templates()
    for root in range(12):
        chroma = _chroma_for(root, _MAJOR_INTERVALS)
        got_root, got_quality, conf = match(chroma, templates, labels)
        assert (got_root, got_quality) == (root, Quality.MAJ)
        assert conf == 1.0  # exact template match -> cosine similarity 1.0


def test_match_identifies_all_minor_triads():
    templates, labels = build_templates()
    for root in range(12):
        chroma = _chroma_for(root, _MINOR_INTERVALS)
        got_root, got_quality, conf = match(chroma, templates, labels)
        assert (got_root, got_quality) == (root, Quality.MIN)
        assert conf == 1.0


def test_match_distinguishes_major_from_minor():
    templates, labels = build_templates()
    # C minor (C, D#, G) must not be read as any major.
    chroma = _chroma_for(0, _MINOR_INTERVALS)
    root, quality, _ = match(chroma, templates, labels)
    assert (root, quality) == (0, Quality.MIN)


def test_confidence_in_unit_range():
    templates, labels = build_templates()
    rng = np.random.default_rng(0)
    for _ in range(50):
        chroma = rng.random(12).astype(np.float32)
        _, _, conf = match(chroma, templates, labels)
        assert 0.0 <= conf <= 1.0 + 1e-6


def test_zero_chroma_returns_zero_confidence():
    templates, labels = build_templates()
    root, quality, conf = match(np.zeros(12, dtype=np.float32), templates, labels)
    assert conf == 0.0
    # Falls back to the first label rather than erroring on a 0-norm vector.
    assert (root, quality) == labels[0]


def test_match_robust_to_noise_and_loudness():
    templates, labels = build_templates()
    rng = np.random.default_rng(1)
    # C major with the tones dominant plus small leakage into other bins, and
    # overall scaled up (cosine similarity should ignore the loudness scaling).
    chroma = _chroma_for(0, _MAJOR_INTERVALS) * 5.0
    chroma += rng.random(12).astype(np.float32) * 0.1
    root, quality, _ = match(chroma, templates, labels)
    assert (root, quality) == (0, Quality.MAJ)
