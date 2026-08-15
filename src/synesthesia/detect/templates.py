from __future__ import annotations

import numpy as np

from synesthesia.types import Quality

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_MAJOR_INTERVALS = (0, 4, 7)
_MINOR_INTERVALS = (0, 3, 7)


def _triad_vector(root: int, intervals: tuple[int, ...]) -> np.ndarray:
    """Binary 12-vector with 1s on the chord tones of one triad."""
    vec = np.zeros(12, dtype=np.float32)
    for iv in intervals:
        vec[(root + iv) % 12] = 1.0
    return vec


def build_templates() -> tuple[np.ndarray, list[tuple[int, Quality]]]:
    """
    Build the v1 chord vocabulary: 12 major + 12 minor triads (PLAN.md §3.3).

    Returns
    -------
    templates:
        Float32 array of shape (24, 12). Each row is a binary pitch-class
        template (1.0 on the three chord tones, 0.0 elsewhere). Rows are ordered
        as all 12 major triads (roots C..B) followed by all 12 minor triads.
    labels:
        Parallel list of (root, quality) for each row, where root is 0-11 and
        quality is a Quality enum member. Lets the matcher report which chord won
        and hands the colour mapper the root/quality it needs downstream.
    """
    rows: list[np.ndarray] = []
    labels: list[tuple[int, Quality]] = []

    for root in range(12):
        rows.append(_triad_vector(root, _MAJOR_INTERVALS))
        labels.append((root, Quality.MAJ))
    for root in range(12):
        rows.append(_triad_vector(root, _MINOR_INTERVALS))
        labels.append((root, Quality.MIN))

    return np.stack(rows), labels


def match(
    chroma: np.ndarray,
    templates: np.ndarray,
    labels: list[tuple[int, Quality]],
) -> tuple[int, Quality, float]:
    """
    Example - imagine we have a C major triad:

    Against every template we compute the dot product;

    C major triad vs C-maj template:
                 C     C#    D     D#    E     F     F#    G     G#    A     A#    B
    chroma:      0.9   0.1   0.0   0.0   0.8   0.0   0.0   0.85  0.0   0.1   0.0   0.0
    C-maj tmpl:  1     0     0     0     1     0     0     1     0     0     0     0
    product:     0.9   0     0     0     0.8   0     0     0.85  0     0     0     0
                                                                  sum = 2.55

    C major traid vs A-maj template:
                 C     C#    D     D#    E     F     F#    G     G#    A     A#    B
    chroma:      0.9   0.1   0.0   0.0   0.8   0.0   0.0   0.85  0.0   0.1   0.0   0.0
    A-min tmpl:  1     0     0     0     1     0     0     0     0     1     0     0
    product:     0.9   0     0     0     0.8   0     0     0     0     0.1   0     0
                                                                  sum = 1.8


    We normlize this score to discount loudness and take the largest value
    """
    chroma = np.ascontiguousarray(chroma, dtype=np.float32)

    chroma_norm = float(np.linalg.norm(chroma))
    if chroma_norm == 0.0:
        root, quality = labels[0]
        return root, quality, 0.0

    # cos = (T @ c) / (|T_row| * |c|), computed for all rows at once.
    template_norms = np.linalg.norm(templates, axis=1)
    sims = (templates @ chroma) / (template_norms * chroma_norm)

    best = int(np.argmax(sims))
    root, quality = labels[best]
    return root, quality, float(sims[best])
