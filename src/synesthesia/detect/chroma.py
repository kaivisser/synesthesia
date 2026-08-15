from __future__ import annotations

import librosa
import numpy as np

ANALYSIS_SR = 44100


def chroma(frame: np.ndarray, sample_rate: int = ANALYSIS_SR) -> np.ndarray:
    """
    Collapse one windowed audio frame into a 12-element pitch-class vector.

    A chromagram folds the whole spectrum onto the 12 pitch classes (C, C#, ...,
    B), summing energy across octaves. So a C played anywhere on the keyboard
    lights the same "C" bin -- exactly what chord detection wants, since a chord
    is defined by its pitch classes, not by which octave they sound in.

    We use the CQT (constant-Q transform) variant rather than the STFT one: the
    CQT spaces its frequency bins logarithmically, matching the equal-tempered
    scale (each semitone is a fixed ratio apart), so musical pitches land cleanly
    on bins instead of being smeared across the linear FFT grid.

    Parameters
    ----------
    frame:
        1-D float32 audio, already Hann-windowed by the Framer (PLAN.md §3.2).
    sample_rate:
        Sample rate of ``frame`` (fixed 44.1 kHz analysis rate).

    Returns
    -------
    np.ndarray
        A 12-element float32 vector, one entry per pitch class starting at C.
        Not normalised here -- the template matcher uses cosine similarity, which
        is scale-invariant, so overall loudness is handled separately via RMS.
    """
    frame = np.ascontiguousarray(frame, dtype=np.float32)
    cq = librosa.feature.chroma_cqt(y=frame, sr=sample_rate)
    return cq.mean(axis=1).astype(np.float32)
