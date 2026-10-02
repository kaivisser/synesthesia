from __future__ import annotations

import librosa
import numpy as np

ANALYSIS_SR = 44100


def chroma(frame: np.ndarray, sample_rate: int = ANALYSIS_SR) -> np.ndarray:
    """
    Collapse one windowed audio frame into a 12-element pitch-class vector.
    Uses Constant-Q transform over STFT as this maps better to pitches.
    """

    frame = np.ascontiguousarray(frame, dtype=np.float32)
    cq = librosa.feature.chroma_cqt(y=frame, sr=sample_rate)
    return cq.mean(axis=1).astype(np.float32)
