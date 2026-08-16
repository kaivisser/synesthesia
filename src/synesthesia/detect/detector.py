from __future__ import annotations

from typing import Optional

import numpy as np

from synesthesia.detect.chroma import chroma as compute_chroma
from synesthesia.detect.templates import build_templates, match
from synesthesia.types import NO_CHORD, ChordEstimate

ANALYSIS_SR = 44100


class ChordDetector:
    """
    Uses an exponential moving average over each chrome to discount transient chords.

    For example;

    raw per-frame chord
    frame:   1    2    3    4    5    6    7    8
    chord:   C    C    C    Am   C    C    F    C

    We don't want to render frames 4 & 7 as they are one-frame flickers,
    instead we want to simulate a smooth transition between chords over a sustained period.
    """

    def __init__(
        self,
        sample_rate: int = ANALYSIS_SR,
        silence_rms: float = 0.01,
        smoothing: float = 0.6,
        onset_rise: float = 0.15,
        onset_floor: float = 0.2,
        onset_cooldown_frames: int = 3,
        onset_min_rms: float = 0.05,
    ) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0.0, 1.0)")

        self.sample_rate = sample_rate
        self.silence_rms = float(silence_rms)
        self.smoothing = float(smoothing)

        # Onset (note-attack) detection tuning. Onsets are computed on the RAW
        # chroma (pre-EMA) so attacks stay crisp even though the chord colour is
        # smoothed. A pitch class fires when its energy jumps by >= onset_rise
        # since last frame AND sits above onset_floor -- both on the *peak-
        # normalised* chroma in [0, 1], so they answer "which note stands out
        # this frame" regardless of loudness. onset_min_rms adds an *absolute*
        # loudness gate (on the raw frame RMS) so quiet notes don't drop rain
        # even when they're the loudest thing playing. After firing a note can't
        # fire again for onset_cooldown_frames frames, so a held note strikes
        # once. (onset_min_rms >= silence_rms in practice: silence already gates
        # everything; this is a higher bar specific to spawning drops.)
        self.onset_rise = float(onset_rise)
        self.onset_floor = float(onset_floor)
        self.onset_cooldown_frames = int(onset_cooldown_frames)
        self.onset_min_rms = float(onset_min_rms)

        self._templates, self._labels = build_templates()

        self._ema: Optional[np.ndarray] = None
        self._prev_chroma: Optional[np.ndarray] = None
        self._cooldown = np.zeros(12, dtype=np.int64)

    def update(self, frame: np.ndarray) -> ChordEstimate:
        """
        Process one windowed frame and return a smoothed ChordEstimate.

        Call once per frame; the EMA state carries between calls. Silence
        (RMS < threshold) short-circuits to NO_CHORD and decays the smoothed
        chroma toward zero, so sound resuming after a pause fades back in rather
        than snapping to the pre-silence chord.
        """
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))

        if rms < self.silence_rms:
            if self._ema is not None:
                self._ema *= self.smoothing
            # Silence resets onset state: the next attack after a pause should
            # register cleanly rather than diff against stale pre-silence chroma.
            self._prev_chroma = None
            self._cooldown[:] = 0
            return NO_CHORD

        chroma = compute_chroma(frame, self.sample_rate)

        onsets = self._detect_onsets(chroma, rms)

        if self._ema is None:
            self._ema = chroma
        else:
            self._ema = self.smoothing * self._ema + (1.0 - self.smoothing) * chroma

        root, quality, confidence = match(self._ema, self._templates, self._labels)
        return ChordEstimate(
            root=root,
            quality=quality,
            confidence=confidence,
            rms=rms,
            chroma=self._ema.copy(),
            onsets=onsets,
        )

    def _detect_onsets(self, chroma: np.ndarray, rms: float) -> tuple[int, ...]:
        """
        Return pitch classes whose energy newly attacked on this raw chroma.

        Rising-edge detector on peak-normalised chroma: a pitch class fires when
        its energy rose by >= ``onset_rise`` since the previous frame *and* now
        exceeds ``onset_floor``, unless it's still within its
        ``onset_cooldown_frames`` window from a recent fire. Cooldowns tick down
        each frame; firing resets the note's cooldown so a held note produces a
        single attack, not one per frame. Operates on the raw (pre-EMA) chroma so
        attacks stay sharp.

        An **absolute** loudness gate (``rms >= onset_min_rms``, on the raw frame
        RMS) is applied first: below it no onset fires, so quiet notes never drop
        rain even when they're the loudest pitch class this frame. Frame state
        (previous chroma, cooldowns) still advances while gated, so the next loud
        attack diffs against the right baseline.

        At most **one** onset is returned per frame -- the pitch class with the
        largest rise among those that fired. A single struck note leaks energy
        into neighbouring bins (they'd all pass the threshold), so collapsing to
        the strongest keeps one attack = one event (one rain drop).
        """
        peak = float(chroma.max())
        norm = chroma / peak if peak > 0.0 else chroma

        self._cooldown = np.maximum(self._cooldown - 1, 0)

        # First frame after silence has no previous frame: diff against an
        # implicit zero baseline so a note rising straight out of silence still
        # registers as an attack (the common piano silence -> chord case).
        prev = self._prev_chroma if self._prev_chroma is not None else np.zeros(12)

        rise = norm - prev
        fired = (
            (rise >= self.onset_rise)
            & (norm >= self.onset_floor)
            & (self._cooldown == 0)
        )
        self._prev_chroma = norm

        # Absolute loudness gate: too quiet -> no drops, but state above has
        # already advanced so subsequent frames behave normally.
        if rms < self.onset_min_rms:
            return ()

        if not fired.any():
            return ()

        # Collapse to the single strongest attack this frame (largest rise among
        # fired bins), so leakage into neighbouring pitch classes doesn't spawn
        # extra events. Only that note's cooldown is reset.
        winner = int(np.argmax(np.where(fired, rise, -np.inf)))
        self._cooldown[winner] = self.onset_cooldown_frames
        return (winner,)
