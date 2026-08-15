from __future__ import annotations

from typing import Optional

import numpy as np

from synesthesia.audio.base import AudioSource

WINDOW_SIZE = 4096
HOP_SIZE = 2048


class Framer:
    """
    Sliding Hann-windowed framer over an AudioSource (PLAN.md §3.2).

    ------------------------------------------------------------------------
    PRIMER (for the audio-curious): samples, windows, hops, and Hann
    ------------------------------------------------------------------------
    Audio is just a stream of numbers. At 44.1 kHz, one second of sound is
    44,100 float "samples", each measuring the speaker/mic displacement at that
    instant. A piano note is a repeating wave: middle C (C4) repeats ~261.6
    times per second (261.6 Hz), G4 ~392 Hz. When two notes sound together their
    waves simply ADD, sample by sample -- one number per sample, but that number
    is the sum of both vibrations.

    The chord detector can't chew on an endless stream; its frequency transform
    needs a fixed-size chunk. That's what this class produces. Three ideas:

    * WINDOW (``window_size``, default 4096 ~= 93 ms) -- one chunk of audio
      handed to the detector. Size is a trade-off: a BIGGER window measures
      pitch more accurately (it "sees" more full cycles of each note), while a
      SMALLER window reacts faster but blurs pitch. You can't maximise both at
      once (the time/frequency uncertainty trade-off). 4096 favours pitch
      accuracy, which is what chord detection needs.

    * HOP (``hop``, default 2048) -- how far we slide before taking the next
      window. Windows OVERLAP by ``window_size - hop`` (here 50%), so each new
      frame reuses the previous frame's second half and adds one fresh hop. This
      decouples the two knobs: a big (accurate) window AND frequent updates. Hop
      2048 @ 44.1 kHz -> a new frame every ~46 ms -> ~22 frames/sec.

    * HANN window -- the frequency transform secretly assumes each chunk loops
      forever end-to-end. Real audio rarely starts and ends at the same value,
      so that imagined loop has a sudden "seam" jump, and sharp jumps inject
      fake frequencies that smear across the spectrum ("spectral leakage").
      Multiplying the chunk by a smooth bell curve that is 0 at both ends and 1
      in the middle (the Hann window) fades the edges out, removing the seam and
      cleaning up the spectrum. The 50% overlap compensates for the attenuated
      edges: a sample dimmed at one frame's edge sits at full weight in the
      centre of the next.

    Worked example -- play C, then 1 s later add G on top:
      * t=0..1s: every ~46 ms we emit a ~93 ms window containing only the C
        wave (~24 cycles of C inside it) -> downstream chroma lights the "C" bin.
      * ~t=1s: one or two overlapping windows STRADDLE the moment G enters
        (C-only start, C+G tail) -- briefly ambiguous, which is exactly why the
        detector later smooths over several frames.
      * t>1s: each window now holds the summed C+G wave -> chroma lights both
        "C" and "G". C+G is a perfect fifth (a C chord's skeleton, but with no
        third yet), so the detector leans toward C with moderate confidence; the
        root C is what drives the colour.

    ------------------------------------------------------------------------
    WHAT THIS CLASS DOES
    ------------------------------------------------------------------------
    Maintains a persistent window of ``window_size`` samples. Each call to
    ``frames()`` (or ``read()``) pulls one ``hop`` of fresh samples from the
    source, shifts them in from the right, and yields the full window multiplied
    by a Hann window. Consecutive frames therefore overlap by
    ``window_size - hop`` samples.

    The source contract (blocking ``read(n)`` returning 1-D float32 or ``None``
    at end of stream) is preserved: this framer returns ``None`` once the source
    can no longer supply a full hop.
    """

    def __init__(
        self,
        source: AudioSource,
        window_size: int = WINDOW_SIZE,
        hop: int = HOP_SIZE,
    ) -> None:
        if hop <= 0 or window_size <= 0:
            raise ValueError("window_size and hop must be positive")
        if hop > window_size:
            raise ValueError("hop must not exceed window_size")

        self.source = source
        self.window_size = window_size
        self.hop = hop

        # Persistent window; older samples on the left, newest on the right.
        self._window = np.zeros(window_size, dtype=np.float32)
        self._hann = np.hanning(window_size).astype(np.float32)

        # Number of hops needed before the window is first fully populated.
        self._priming = window_size // hop
        self._exhausted = False

    def read(self) -> Optional[np.ndarray]:
        """
        Advance by one hop and return the latest Hann-windowed frame.

        """
        if self._exhausted:
            return None

        needed = self._priming if self._priming > 0 else 1
        for _ in range(needed):
            chunk = self.source.read(self.hop)
            if chunk is None:
                self._exhausted = True
                return None
            self._push(chunk)
        self._priming = 0

        return self._window * self._hann

    def _push(self, chunk: np.ndarray) -> None:
        self._window[: -self.hop] = self._window[self.hop :]
        self._window[-self.hop :] = chunk

    def frames(self):
        """Iterate windowed frames until the source is exhausted."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
