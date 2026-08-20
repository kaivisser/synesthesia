## Data flow

# Analysis thread
```
AudioSource ──────────► Framer ──────────► Detector ──────────► Mapper ──────────► LatestTarget / OnsetQueue
            float32[n]         float32[4096]        ChordEstimate      ColourState 
            mono samples       window × Hann        root/qual/conf     target HSV  
            None at EOF        None at EOF          +chroma[12]        +onset (pc,hue)
                                                    +onsets (pc)
```

# Consumer thread
```
LatestTarget / OnsetQueue ──────────► Interpolator ──────────► Renderer──────────► Terminal
                          ColourState              ANSI truecolor      print
                          smoothed HSV             pixels              
                          +onset (pc,hue)                         
  
```
---

## Stage 1 — Audio capture (audio/mic.py)

- PortAudio (C/C++) invokes _callback and feeds us `float32[n]`
- Data is written to a ring buffer of `float32[44100]`

---

## Stage 2 — capture (audio/framer.py)

- Framer consumes fixed chunks of default size 2048
- 2048 samples/frame ÷ 44100 samples/sec ≈ 46 ms/frame ≈ 21.5 frames/sec
- Chunks are inserted into a sliding window of size 4096
- For every window we apply a Hann filter

---

## Stage 3 - Chroma detection (detect/detector.py)

- For each frame we compute the chroma (CQT) and onsets (impulse)
- We take the dot product of our chroma and our chord templates to calculate the closest matching chord

---

## Stage 4 - Colour mapper (colour/mapper.py)

- For any given chord, we compute the corresponding hue, saturation, value, intensity
- Chord root determines background colour
- Chord type (major, minor) determines saturation and brightness multipliers
- Mapper confidence also affects saturation
- Loudness also affects brightness and motion
- Onsets are mapped individually to the respective hue of the colour scheme

---

## Stage 5 - Interpolator (colour/interpolator.py)

- Since we don't want to instantly snap between colours, we smoothly interpolate between incoming colours

---

## Stage 6 - Renderer (render/terminal.py)

- 

---
## Concurrency model

**Files:** [`app.py`](src/synesthesia/app.py),
[`app_state.py`](src/synesthesia/app_state.py)

The headline architectural feature: **two threads, three independent clocks**,
joined by two hand-offs chosen for their data's semantics.

**Three clocks** (each decoupled, so nothing waits on a slower stage):

| Clock | Rate | Driven by |
|---|---|---|
| Capture | ~44.1k samples/s, irregular blocks | hardware / PipeWire |
| Analysis (`read(hop)` → detect) | ~21.5 Hz (44100 / 2048) | hop size |
| Render (motion) | ~60 Hz (`--fps`) | desired smoothness |

**The split:**
```
┌ analysis thread (~22 fps) ┐      ┌ render/main thread (~60 fps) ┐
 read → framer → detect → map ─┐ ┌─► interpolate → render
                               │ │
                  [ LatestTarget: newest ColourState (HSV, onset-free) ]  ← retained state
                  [ OnsetQueue:  pending (pc, hue) note attacks        ]  ← events
```
- Analysis publishes a *target* colour; it does **not** interpolate.
- Render **interpolates toward the target at 60 fps**, so fades stay smooth
  regardless of the ~22 Hz chord clock. Runs on the **main thread** (terminal
  setup/teardown + Ctrl-C must live there).

**Why threads, not async:** no awaitable audio fd (PortAudio pushes via
callback) and DSP is CPU-bound. Threads win by (a) NumPy/librosa releasing the
GIL, and mainly (b) decoupling the analysis and render clocks.

**Two hand-offs — the key insight:**
- `LatestTarget` = **retained state → latest-wins** (a lock-protected slot;
  newest colour is all that matters, older overwritten).
- `OnsetQueue` = **events → drain-once** (note attacks must fire exactly once).

> **Bug this caught:** onsets originally rode on the `LatestTarget` slot; the
> ~60 Hz render loop re-read it ~3× per ~22 Hz analysis frame, so one key press
> → **~3 rain drops**. Fix: give events their own drain-once queue. One attack =
> one drop, at any fps ratio. **Principle: retained state wants latest-wins;
> events want a queue.**

**Clean shutdown** (`run()` supervises; covers file EOF / Ctrl-C / live source):
`finished.set()` → `source.stop()` (closes ring buffer, unblocks a parked
`read()`) → `analysis.join(timeout)`.

---

## Stage 2 — Framing / windowing

**File:** [`audio/framer.py`](src/synesthesia/audio/framer.py)

**In:** `source.read(hop) → float32[2048]` per step &nbsp;•&nbsp;
**Out:** `read() → float32[4096]` (one window × Hann; `None` at EOF)

Turns the raw sample stream into fixed-size, overlapping, Hann-windowed frames
for the detector. Maintains a persistent sliding window.

**Sliding window (toy example: window = 4, hop = 2; real config is 4096 / 2048):**

```
stream:  s0 s1 s2 s3 s4 s5 s6 s7 ...

read #1  [ s0 s1 s2 s3 ]                 (primes with window/hop = 2 hops)
read #2        [ s2 s3 s4 s5 ]           (drop s0 s1, append s4 s5)
read #3              [ s4 s5 s6 s7 ]     (drop s2 s3, append s6 s7)
                └reused┘└─new─┘
```

Each frame is `window` wide; each step advances by `hop`; the right half of one
frame becomes the left half of the next (the 50% overlap).

- **Window 4096 (~93 ms)** = pitch accuracy; **hop 2048 (50% overlap)** =
  frequent updates. Decoupling the two gives *both* accuracy and responsiveness.
- **Hann window** kills spectral leakage (smooths the chunk's edges so the FFT's
  implied loop has no seam); the 50% overlap compensates for the dimmed edges.
- Each `read()` pulls one hop, slides it in, returns `window × hann`. First call
  primes with `window/hop = 2` hops. `None` from the source propagates as EOF.
- **Defines the analysis clock:** one frame per hop = 44100 / 2048 ≈ **21.5 Hz**.

---

## Stage 3 — Chord detection

**Files:** [`detect/chroma.py`](src/synesthesia/detect/chroma.py),
[`detect/templates.py`](src/synesthesia/detect/templates.py),
[`detect/detector.py`](src/synesthesia/detect/detector.py)

**In:** `float32[4096]` Hann frame &nbsp;•&nbsp;
**Out:** `ChordEstimate` = `{root, quality, confidence, rms, chroma[12], onsets:(pc,…)}`
(or `NO_CHORD` on silence)

`ChordDetector.update()` runs the pipeline:

1. **RMS gate** — cheap loudness check first; below `silence_rms` short-circuits
   to `NO_CHORD` (skip the CQT on silence). Silence *decays* the smoothed chroma
   rather than resetting it, so sound after a pause fades back in, not snaps.
2. **Frame → 12-bin chroma** — a chromagram folds the spectrum onto the 12 pitch
   classes (C..B), summing across octaves: a C anywhere lights the same bin,
   which is all chord ID needs. Uses **CQT, not STFT** — log-spaced bins match
   equal temperament so pitches land cleanly instead of smearing across the
   linear FFT grid. Left unnormalised (matcher is scale-invariant).
3. **Onset detection** — rising-edge detector on the **raw** chroma (pre-smooth,
   so attacks stay crisp): a pitch class fires when its energy jumps
   `≥ onset_rise` since last frame *and* clears `onset_floor`. A per-note
   **cooldown** makes a held note strike once; collapse-to-strongest keeps
   spectral leakage from spawning extras → **one attack = one drop**.
4. **EMA smoothing** — exponential moving average over chroma vectors filters
   one-frame flickers (`…C C Am C…`), so we render *sustained* chords.
5. **Template match** — 24 binary triads (12 maj + 12 min); **cosine similarity**
   against all at once (one matmul), take the max. Cosine normalises out loudness
   so quiet and loud C-maj score the same.

**Design theme — two time constants:** onsets read the *raw* chroma
(responsiveness); chord identity reads the *smoothed* EMA (stability). That split
mirrors the concurrency layer downstream (drain-once events vs. latest-wins
state).

---

## Stage 4 — Colour mapping

**Files:** [`colour/schemes.py`](src/synesthesia/colour/schemes.py),
[`colour/mapper.py`](src/synesthesia/colour/mapper.py),
[`colour/interpolator.py`](src/synesthesia/colour/interpolator.py)

**In:** `ChordEstimate` &nbsp;•&nbsp;
**Out:** `ColourState` = `{hue, saturation, value, intensity, onsets:(pc,hue)}`

Two sub-parts, deliberately split: a **pure mapper** (*what* colour is this
chord?) and a **stateful interpolator** (*how fast* do we move toward it?).

**Mapper** (`ChordEstimate → ColourState`, pure/stateless) — one perceptual axis
per musical property:

| Input | → Output | Why |
|---|---|---|
| root (0-11) | **hue** (scheme table) | the note *is* the colour |
| quality maj/min | sat/value **bias** | major brighter/purer, minor darker |
| RMS (loudness) | **value + intensity** | louder = brighter + more motion |
| confidence | **purity** (`0.4 + 0.6·conf`) | shaky match → grey, never a bold *wrong* colour |
| `NO_CHORD` | `BLACK` | silence short-circuits |

- Onset hue = **the struck note's own hue**, not the chord root (a B in a C chord
  drops a B-coloured drop). Pass-through event data, not smoothed.
- **Schemes** (swappable): `scriabin` (composer's synesthetic map, default),
  `chromatic` (even 12-step wheel), `fifths` (harmonically-close = colour-close).

**Interpolator** (`ColourState → ColourState`, stateful; runs on render thread @
60 fps) — where "smoothness over accuracy" is enforced:

- **Framerate-independent** exponential smoothing: `alpha = 1 - exp(-dt/tau)`, so
  fade speed is identical at 30 or 120 fps (`tau` is real seconds, not per-frame).
- **Hue lerps the short way** round the wheel (C→B goes near, not ¾ around).
- **Onsets bypass smoothing**: `self.current` is kept onset-free (a drop never
  re-fires); the target's onsets attach only to the returned value. Same
  retained-vs-event split as the concurrency layer.

---

## Stage 5 — Rendering & motion

**Files:** [`render/base.py`](src/synesthesia/render/base.py),
[`render/terminal.py`](src/synesthesia/render/terminal.py),
[`render/motion.py`](src/synesthesia/render/motion.py)

**In:** `render(colour: ColourState, dt) ` (smoothed HSV + this frame's onsets)
&nbsp;•&nbsp; **Out:** a grid of ANSI truecolor pixels written to stdout

*Walkthrough in progress.*

---

## Testing

```bash
uv run pytest
```

Highlights:
- `test_ringbuffer.py` — FIFO correctness, wraparound, drop-oldest, close/EOF
  semantics, and a threaded producer/consumer stress test.
- `test_framer.py` — window priming, Hann application, hop overlap, EOF.
- `test_app_state.py` — the two concurrency hand-offs: `LatestTarget`
  (latest-wins) and `OnsetQueue` (drain-once), including a regression test for
  the "one attack = one drop" bug and a threaded no-loss stress test.
- `test_app.py` — the analysis worker, onset stripping, and clean thread
  shutdown.

---

## Project layout

```
src/synesthesia/
├── __main__.py     # python -m synesthesia → parse CLI, hand to app.run()
├── config.py       # Config dataclass + argparse (all CLI flags)
├── types.py        # shared data types: Frame, Chroma, ChordEstimate, ColourState
├── app.py          # builds the pipeline from Config; supervises the two threads
├── app_state.py    # LatestTarget + OnsetQueue (analysis→render hand-offs)
├── audio/          # sources (mic/monitor/file) + ring buffer + framer
├── detect/         # chroma + template matching → ChordEstimate
├── colour/         # schemes + chord→colour mapper + interpolator
└── render/         # renderer interface + terminal backend + motion styles
```
