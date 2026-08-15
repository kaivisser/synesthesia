# Synesthesia — Chord-to-Colour Visualizer

An immersive audio-reactive visualizer that listens to piano chords and produces
"colours". Audio can come from a microphone, an audio file, or live system audio
(e.g. a music player). The default render target is the **terminal** (truecolor),
with a swappable renderer interface so a graphical backend (e.g. Pygame) can be
dropped in later.

> Status: **Design** — nothing built yet. This document is the working blueprint
> and will evolve as we lock down specifics.

> [!IMPORTANT]
> **Note to the agent:** You **may** implement code, but **only one small feature or
> chunk at a time** — never a whole file or class in one go. Work in the smallest
> reasonable units and stop after each:
> - the class definition (with its docstring/attributes) on its own,
> - then the constructor (`__init__`) on its own,
> - then **each individual method, one by one**.
>
> After writing each chunk, **explain what it does and why** before moving on, and
> wait for the human to review/approve before continuing to the next chunk. Do not
> batch multiple methods together. You may still edit this planning document and use
> illustrative snippets within it to communicate design ideas.

---

## 1. Goals & non-goals

### Goals
- Immersive, ambient **colour art** that reacts to chords (not an analytical tool).
- Three audio inputs, all funnelled through one processing path:
  - Microphone (live).
  - Audio file (`.wav`/`.flac`/`.mp3`), played back through the real-time path.
  - Live system audio (PulseAudio/PipeWire monitor on Linux).
- Chord detection via **chroma + template matching** (simple, real-time capable).
- **Terminal renderer** first (truecolor via `blessed`), swappable for graphics later.

### Non-goals (for now)
- High-accuracy chord recognition on dense, fully-mixed tracks.
- ML-based detection (kept as a possible future upgrade).
- Cross-platform system-audio capture (Linux-first for the monitor source).

---

## 2. Architecture: swappable-stage pipeline

```
AudioSource ──► Framer/Windower ──► ChordDetector ──► ColourMapper ──► Renderer
 (mic/file/       (overlapping         (chroma +          (chord →         (terminal /
  monitor)         Hann windows)        templates)         RGB + dyn.)      pygame)
   frames            chroma(12)          chord+conf         colour+motion
```

Each stage has a small, well-defined interface and data type, so stages are
independently testable and swappable. For live sources a thread-safe ring buffer
decouples the PortAudio capture callback (a C thread) from the analysis loop.

**Concurrency (v1): single-threaded MVP.** One loop runs analysis + render at the
hop rate (~22 fps @ hop 2048 / 44.1 kHz). The `AudioSource` interface is a blocking
`read(n)`, so the loop is a simple linear flow. **Planned upgrade (polish phase):**
move analysis to its own thread that publishes the latest `ColourState`, letting the
render loop animate motion at its own fps (~30–60) decoupled from chord updates. This
is a localised change — the `read(n)` source interface is unchanged either way.
_Rationale: async offers nothing here (PortAudio pushes via callback — no awaitable fd;
DSP is CPU-bound). Threads help only by (a) releasing the GIL inside NumPy/librosa C
code and, more importantly, (b) decoupling the fixed ~22 fps analysis clock from the
desired ~30–60 fps motion clock. Not worth the complexity until motion needs it._

### Data types (draft)
- `Frame`: `np.ndarray` float32 mono, fixed length, known sample rate.
- `Chroma`: 12-element float vector (pitch-class energy).
- `ChordEstimate`: `{ root: int(0-11), quality: enum, confidence: float, rms: float }`
  (plus a `NO_CHORD`/silence sentinel).
- `ColourState`: target colour(s) in HSV/Oklab + dynamics (intensity, motion).

---

## 3. Components

### 3.1 Audio sources (unified interface)

All three sources produce the **same thing**: a stream of fixed-size float32 **mono**
frames at a fixed **44.1 kHz** analysis rate. Each source has a different clock, so a
ring buffer (live sources) or output-stream pacing (file) sits in the middle.

#### Interface
```python
class AudioSource(ABC):
    sample_rate: int = 44100          # fixed analysis rate (all sources resample to this)

    @abstractmethod
    def start(self) -> None: ...      # begin producing

    @abstractmethod
    def read(self, n: int) -> np.ndarray | None:  # BLOCKING; see contract below
        ...

    @abstractmethod
    def stop(self) -> None: ...

    # Concrete, shared by ALL sources (no per-source boilerplate):
    def __enter__(self) -> "AudioSource":
        self.start()
        return self
    def __exit__(self, *exc) -> None:
        self.stop()
```
Defined as an **ABC** (not a `Protocol`): the three sources share the *exact* same
context-manager teardown, so `__enter__`/`__exit__` live once on the base and only
`start`/`read`/`stop` are abstract.

The consumer just loops `source.read(HOP)`. Blocking `read` is self-pacing: live
sources block until the hardware has produced enough samples; the file source blocks
on its audible output stream (which paces at real-time for free).

#### `read(n)` contract (precise semantics)
- **Return type/shape:** a **1-D `np.ndarray`, dtype `float32`, shape `(n,)`** (mono),
  or `None`. Never `(n, 1)`; downstream (`Frame`) assumes 1-D.
- **Blocking:** blocks until exactly `n` samples are available (live) or produced
  (file). Live sources therefore **never return a short array**.
- **End-of-stream (file):** returns **`None` as soon as a full `n` cannot be filled**.
  The trailing partial block (< `n` samples) is **dropped** — simplest for the
  fixed-hop framer, and inaudible/irrelevant for a visualizer. Live sources return
  `None` only after `stop()`.
- **After `stop()` / EOF:** `read` returns `None` (idempotent; safe to call again).
- **Value range:** float samples in roughly `[-1.0, 1.0]` (post-downmix mean).

#### Per-source processing order
`decode/capture → resample to 44.1 kHz → downmix to mono (mean of channels)
 → [ring buffer for live sources] → read(n)`

- **`MicSource`**: `sounddevice.InputStream` with a callback (runs on PortAudio's C
  thread) that resamples, downmixes, and pushes into a thread-safe **ring buffer**.
  `read(n)` blocks until `n` samples are available. Overflow policy: **drop oldest**
  (always analyse the freshest audio; stay real-time).
- **`MonitorSource`**: a `MicSource` pointed at the PulseAudio/PipeWire `.monitor`
  device → captures system audio from any player. Nearly free once mic works.
- **`FileSource`**: decode via `soundfile` (+ `librosa` resample). Acts as the single
  clock — each block is (a) written to a `sounddevice` **output** stream so the file
  is **audible in sync with the visuals**, and (b) returned from `read()` for analysis.
  Output playback paces at real-time, so no manual `sleep` throttle is needed. No ring
  buffer required (single-threaded, no separate producer). Returns `None` at EOF.

#### Decisions
- **Sample rate**: everything **resampled to a fixed 44.1 kHz** inside the source, so
  downstream math and chord-detection tuning assume one rate.
- **Channels**: **downmix to mono by averaging all channels** at the source (before the
  buffer); downstream only ever sees mono.
- **File audio**: **audible playback** through speakers, in sync with visuals.
- **Overflow (live)**: **drop oldest** samples when the consumer falls behind.

### 3.2 Framer / windowing
- Overlapping windows (default **4096 samples**, hop **2048**) with a **Hann** window.
- Input is already mono @ 44.1 kHz (resampling + downmix happen in the source, §3.1).
- Maintains a sliding window: each `read(hop)` advances by one hop; the detector sees
  the latest 4096-sample Hann-windowed frame.

### 3.3 Chord detector (chroma + templates)
- Chromagram via `librosa.feature.chroma_cqt` (CQT better than STFT for pitch).
- Templates: **12 major + 12 minor triads (v1)**. dim/aug/7ths planned for later.
- Cosine-similarity match → best chord + confidence.
- **Temporal smoothing**: EMA / majority vote over last N frames to avoid flicker.
  Tuned toward **smoothness** (see §3.4 / §6.4) — dreamy transitions over snappiness.
- **Silence gate**: RMS below threshold → `NO_CHORD` (fade to black).
- **Target input: clean solo piano** — detection tuned/validated for this case,
  not dense mixed tracks.

### 3.4 Colour mapper (the synesthesia core)
- **Root → hue**, with selectable schemes (default **Scriabin**):
  - *Scriabin* — his documented synesthetic mapping. **Default.**
  - *Chromatic* (C=0°, equal 30° spacing).
  - *Circle-of-fifths* (smoother transitions between related keys).
- **Single dominant colour** from the chord root (no per-note blending in v1).
- **Quality → saturation/value**: major = bright/saturated, minor = darker,
  dim = desaturated, aug = extra bright.
- **Loudness (RMS) → brightness/intensity**; **confidence → purity vs muddiness**.
- Smooth interpolation (LERP in HSV or **Oklab**) from current → target colour for
  pleasing transitions instead of hard cuts. **Bias toward smoothness**: longer
  fade time constants (~0.5–1s) for a dreamy feel.
- **Silence (`NO_CHORD`) → fade to black** (smooth fade using the same interpolation).

#### Scriabin colour mapping (reference)
| Note | Colour        | Note | Colour             |
|------|---------------|------|--------------------|
| C    | red           | F#   | bright blue        |
| G    | orange        | C#   | violet / purple    |
| D    | yellow        | G#   | purple / lilac     |
| A    | green         | D#   | steel / metallic   |
| E    | sky blue      | A#   | rose / steel       |
| B    | pale blue     | F    | deep red           |
_(Exact RGB values to be finalised in implementation; several sources vary.)_

### 3.5 Renderer (swappable backend)
- `Renderer` interface: `render(colour_state, dynamics)`.
- `TerminalRenderer` (`blessed`) for v1:
  - Full-screen colour field with selectable **motion style** (see below).
  - **Half-block (`▀`) pixels** for 2× vertical resolution; truecolor bg/fg.
  - One buffered write per frame; alternate screen buffer; hidden cursor.
  - Clean teardown on exit (restore terminal, show cursor).

#### Motion styles (`--motion` flag, default `perlin`)
The chord provides the **base colour**; motion only modulates brightness/hue
slightly, so all styles share one cheap model and swap behind a flag.
- `solid` — flat colour field (baseline/debug).
- `breathe` — slow sine brightness pulse (optionally driven by loudness). Calm.
- `plasma` — sine-interference plasma; smooth, hypnotic, cheap.
- `perlin` — drifting simplex noise (pure-Python `opensimplex`); organic. **Default.**
- `radial` — colour pulses outward from centre on chord changes.
- `gradient-drift` — slow-moving gradient across the screen; minimal, elegant.
- `aurora` — undulating vertical bands (northern-lights); pairs well with half-blocks.
- `ripple` — chord changes drop a "stone" that ripples across the field. Reactive.
- Later: `PygameRenderer` implementing the same interface (no upstream changes).

### 3.6 App orchestration & config
- Main loop wires stages together; handles clean shutdown (Ctrl-C restores term).
- Config via dataclass + CLI flags: input source, hue scheme, fps, window size,
  smoothing amount, chord vocabulary.

---

## 4. Dependencies

Managed with **`uv`** (PEP 621 `pyproject.toml` + committed `uv.lock` for reproducibility).

### Python packages
- `numpy` — arrays/DSP.
- `sounddevice` — mic + monitor capture, and file playback (output stream).
- `soundfile` / `librosa` — file decode + resample + chroma/CQT.
- `blessed` — terminal truecolor rendering.
- `opensimplex` — **pure-Python** simplex noise for the `perlin` motion style.

### System libraries (Linux)
- `libportaudio2` (for `sounddevice`) and `libsndfile1` (for `soundfile`) — install via
  the OS package manager; `uv` provides only the Python wheels.

### Common uv commands
```
uv add numpy sounddevice soundfile librosa blessed opensimplex   # deps + uv.lock
uv run python -m synesthesia --source mic --motion perlin        # run
uv run pytest                                                     # tests
uv run scripts/list_devices.py                                   # device helper
```

---

## 4b. Project structure

Package mirrors the pipeline stages (§2). Strict one-way dependency direction:
`__main__ → config → app → {audio, detect, colour, render} → types`. No stage imports
another stage; everything bottoms out at shared data types, keeping stages swappable.

```
synesthesia/
├── PLAN.md
├── README.md
├── pyproject.toml            # uv / PEP 621; deps + metadata
├── uv.lock                   # committed lockfile
│
├── src/synesthesia/          # the package (uv `--package` src/ layout)
│   ├── __init__.py
│   ├── __main__.py           # `python -m synesthesia` → parse CLI, hand to app.run()
│   ├── config.py             # Config dataclass + argparse (all CLI flags live here)
│   ├── types.py              # shared: Frame, Chroma, ChordEstimate, ColourState
│   │
│   ├── audio/                # §3.1 sources + §3.2 framing
│   │   ├── __init__.py
│   │   ├── base.py           # AudioSource protocol/ABC
│   │   ├── mic.py            # MicSource
│   │   ├── monitor.py        # MonitorSource (thin over mic)
│   │   ├── file.py           # FileSource (audible playback + analysis)
│   │   ├── ringbuffer.py     # thread-safe ring buffer (drop-oldest)
│   │   └── framer.py         # sliding Hann windower
│   │
│   ├── detect/               # §3.3 chord detection
│   │   ├── __init__.py
│   │   ├── chroma.py         # chromagram (librosa CQT wrapper)
│   │   ├── templates.py      # maj/min triad templates + matching
│   │   └── detector.py       # chroma → ChordEstimate (+ smoothing, silence gate)
│   │
│   ├── colour/               # §3.4 colour mapping
│   │   ├── __init__.py
│   │   ├── schemes.py        # Scriabin / chromatic / circle-of-fifths tables
│   │   ├── mapper.py         # ChordEstimate → ColourState
│   │   └── interp.py         # HSV/Oklab LERP + fade-to-black
│   │
│   ├── render/               # §3.5 renderer
│   │   ├── __init__.py
│   │   ├── base.py           # Renderer interface
│   │   ├── terminal.py       # TerminalRenderer (blessed, half-blocks)
│   │   └── motion.py         # motion styles
│   │
│   └── app.py                # build pipeline from Config; main loop
│
├── tests/
│   ├── test_ringbuffer.py
│   ├── test_framer.py
│   ├── test_templates.py
│   ├── test_schemes.py
│   └── fixtures/             # tiny single-chord WAVs for detector tests
│
└── scripts/
    └── list_devices.py       # print sounddevice devices / .monitor names
```

Notes:
- `__main__.py` stays thin (CLI → `Config` → `app.run(config)`); `app.py` owns wiring +
  loop so orchestration is testable without the CLI.
- `base.py` per subpackage holds the abstract interface; concrete impls sit alongside,
  making "add a source" / "drop in `PygameRenderer`" localised.

---

## 5. Build order
1. **Skeleton + interfaces**: data types, abstract classes, config.
2. **MicSource → RMS meter in terminal**: prove audio + render loop end-to-end.
3. **Chroma + chord detection**: print detected chord names; verify accuracy.
4. **Colour mapper**: chord → RGB.
5. **TerminalRenderer**: the pretty part — smoothing & motion.
6. **FileSource & MonitorSource**: near-free reuse.
7. **Polish**: presets, CLI flags, smoothing tuning.

---

## 6. Open questions / still to decide
_All major design questions resolved for v1. Remaining items are implementation
details to finalise during coding:_
1. **Canonical Scriabin RGB table**: pick a specific well-regarded 12-entry set and
   pin the exact hex values.
2. **`opensimplex` perf at render resolution**: confirm it's fast enough per frame;
   if not, cache/scale the noise grid.

---

## 7. Decisions log
- Render target: **terminal first** (`blessed`), graphics backend later. _(agreed)_
- Detection: **chroma + template matching** for v1. _(agreed)_
- Vibe: **immersive colour art**, not analytical. _(agreed)_
- Motion: configurable **`--motion`** flag; **default `perlin`** (menu of 8 styles). _(agreed)_
- Chord vocabulary: **maj/min only for v1**; 7ths/dim/aug planned later. _(agreed)_
- Input focus: **clean solo piano**. _(agreed)_
- Latency vs smoothness: **smoothness** (dreamy slow fades). _(agreed)_
- Colour scheme: **Scriabin** default (chromatic + circle-of-fifths as presets). _(agreed)_
- Voicing colour: **single dominant colour** (root-based), no per-note blend in v1. _(agreed)_
- Perlin noise: **pure-Python `opensimplex`**. _(agreed)_
- Scriabin RGB: use a **well-regarded canonical table**. _(agreed)_
- Silence behaviour: **fade to black**. _(agreed)_
- Config surface: **CLI flags only** for v1. _(agreed)_
- Analysis/render rates: **4096/2048 @ 44.1 kHz, ~30 FPS** terminal render. _(agreed)_
- Source interface: **blocking `read(n)`** (self-pacing, uniform across sources). _(agreed)_
- Source base type: **`ABC`** with shared concrete `__enter__`/`__exit__`; only
  `start`/`read`/`stop` abstract (no per-source teardown boilerplate). _(agreed)_
- `read(n)` return: **1-D `float32` `(n,)` mono**, or `None`. _(agreed)_
- EOF (file): **return `None` once a full `n` can't be filled** (drop trailing partial);
  live sources return `None` only after `stop()`; `read` idempotent post-EOF/stop. _(agreed)_
- Concurrency: **single-thread MVP**, split analysis/render into threads later if motion
  needs decoupling (async rejected: no awaitable audio fd, DSP is CPU-bound). _(agreed)_
- Overflow (live sources): **drop oldest** samples to stay real-time. _(agreed)_
- Sample rate: **resample every source to fixed 44.1 kHz**. _(agreed)_
- Channels: **downmix to mono by averaging** all channels at the source. _(agreed)_
- File playback: **audible** through speakers, in sync with visuals; file output stream
  acts as the real-time clock (no manual sleep throttle). _(agreed)_
- Tooling: **`uv`** for env/deps (`pyproject.toml` + committed `uv.lock`). _(agreed)_
- Layout: **subpackages per pipeline stage** (`audio/detect/colour/render`), shared
  `types.py`, thin `__main__.py` + `app.py` orchestration. _(agreed)_
- Entry point: **`python -m synesthesia …`** (run via `uv run`). _(agreed)_
