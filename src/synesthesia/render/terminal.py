from __future__ import annotations

from typing import Optional

import numpy as np
from blessed import Terminal

from synesthesia.render.base import Renderer
from synesthesia.render.motion import Motion
from synesthesia.types import ColourState

_HALF_BLOCK = "\u2580"  # ▀


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Vectorized HSV -> 8-bit RGB for a whole grid.

    hsv : np.ndarray
        a height x width grid of hsv values;
            [
                [[0.0, 1.0, 1.0], [0.33, 1.0, 1.0]],   # row 0: two pixels
                [[0.66, 1.0, 1.0], [0.0, 0.0, 0.5]],   # row 1: two pixels
            ]
    """
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(np.int64) % 6

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# Precomputed "000".."255" decimal strings so we don't have to do str(int) every for every pixel
_DEC = np.array([f"{i:d}" for i in range(256)], dtype="U3")


def _ansi_frame(top: np.ndarray, bottom: np.ndarray, moves: list[str]) -> str:
    """
    Build the full frame string from top/bottom RGB grids, fully vectorized.

    Each cell becomes ``ESC[38;2;fr;fg;fb;48;2;br;bg;bbm▀`` (fg = top pixel,
    bg = bottom pixel). We assemble the per-cell strings with array-wide string
    concatenation (no Python per-pixel loop), reset SGR at each row end, and
    join rows with explicit cursor moves.

    ``top`` / ``bottom`` are ``(H, W, 3)`` uint8 arrays; ``moves[r]`` positions
    the cursor at the start of row ``r + 1``.
    """
    tr, tg, tb = (_DEC[top[..., k]] for k in range(3))
    br, bg, bb = (_DEC[bottom[..., k]] for k in range(3))

    # Per-cell escape: "\x1b[38;2;R;G;B;48;2;R;G;Bm▀"  (object-dtype string add).
    cells = (
        "\x1b[38;2;"
        + tr
        + ";"
        + tg
        + ";"
        + tb
        + ";48;2;"
        + br
        + ";"
        + bg
        + ";"
        + bb
        + "m"
        + _HALF_BLOCK
    )

    height = cells.shape[0]
    rows = ["".join(cells[r].tolist()) for r in range(height)]

    out: list[str] = []
    for r, row in enumerate(rows):
        out.append(row)
        out.append("\x1b[0m")
        if r + 1 < height:
            out.append(moves[r])
    return "".join(out)


class TerminalRenderer(Renderer):
    def __init__(self, motion: Motion, term: Optional[Terminal] = None) -> None:
        self.term = term if term is not None else Terminal()
        self.motion = motion
        self._grid_shape: Optional[tuple[int, int]] = None
        self._xs: np.ndarray = np.array([])
        self._ys: np.ndarray = np.array([])
        self._moves: Optional[list[str]] = None

    def start(self) -> None:
        t = self.term
        print(t.enter_fullscreen + t.hide_cursor + t.home + t.clear, end="", flush=True)

    def stop(self) -> None:
        t = self.term
        print(t.normal + t.normal_cursor + t.exit_fullscreen, end="", flush=True)

    def render(self, colour: ColourState, dt: float) -> None:
        self.motion.advance(dt)
        print(self._compose(colour), end="", flush=True)

    def _coords(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return cached normalised (xs, ys) coordinate grids of shape
        (2*height, width) -- one row per half-block pixel. Recomputed only when
        the terminal is resized, since the coordinate grids are otherwise fixed.
        """
        pixel_rows = height * 2
        if self._grid_shape != (pixel_rows, width):
            xs_axis = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
            ys_axis = np.linspace(
                0.0, 1.0, pixel_rows, endpoint=False, dtype=np.float32
            )
            self._ys, self._xs = np.meshgrid(ys_axis, xs_axis, indexing="ij")
            self._grid_shape = (pixel_rows, width)
        return self._xs, self._ys

    def _compose(self, base: ColourState) -> str:
        t = self.term
        width = t.width or 80
        height = t.height or 24

        xs, ys = self._coords(height, width)
        hsv = self.motion.sample(base, xs, ys)  # (2*height, width, 3)
        rgb = _hsv_to_rgb(hsv)  # (2*height, width, 3) uint8

        # Even pixel rows -> top (foreground); odd rows -> bottom (background).
        top = rgb[0::2]  # (height, width, 3)
        bottom = rgb[1::2]  # (height, width, 3)

        return t.home + _ansi_frame(top, bottom, self._row_move(height))

    def _row_move(self, height: int) -> list[str]:
        """
        Cached per-row cursor-move strings (one per row boundary). Recomputed
        only on resize; used to reposition each row explicitly (avoids the
        deferred auto-wrap "xn" glitch that drifts rows -> flicker).
        """
        if self._moves is None or len(self._moves) != height:
            self._moves = [self.term.move_xy(0, r + 1) for r in range(height)]
        return self._moves
