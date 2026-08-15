from __future__ import annotations

import colorsys

from blessed import Terminal

from synesthesia.render.base import Renderer
from synesthesia.render.motion import Motion
from synesthesia.types import ColourState

# Half-block glyph: its foreground paints the TOP pixel, the cell background the
# BOTTOM pixel, so one character row shows two vertically-stacked pixels (§3.5).
_HALF_BLOCK = "\u2580"  # ▀


class TerminalRenderer(Renderer):
    """
    Truecolor terminal backend (PLAN.md §3.5) -- the v1 renderer.

    Draws a full-screen colour field using **half-block** glyphs (``▀``): each
    character cell stacks two vertical "pixels" by colouring the glyph's
    foreground (top pixel) and the cell background (bottom pixel), doubling
    vertical resolution for free. Every cell's colour comes from the active
    ``Motion`` style sampling around the frame's base ``ColourState``.

    Rendering discipline (per §3.5):

    * **One buffered write per frame** -- the whole screen is composed into a
      single string and emitted in one ``print`` to avoid tearing/flicker.
    * **Alternate screen buffer + hidden cursor** on ``start``; both restored on
      ``stop`` so the user's shell is left exactly as it was, even on Ctrl-C.

    Attributes
    ----------
    term:
        The ``blessed.Terminal`` used for capability strings (truecolor codes,
        cursor moves, screen setup/teardown).
    motion:
        The active motion style; sampled once per half-block pixel per frame.
    _clock:
        Accumulated elapsed seconds handed to ``motion.advance`` each frame.
    """

    def __init__(self, motion: Motion, term: Terminal | None = None) -> None:
        self.term = term if term is not None else Terminal()
        self.motion = motion
        self._clock = 0.0

    def start(self) -> None:
        """
        Enter the alternate screen buffer and hide the cursor (§3.5).

        We emit the capability strings directly rather than using blessed's
        ``fullscreen()``/``hidden_cursor()`` context managers, because the
        render loop's setup/teardown is driven by ``Renderer.__enter__`` /
        ``__exit__`` (and ``stop`` below), not a ``with`` block here. Using the
        alternate screen means the user's scrollback is untouched: on ``stop``
        we pop back to exactly what was on screen before.
        """
        # Raw ANSI (not blessed capabilities, which can be empty when terminfo
        # isn't resolved): enter alt-screen (1049h), hide cursor (25l),
        # home (H), clear screen (2J).
        print("\x1b[?1049h\x1b[?25l\x1b[H\x1b[2J", end="", flush=True)

    def stop(self) -> None:
        """
        Restore the terminal: show the cursor and leave the alternate screen
        (§3.5). The exact inverse of ``start`` and safe to call on any exit path
        (normal or Ctrl-C), leaving the user's shell exactly as it was.
        """
        # Inverse of start: reset SGR, show cursor (25h), leave alt-screen (1049l).
        print("\x1b[0m\x1b[?25h\x1b[?1049l", end="", flush=True)

    def render(
        self,
        colour: ColourState,
        dt: float,
        overlay: list[str] | None = None,
    ) -> None:
        """
        Draw one frame for ``colour`` (§3.5).

        Advances the motion clock by ``dt``, then composes the entire screen --
        every half-block cell sampled from the motion style around ``colour`` --
        into a single string and emits it in **one buffered write** to avoid
        flicker/tearing. If ``overlay`` is given, those text lines are drawn on
        top of the field (top-left) for debug/HUD use.
        """
        self._clock += dt
        self.motion.advance(dt)
        frame = self._compose(colour)
        if overlay:
            frame += self._overlay(overlay)
        print(frame, end="", flush=True)

    def _compose(self, base: ColourState) -> str:
        """
        Build the full-screen frame string for ``base``.

        The terminal is ``width`` columns by ``height`` rows; half-blocks give
        ``2 * height`` pixel rows. For each character cell we sample the motion
        style at the top pixel (foreground) and the bottom pixel (background),
        normalising coordinates to [0, 1) so a style is resolution-independent.
        """
        t = self.term
        width = t.width or 80
        height = t.height or 24
        pixel_rows = height * 2

        # Precompute inverse spans once (avoid per-cell division); guard width/
        # height of 1 so we never divide by zero.
        inv_w = 1.0 / width if width > 1 else 0.0
        inv_h = 1.0 / pixel_rows if pixel_rows > 1 else 0.0

        # Home via a raw CUP escape (row 1, col 1) rather than a blessed
        # capability: terminfo lookups can come back empty in some environments,
        # leaving rows unpositioned and making the field fight the overlay
        # (screen-wide flicker). Raw ANSI is deterministic and universal.
        parts: list[str] = ["\x1b[H"]
        for row in range(height):
            top_py = row * 2
            bottom_py = top_py + 1
            for col in range(width):
                x = col * inv_w
                top = self.motion.sample(base, x, top_py * inv_h)
                bottom = self.motion.sample(base, x, bottom_py * inv_h)
                tr, tg, tb = _to_rgb(top)
                br, bg, bb = _to_rgb(bottom)
                # fg = top pixel, bg = bottom pixel; ▀ fills the top half.
                parts.append(f"\x1b[38;2;{tr};{tg};{tb};48;2;{br};{bg};{bb}m{_HALF_BLOCK}")
            parts.append("\x1b[0m")
            # Explicit next-row move (CUP) instead of relying on auto-wrap, which
            # terminals defer after the last column ("xn" glitch) -> row drift.
            # rows/cols are 1-based in ANSI; skip the final trailing move.
            if row + 1 < height:
                parts.append(f"\x1b[{row + 2};1H")
        return "".join(parts)

    def _overlay(self, lines: list[str]) -> str:
        """
        Compose ``lines`` as a HUD in the top-left, over the colour field.

        Each line is drawn as bright white text on a solid black background so it
        stays legible regardless of the field colour behind it. Positioned with
        absolute cursor moves and appended *after* the field so it paints last.
        """
        t = self.term
        width = t.width or 80
        box = min(width, 40)
        parts: list[str] = []
        for i, line in enumerate(lines):
            # Pad to a fixed box width so old, longer text is fully overwritten.
            text = f" {line} ".ljust(box)
            # Raw CUP escape (1-based) to the top-left; deterministic regardless
            # of terminfo, so it lands exactly on the field cells it overwrites.
            parts.append(f"\x1b[{i + 1};1H")
            parts.append(f"\x1b[1;38;2;255;255;255;48;2;0;0;0m{text}\x1b[0m")
        return "".join(parts)


def _to_rgb(c: ColourState) -> tuple[int, int, int]:
    """HSV ``ColourState`` -> 8-bit RGB triple (mirrors ColourState.as_rgb)."""
    r, g, b = colorsys.hsv_to_rgb(c.hue, c.saturation, c.value)
    return round(r * 255), round(g * 255), round(b * 255)
