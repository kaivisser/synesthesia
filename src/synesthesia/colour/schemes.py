from __future__ import annotations

import colorsys

# see https://en.wikipedia.org/wiki/Clavier_%C3%A0_lumi%C3%A8res
SCRIABIN_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),  # C   red
    (255, 20, 147),  # C#  violet / deep pink
    (255, 255, 0),  # D   yellow
    (123, 104, 238),  # D#  steel / metallic (medium slate blue)
    (144, 213, 255),  # E   sky blue / pearly
    (178, 34, 34),  # F   deep red
    (0, 100, 255),  # F#  bright blue
    (255, 140, 0),  # G   orange
    (170, 100, 255),  # G#  purple / lilac
    (0, 200, 60),  # A   green
    (120, 160, 180),  # A#  rose / steel
    (150, 210, 235),  # B   pale blue
)


def _rgb_to_hue(rgb: tuple[int, int, int]) -> float:
    r, g, b = (c / 255.0 for c in rgb)
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return h


SCRIABIN_HUES: tuple[float, ...] = tuple(_rgb_to_hue(c) for c in SCRIABIN_RGB)
CHROMATIC_HUES: tuple[float, ...] = tuple(i / 12.0 for i in range(12))
CIRCLE_OF_FIFTHS_HUES: tuple[float, ...] = tuple(
    ((r * 7) % 12) / 12.0 for r in range(12)
)

SCHEMES: dict[str, tuple[float, ...]] = {
    "scriabin": SCRIABIN_HUES,
    "chromatic": CHROMATIC_HUES,
    "fifths": CIRCLE_OF_FIFTHS_HUES,
}
