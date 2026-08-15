from __future__ import annotations

from abc import ABC, abstractmethod

from synesthesia.types import ColourState


class Motion(ABC):
    """
    Abstract base for the motion styles selected by --motion

    The chord provides the **base colour**; motion only *modulates* it slightly
    over time and space so the field feels alive rather than flat. Every style
    shares this one cheap model and swaps behind the CLI flag.

    A motion is a pure, per-cell function of *where* (normalised screen
    coordinates) and *when* (an accumulated clock), returning the colour for that
    cell this frame. Keeping it per-cell (rather than whole-frame) lets the
    renderer own the grid geometry and half-block packing, while each style only
    decides how a single point should shimmer around the base ``ColourState``.

    Concrete styles (``solid``, ``breathe``, ``plasma``, ``perlin`` …) implement
    ``sample`` and, if they need to advance internal state, ``advance``.
    """

    def advance(self, dt: float) -> None:
        """
        Advance the style's internal clock by ``dt`` seconds. Called once per
        frame before sampling. Static styles (e.g. ``solid``) can ignore this;
        the default is a no-op so trivial styles need no boilerplate.
        """

    @abstractmethod
    def sample(self, base: ColourState, x: float, y: float) -> ColourState:
        """
        Return the colour for one cell at normalised coordinates ``x``, ``y``
        (both in [0, 1)), given the frame's ``base`` colour. Called once per grid
        cell per frame, so implementations must stay cheap.
        """
        pass


class Solid(Motion):
    """
    Flat colour field: every cell shows the base colour unchanged (PLAN.md §3.5,
    "solid — baseline/debug"). No time or space dependence, so it inherits the
    no-op ``advance`` and is the cheapest possible style — handy for verifying the
    render/colour pipeline in isolation before layering motion on top.
    """

    def sample(self, base: ColourState, x: float, y: float) -> ColourState:
        return base


# Name -> style registry. The `--motion` flag (config.MOTIONS) is validated at
# the CLI, but this is the single source of truth for which names are actually
# *implemented*; the remaining §3.5 styles slot in here as they're built.
_MOTIONS: dict[str, type[Motion]] = {
    "solid": Solid,
}


def make_motion(name: str) -> Motion:
    """
    Construct the motion style registered under ``name``.

    Raises ``ValueError`` (listing what *is* implemented) if the name is unknown
    -- this covers valid ``--motion`` choices that simply aren't built yet, so
    the failure is explicit rather than a silent fall-through to ``solid``.
    """
    try:
        cls = _MOTIONS[name]
    except KeyError:
        valid = ", ".join(sorted(_MOTIONS))
        raise ValueError(
            f"motion style {name!r} not implemented; available: {valid}"
        ) from None
    return cls()
