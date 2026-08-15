from __future__ import annotations

from synesthesia.app import run
from synesthesia.config import parse_args


def main() -> None:
    """Entry point for ``python -m synesthesia`` (PLAN.md §4b): parse CLI -> run."""
    run(parse_args())


if __name__ == "__main__":
    main()
