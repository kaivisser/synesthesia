from __future__ import annotations

from synesthesia.app import run
from synesthesia.config import parse_args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
