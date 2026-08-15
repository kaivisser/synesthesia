from __future__ import annotations

import sounddevice as sd


def main() -> None:
    print(sd.query_devices())
    print()
    default_in, default_out = sd.default.device
    print(f"default input index:  {default_in}")
    print(f"default output index: {default_out}")


if __name__ == "__main__":
    main()
