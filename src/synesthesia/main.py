import numpy as np

from synesthesia.audio.mic import MicSource


def test_microphone():
    with MicSource(gain=8) as src:
        frame = src.read(2048)
        while frame is not None:
            rms = float(np.sqrt(np.mean(frame**2)))
            bar = "#" * int(rms * 200)
            print(f"\r{rms:.4f} {bar:<50}", end="", flush=True)
            frame = src.read(2048)


if __name__ == "__main__":
    test_microphone()
