from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from typing import Optional

import numpy as np

from synesthesia.audio.base import AudioSource
from synesthesia.audio.ringbuffer import RingBuffer


def _run(cmd: list[str], check: bool = True) -> Optional[str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        if check:
            raise RuntimeError(f"failed to run {cmd[0]!r}: {exc}") from exc
        return None
    if proc.returncode != 0:
        if check:
            raise RuntimeError(
                f"{' '.join(cmd)} exited {proc.returncode}: {proc.stderr.strip()}"
            )
        return None
    return proc.stdout


def _default_sink() -> str:
    """
    Return the current default PipeWire audio sink node name.
    """
    try:
        out = subprocess.run(
            ["pw-metadata"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "could not run pw-metadata to find the default sink; "
            "pass a sink name explicitly (see `pw-link -o`)"
        ) from exc

    for line in out.splitlines():
        if "default.audio.sink" not in line:
            continue
        m = re.search(r"value:'(\{.*\})'", line)
        if not m:
            continue
        try:
            name = json.loads(m.group(1)).get("name")
        except json.JSONDecodeError:
            continue
        if name:
            return str(name)

    raise RuntimeError(
        "no default.audio.sink reported by pw-metadata; "
        "pass a sink name explicitly (see `pw-link -o`)"
    )


class MonitorSource(AudioSource):
    """
    Live system-audio capture via a PipeWire
    """

    # pw-record captures stereo; we downmix to mono ourselves.
    _CHANNELS = 2

    # Sample-frames to pull from the pipe per read; small = low latency.
    _READ_FRAMES = 1024

    def __init__(
        self,
        sink: Optional[str] = None,
        buffer_seconds: float = 1.0,
        gain: float = 1.0,
    ) -> None:
        self.sink = sink
        self.gain = float(gain)

        capacity = int(buffer_seconds * self.sample_rate)
        self._ring = RingBuffer(capacity=capacity)

        # Producer: pw-record subprocess + a thread draining its stdout.
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._proc is not None:
            return

        sink = self.sink or _default_sink()

        self._proc = subprocess.Popen(
            [
                "pw-record",
                "--target",
                "0",
                "--rate",
                str(self.sample_rate),
                "--channels",
                str(self._CHANNELS),
                "--format",
                "f32",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # Explicitly connect the sink's monitor ports onto our capture node.
        self._link_monitor(sink)

        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _link_monitor(self, sink: str) -> None:
        """
        Wire sink's monitor ports onto our pw-record capture node.
        """
        deadline = time.monotonic() + 2.0
        inputs: list[str] = []
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError("pw-record exited before its ports appeared")
            inputs = [
                line.split()[0]
                for line in _run(["pw-link", "-i"]).splitlines()
                if line.split() and line.split()[0].startswith("pw-record:input_")
            ]
            if inputs:
                break
            time.sleep(0.05)

        if not inputs:
            raise RuntimeError("pw-record capture ports never appeared")

        linked = 0
        for chan, port in zip(("FL", "FR"), sorted(inputs)):
            src = f"{sink}:monitor_{chan}"
            if _run(["pw-link", src, port], check=False) is not None:
                linked += 1

        if linked == 0:
            raise RuntimeError(
                f"could not link any monitor port of {sink!r}; "
                "check the sink name with `pw-link -o`"
            )

    def _read_loop(self) -> None:
        """
        Drain pw-record's stdout into the ring buffer until stop/EOF.
        """
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout

        bytes_per_frame = self._CHANNELS * 4  # f32 = 4 bytes
        chunk_bytes = self._READ_FRAMES * bytes_per_frame

        while not self._stop.is_set():
            raw = stdout.read(chunk_bytes)
            if not raw:
                break  # pw-record closed its pipe (EOF)

            # Trim any partial trailing frame before reshaping.
            usable = len(raw) - (len(raw) % bytes_per_frame)
            if usable == 0:
                continue

            stereo = np.frombuffer(raw, dtype=np.float32, count=usable // 4)
            stereo = stereo.reshape(-1, self._CHANNELS)
            mono = stereo.mean(axis=1).astype(np.float32, copy=False)
            if self.gain != 1.0:
                mono *= self.gain
            self._ring.write(mono)

        # Producer done: unblock any consumer waiting in read().
        self._ring.close()

    def read(self, n: int) -> Optional[np.ndarray]:
        return self._ring.read(n)

    def stop(self) -> None:
        # Signal the reader thread to stop pulling from the pipe.
        self._stop.set()

        # Terminate pw-record; kill if it ignores SIGTERM.
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None

        # Join the reader (it exits on EOF once the pipe closes).
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None

        # Idempotent; also unblocks any consumer if the thread hadn't yet.
        self._ring.close()
