from __future__ import annotations

import threading

import numpy as np

ANALYSIS_SR = 44100


class RingBuffer:
    def __init__(self, capacity: int = ANALYSIS_SR) -> None:
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._start = 0
        self._count = 0
        self._closed = False
        self._dropped = 0
        self._lock = threading.Condition()

    def write(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = samples.size

        with self._lock:
            if self._closed:
                return

            # If incoming chunk is bigger than the ring, keep only the tail.
            if n >= self._capacity:
                self._buf[:] = samples[-self._capacity :]
                self._dropped += self._count + n - self._capacity
                self._start = 0
                self._count = self._capacity
                self._lock.notify()
                return

            # Drop oldest if this push would overflow.
            overflow = self._count + n - self._capacity
            if overflow > 0:
                self._start = (self._start + overflow) % self._capacity
                self._count -= overflow
                self._dropped += overflow

            # Write n samples starting at write_pos, handling wrap-around.
            write_pos = (self._start + self._count) % self._capacity
            first = min(n, self._capacity - write_pos)
            self._buf[write_pos : write_pos + first] = samples[:first]
            if first < n:
                self._buf[0 : n - first] = samples[first:]

            self._count += n
            self._lock.notify()

    def read(self, n: int) -> np.ndarray | None:
        with self._lock:
            # Wait until we have n samples, or we're closed.
            while self._count < n and not self._closed:
                self._lock.wait()
            if self._count < n:
                # closed AND not enough left -> end of stream (drop partial)
                return None
            out = np.empty(n, dtype=np.float32)
            first = min(n, self._capacity - self._start)
            out[:first] = self._buf[self._start : self._start + first]
            if first < n:
                out[first:] = self._buf[0 : n - first]
            self._start = (self._start + n) % self._capacity
            self._count -= n
            return out

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._lock.notify_all()

    @property
    def dropped(self) -> int:
        """Total samples discarded to overflow since creation (drop-oldest)."""
        with self._lock:
            return self._dropped

    def __len__(self) -> int:
        with self._lock:
            return self._count
