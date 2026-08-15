import threading

import numpy as np

from synesthesia.audio.ringbuffer import RingBuffer


def test_basic_fifo():
    rb = RingBuffer(capacity=16)
    samples = np.array([1, 2, 3, 4, 5])
    rb.write(samples)
    output_1 = rb.read(2)
    output_2 = rb.read(2)
    assert output_1 is not None
    assert output_2 is not None
    assert np.array_equal(output_1, np.array([1, 2]))
    assert np.array_equal(output_2, np.array([3, 4]))


def test_read_multiple_writes():
    rb = RingBuffer(capacity=16)

    rb.write(np.array([1, 2]))
    rb.write(np.array([3, 4, 5]))

    output = rb.read(4)

    assert output is not None
    assert np.array_equal(output, np.array([1, 2, 3, 4]))
    assert len(rb) == 1


def test_wraparound():
    rb = RingBuffer(capacity=8)
    rb.write(np.arange(6))

    output = rb.read(4)

    assert output is not None
    assert np.array_equal(output, np.array([0, 1, 2, 3]))

    rb.write(np.arange(6, 12))

    output = rb.read(8)

    assert output is not None
    assert np.array_equal(output, np.arange(4, 12))


def test_drop_oldest():
    rb = RingBuffer(capacity=8)
    rb.write(np.arange(12))

    assert rb.dropped == 4
    assert len(rb) == 8

    output = rb.read(8)

    assert output is not None
    assert np.array_equal(output, np.arange(4, 12))


def test_drop_oldest_incremental():
    rb = RingBuffer(capacity=8)
    rb.write(np.arange(6))
    rb.write(np.arange(6, 12))

    output = rb.read(8)

    assert rb.dropped == 4
    assert output is not None
    assert np.array_equal(output, np.arange(4, 12))


def test_write_larger_than_capacity():
    rb = RingBuffer(capacity=8)
    rb.write(np.arange(20))

    assert len(rb) == 8

    output = rb.read(8)

    assert output is not None
    assert np.array_equal(output, np.arange(12, 20))
    assert rb.dropped == 12


def test_write_larger_than_capacity_with_existing():
    rb = RingBuffer(capacity=8)
    rb.write(np.arange(3))
    rb.write(np.arange(100, 120))

    assert rb.dropped == 15

    output = rb.read(8)

    assert output is not None
    assert np.array_equal(output, np.arange(112, 120))


def test_close_drains_then_none():
    rb = RingBuffer(capacity=8)
    rb.write(np.arange(5))
    rb.close()

    output = rb.read(4)
    assert output is not None
    assert np.array_equal(output, np.array([0, 1, 2, 3]))
    assert rb.read(4) is None


def test_write_after_close_ignored():
    rb = RingBuffer(capacity=8)
    rb.close()
    rb.write(np.arange(4))
    assert rb.read(1) is None
    assert len(rb) == 0


def test_close_idempotent():
    rb = RingBuffer(capacity=8)
    rb.close()
    rb.close()
    assert rb.read(1) is None


def test_stress_no_corruption_when_no_overflow():
    rb = RingBuffer(capacity=100_000)
    total = 50_000

    def producer():
        i = 0
        while i < total:
            k = min(np.random.randint(1, 500), total - i)
            rb.write(np.arange(i, i + k, dtype=np.float32))
            i += k
        rb.close()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    got = []
    while True:
        out = rb.read(1024)
        if out is None:
            break
        got.append(out)
    t.join(timeout=5.0)
    got = np.concatenate(got)

    assert np.array_equal(got, np.arange(got.size))
    assert got.size >= total - 1024
