# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

"""Async device semantics (see docs/Async CPU Backend.md). Must pass with MAG_CPU_ASYNC=0 and 1."""

from __future__ import annotations

import gc
import threading

import numpy as np
import pytest

import magnetron as mag
from magnetron import Tensor, dtype


def test_sync_api() -> None:
    x = Tensor.full(256, 256, fill_value=1.0)
    y = (x * 2 + 1) @ x
    assert y.sync() is y
    mag.sync()
    mag.sync('cpu')
    assert y[0, 0].item() == 3.0 * 256


def test_sync_rejects_unknown_device() -> None:
    with pytest.raises(Exception):
        mag.sync('not_a_device')


def test_numpy_view_observes_later_inplace_ops() -> None:
    t = Tensor.zeros(64, 64)
    view = t.numpy()  # Shares storage with t
    for i in range(1, 20):
        t += 1.0
        assert view[0, 0] == i  # The ndarray must see each op without an explicit sync


def test_borrowed_numpy_memory_is_read_in_program_order() -> None:
    a = np.ones((128, 128), dtype=np.float32)
    t = Tensor(a, copy=False)
    y = t * 2.0
    a[:] = 7.0  # Mutating the source after the op returned must not affect its result
    assert y.sum().item() == 2.0 * a.size
    z = t * 2.0
    assert z.sum().item() == 14.0 * a.size


def test_borrowed_tensor_released_while_in_flight() -> None:
    for _ in range(50):
        a = np.random.rand(256, 256).astype(np.float32)
        t = Tensor(a, copy=False)
        y = (t @ t).sum()
        del t, a  # Last Python reference dropped while the command may still be queued
        gc.collect()
        assert np.isfinite(y.item())
    mag.sync()


def test_tolist_and_tobytes_wait_for_pending_work() -> None:
    x = Tensor.zeros(1024)
    for _ in range(100):
        x += 1.0
    assert x.tolist() == [100.0] * 1024
    assert np.frombuffer(x.tobytes(), dtype=np.float32).tolist() == [100.0] * 1024


def test_pow_with_python_scalar() -> None:
    x = Tensor.full(1024, fill_value=3.0)
    assert (x**2).sum().item() == 9.0 * 1024
    assert (x**3).sum().item() == 27.0 * 1024
    assert (x**0.5).tolist()[0] == pytest.approx(3.0**0.5)


def test_manual_seed_is_reproducible() -> None:
    def draw() -> list[float]:
        mag.context.manual_seed(42)
        return Tensor.uniform(4096).tolist()

    assert draw() == draw()


def test_deferred_kernel_error_is_raised() -> None:
    src = Tensor.full(10, fill_value=1.0)
    idx = Tensor([100], dtype=dtype.int64)
    with pytest.raises(RuntimeError, match='gather'):
        g = src.gather(0, idx)  # Async: queued, the kernel error surfaces at the next sync point
        mag.sync()
        g.tolist()
    mag.sync()  # The error was reported once and cleared
    assert (src + src).sum().item() == 20.0


def test_concurrent_submission_from_threads() -> None:
    errors: list[BaseException] = []
    results: dict[int, float] = {}

    def work(i: int) -> None:
        try:
            x = Tensor.zeros(128, 128)
            for _ in range(200):
                x += 1.0
            results[i] = x.sum().item()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(v == 200.0 * 128 * 128 for v in results.values())
