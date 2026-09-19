# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

"""The context used to be pinned to its creating thread. See THREAD_SAFETY.md."""

from __future__ import annotations

import threading
import traceback

import magnetron as mag
from magnetron import Tensor, context, dtype

_THREADS = 8
_STEPS = 50
_ROWS = 256


def _run(target, count: int = _THREADS) -> list[str]:
    errors: list[str] = []

    def guarded(i: int) -> None:
        try:
            target(i)
        except Exception:
            errors.append(traceback.format_exc())

    threads = [threading.Thread(target=guarded, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_tensors_are_usable_from_any_thread() -> None:
    sums: dict[int, float] = {}

    def work(i: int) -> None:
        x = Tensor.full(_ROWS, _ROWS, fill_value=float(i))
        for _ in range(_STEPS):
            x = (x + 1.0).relu()
        sums[i] = x.sum().item()

    errors = _run(work)
    assert not errors, errors[0]
    for i in range(_THREADS):
        expected = float(i + _STEPS) * _ROWS * _ROWS
        assert abs(sums[i] - expected) <= expected * 1e-4


def test_views_are_usable_from_any_thread() -> None:
    """Covers mag_strided_view, which carried its own copy of the thread guard."""
    shapes: dict[int, int] = {}

    def work(i: int) -> None:
        x = Tensor.full(_ROWS, _ROWS, fill_value=float(i))
        shapes[i] = x[: _ROWS // 2, :].contiguous().numel

    errors = _run(work)
    assert not errors, errors[0]
    assert set(shapes.values()) == {_ROWS // 2 * _ROWS}


def test_tensor_freed_on_another_thread() -> None:
    """A refcount can reach zero on any thread, so the storage destructor must tolerate it."""
    box: dict[str, Tensor] = {}

    create = threading.Thread(target=lambda: box.__setitem__('x', Tensor.ones(1024, 1024)))
    create.start()
    create.join()
    assert box['x'].numel == 1024 * 1024

    release = threading.Thread(target=lambda: box.pop('x'))
    release.start()
    release.join()
    assert not box


def test_default_device_is_per_thread() -> None:
    seen: dict[int, str] = {}

    def work(i: int) -> None:
        with mag.device('cpu'):
            seen[i] = Tensor.zeros(8).device

    errors = _run(work)
    assert not errors, errors[0]
    assert len(seen) == _THREADS
    assert set(seen.values()) == {'cpu'}


def test_grad_recording_is_per_thread() -> None:
    inner: dict[int, bool] = {}
    outer: dict[int, bool] = {}

    def work(i: int) -> None:
        if i % 2:
            with mag.no_grad():
                inner[i] = context.is_grad_recording()
        else:
            for _ in range(200):
                outer[i] = context.is_grad_recording()

    errors = _run(work)
    assert not errors, errors[0]
    assert inner and not any(inner.values()), 'no_grad leaked out of its thread'
    assert outer and all(outer.values()), 'another thread disabled grad recording'


def test_default_dtype_is_per_thread() -> None:
    seen: dict[int, str] = {}

    def work(i: int) -> None:
        if i % 2:
            context.set_default_dtype(dtype.float16)
        seen[i] = str(context.get_default_dtype())

    errors = _run(work)
    assert not errors, errors[0]
    assert 'float16' in seen[1]
    assert 'float32' in seen[0]
    assert 'float32' in str(context.get_default_dtype())


def test_fresh_thread_gets_the_documented_defaults() -> None:
    got: dict[int, tuple[str, str, bool]] = {}

    with mag.no_grad(), mag.device('cpu'):
        context.set_default_dtype(dtype.float16)
        try:

            def work(i: int) -> None:
                got[i] = (
                    context.get_default_device(),
                    str(context.get_default_dtype()),
                    context.is_grad_recording(),
                )

            errors = _run(work, count=2)
        finally:
            context.set_default_dtype(dtype.float32)

    assert not errors, errors[0]
    assert len(got) == 2
    for dev, dt, grad in got.values():
        assert dev == 'cpu'
        assert 'float32' in dt
        assert grad


def test_backward_restores_grad_state() -> None:
    x = Tensor.ones(4, requires_grad=True)
    loss = (x * x).sum()
    with mag.no_grad():
        assert not context.is_grad_recording()
        loss.backward()
        assert not context.is_grad_recording()
    assert context.is_grad_recording()


def test_concurrent_backward_on_disjoint_graphs() -> None:
    grads: dict[int, float] = {}

    def work(i: int) -> None:
        x = Tensor.full(64, fill_value=float(i + 1), requires_grad=True)
        (x * x).sum().backward()
        grads[i] = x.grad.sum().item()

    errors = _run(work)
    assert not errors, errors[0]
    for i in range(_THREADS):
        expected = 2.0 * float(i + 1) * 64
        assert abs(grads[i] - expected) <= expected * 1e-4
