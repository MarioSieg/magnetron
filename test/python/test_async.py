# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

"""Async device semantics (see docs/Async CPU Backend.md). Must pass with MAG_CPU_ASYNC=0 and 1."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
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
    with pytest.raises(IndexError, match='gather'):  # Out of bounds kernel error
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


def test_write_after_read_hazard() -> None:
    a = Tensor.full(512, 512, fill_value=1.0)
    b = a * 3.0  # Reads a
    a += 5.0  # Must not be visible to the queued read above
    assert b.sum().item() == 3.0 * a.numel
    assert a.sum().item() == 6.0 * a.numel


def test_write_through_view_is_seen_by_base() -> None:
    base = Tensor.zeros(64, 64)
    row = base[3]
    for _ in range(10):
        row += 1.0
    col_sum = base.sum(dim=1)
    assert col_sum.tolist()[3] == 10.0 * 64
    assert base.sum().item() == 10.0 * 64


def test_long_dependent_chain_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((128, 128)).astype(np.float32)
    x = Tensor(a_np)
    ref = a_np.copy()
    for i in range(300):
        x = (x * 0.5 + 0.25).tanh() if i % 2 else x @ Tensor(a_np) * (1.0 / 128)
        ref = np.tanh(ref * 0.5 + 0.25) if i % 2 else ref @ a_np * (1.0 / 128)
    np.testing.assert_allclose(x.numpy(), ref, rtol=1e-4, atol=1e-5)


def test_temporaries_dropped_while_queued() -> None:
    acc = Tensor.zeros(256)
    for i in range(2000):
        tmp = Tensor.full(256, fill_value=float(i))  # Dropped right after the op is queued
        acc += tmp
        del tmp
    assert acc.tolist()[0] == float(sum(range(2000)))


def test_repr_waits_for_pending_work() -> None:
    x = Tensor.zeros(4)
    for _ in range(50):
        x += 2.0
    assert '100' in repr(x)


def test_numpy_copy_is_decoupled() -> None:
    a = np.zeros(1024, dtype=np.float32)
    t = Tensor(a)  # Copy
    y = t + 1.0
    a[:] = 99.0
    assert y.sum().item() == 1024.0


def test_numpy_export_after_queued_ops() -> None:
    x = Tensor.zeros(256, 256)
    for _ in range(64):
        x = x + 1.0
    assert np.all(x.numpy() == 64.0)


def test_data_ptr_waits_for_pending_work() -> None:
    import ctypes

    x = Tensor.zeros(4096)
    for _ in range(32):
        x += 1.0
    buf = (ctypes.c_float * x.numel).from_address(x.data_ptr)
    assert buf[0] == 32.0 and buf[x.numel - 1] == 32.0


def test_autograd_through_async_ops() -> None:
    x = Tensor.full(64, 64, fill_value=2.0, requires_grad=True)
    w = Tensor.full(64, 64, fill_value=0.5, requires_grad=True)
    y = ((x @ w).tanh() * 3.0).sum()
    y.backward()
    t = np.tanh(np.full((64, 64), 64.0, dtype=np.float32))
    gx_ref = (3.0 * (1.0 - t * t)) @ np.full((64, 64), 0.5, dtype=np.float32).T
    np.testing.assert_allclose(x.grad.numpy(), gx_ref, rtol=1e-4, atol=1e-6)


def test_snapshot_writes_queued_results(tmp_path) -> None:
    from magnetron.snapshot import SnapshotWriter, deserialize

    x = Tensor.zeros(1024)
    for _ in range(25):
        x += 4.0
    path = tmp_path / 'async.mag'
    with SnapshotWriter(path) as w:
        w.declare('x', x.shape, x.dtype)
        w.write('x', x)
    tensors, _ = deserialize(path, mmap=False)
    assert tensors['x'].tolist() == [100.0] * 1024


def test_deferred_error_surfaces_on_item() -> None:
    src = Tensor.full(10, fill_value=1.0)
    idx = Tensor([1000], dtype=dtype.int64)
    with pytest.raises(IndexError):  # Out of bounds kernel errors raise IndexError, also when deferred
        src.gather(0, idx).sum().item()
    assert (src * 2).sum().item() == 20.0  # Device usable again


def test_interleaved_seed_and_rng_ops() -> None:
    mag.context.manual_seed(7)
    a = Tensor.uniform(1024)
    b = a * 2.0
    mag.context.manual_seed(7)
    c = Tensor.uniform(1024)
    assert (b - c * 2.0).abs().sum().item() == 0.0


# Mode specific behavior: each case runs in a fresh interpreter so the env vars take effect at context creation.

_ASYNC_MODES = ['0', '1']


def _run(code: str, **env: str) -> subprocess.CompletedProcess:
    # Hermetic child: drop inherited MAG_* config (CI sets e.g. MAG_LOG_LEVEL, MAG_LAUNCH_BLOCKING, MAG_CPU_ASYNC)
    full_env = {k: v for k, v in os.environ.items() if not k.startswith('MAG_')}
    full_env.update({'MAG_LOG_LEVEL': 'error', **env})
    return subprocess.run([sys.executable, '-c', textwrap.dedent(code)], env=full_env, capture_output=True, text=True, timeout=120)


_DETERMINISM_SCRIPT = """
    import hashlib, numpy as np
    import magnetron as mag
    from magnetron import Tensor
    mag.context.manual_seed(1234)
    x = Tensor.uniform(256, 256)
    w = Tensor.uniform(256, 256)
    for _ in range(40):
        x = (x @ w * (1.0 / 256)).gelu() + x.softmax(dim=-1)
    print(hashlib.sha256(x.numpy().tobytes()).hexdigest())
"""


def test_async_and_sync_results_are_bitwise_identical() -> None:
    env = {'MAG_CPU_THREADS': '8'}  # Same intra-op split in both modes
    outs = []
    for mode in _ASYNC_MODES:
        r = _run(_DETERMINISM_SCRIPT, MAG_CPU_ASYNC=mode, **env)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1]


@pytest.mark.parametrize('mode', _ASYNC_MODES)
def test_async_mode_is_selected(mode: str) -> None:
    r = _run('import magnetron as mag; mag.Tensor.zeros(4).sum().item()', MAG_CPU_ASYNC=mode, MAG_LOG_LEVEL='info')
    assert r.returncode == 0, r.stderr
    log = r.stdout + r.stderr
    assert ('async command stream enabled' in log) == (mode == '1')


def test_launch_blocking_raises_at_call_site() -> None:
    code = """
        from magnetron import Tensor, dtype
        src = Tensor.full(10, fill_value=1.0)
        idx = Tensor([100], dtype=dtype.int64)
        try:
            src.gather(0, idx)
        except IndexError:
            print('raised')
    """
    r = _run(code, MAG_CPU_ASYNC='1', MAG_LAUNCH_BLOCKING='1')
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == 'raised'


@pytest.mark.parametrize('depth,mb', [('1', '1'), ('2', '1'), ('7', '1')])
def test_tiny_queue_backpressure(depth: str, mb: str) -> None:
    code = """
        from magnetron import Tensor
        x = Tensor.zeros(1024, 1024)  # 4 MiB, larger than the 1 MiB budget
        y = Tensor.zeros(16)
        for _ in range(200):
            x += 1.0
            y += 1.0
        print(x.sum().item(), y.sum().item())
    """
    r = _run(code, MAG_CPU_ASYNC='1', MAG_CPU_QUEUE_DEPTH=depth, MAG_CPU_QUEUE_MB=mb)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(200.0 * 1024 * 1024), str(200.0 * 16)]


@pytest.mark.parametrize('mode', _ASYNC_MODES)
def test_exit_with_pending_work(mode: str) -> None:
    code = """
        from magnetron import Tensor
        x = Tensor.uniform(512, 512)
        for _ in range(500):
            x = x @ x * 0.001
    """  # No sync: interpreter shutdown must drain the queue without hanging or crashing
    r = _run(code, MAG_CPU_ASYNC=mode)
    assert r.returncode == 0, r.stderr


def test_exit_with_pending_error() -> None:
    code = """
        from magnetron import Tensor, dtype
        x = Tensor.uniform(1024, 1024)
        y = x @ x  # Keeps the queue busy so the gather below is queued, not run inline
        Tensor.full(10, fill_value=1.0).gather(0, Tensor([100], dtype=dtype.int64))
    """  # The error is never synced by the program: shutdown must report it without crashing or hanging
    r = _run(code, MAG_CPU_ASYNC='1')
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize('value', ['0', 'abc', '99999999'])
def test_invalid_queue_env_falls_back(value: str) -> None:
    r = _run('from magnetron import Tensor; print((Tensor.zeros(8) + 1).sum().item())', MAG_CPU_ASYNC='1', MAG_CPU_QUEUE_DEPTH=value, MAG_CPU_QUEUE_MB=value)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == '8.0'


def test_scalar_exponent_pow_with_queued_work() -> None:
    busy = Tensor.zeros(512, 512)
    for _ in range(50):
        busy += 1.0
    x = Tensor.full(256, fill_value=3.0)
    assert (x**2).tolist()[0] == 9.0  # Python number exponent goes through mag_pow_scalar, never read back
    assert busy.sum().item() == 50.0 * busy.numel


@pytest.mark.parametrize('dt', [dtype.float32, dtype.float16, dtype.bfloat16, dtype.int32, dtype.int64, dtype.uint8])
def test_scalar_ops_all_dtypes(dt: dtype.DType) -> None:
    x = Tensor.full(8, fill_value=3, dtype=dt)
    assert (x + 2).tolist() == [5] * 8  # Python scalar operand becomes a scalar tensor of dtype dt
    assert (x * 0.5 if dt in (dtype.float32, dtype.float16, dtype.bfloat16) else x * 2).tolist() == [1.5 if dt in (dtype.float32, dtype.float16, dtype.bfloat16) else 6] * 8


def test_tensor_index_does_not_read_indices_on_host() -> None:
    table = Tensor.arange(0, 40, dtype=dtype.float32).reshape(10, 4)
    busy = Tensor.zeros(512, 512)
    for _ in range(50):
        busy += 1.0  # Pending work must not change indexing results
    assert table[Tensor([[3]], dtype=dtype.int64)].tolist() == [[[12.0, 13.0, 14.0, 15.0]]]
    assert table[Tensor([-1, 0], dtype=dtype.int64)].tolist() == [[36.0, 37.0, 38.0, 39.0], [0.0, 1.0, 2.0, 3.0]]
    assert table[Tensor([2], dtype=dtype.int64), Tensor([-1], dtype=dtype.int64)].tolist() == [11.0]
    assert busy.sum().item() == 50.0 * busy.numel


def test_tensor_index_out_of_bounds_raised_by_kernel() -> None:
    table = Tensor.arange(0, 40, dtype=dtype.float32).reshape(10, 4)
    with pytest.raises(IndexError):  # Async: raised when the result is read
        table[Tensor([10], dtype=dtype.int64)].tolist()
    with pytest.raises(IndexError):
        table[Tensor([-11], dtype=dtype.int64)].tolist()
    with pytest.raises(IndexError):
        table[Tensor([0], dtype=dtype.int64), Tensor([4], dtype=dtype.int64)].tolist()  # In range once linearized, still out of bounds per axis
    with pytest.raises(IndexError):
        table[Tensor([0], dtype=dtype.int64), Tensor([-5], dtype=dtype.int64)].tolist()
    assert table[Tensor([1], dtype=dtype.int64)].tolist() == [[4.0, 5.0, 6.0, 7.0]]  # Error was taken, device usable again


def test_tensor_index_computed_on_device() -> None:
    table = Tensor.arange(0, 40, dtype=dtype.float32).reshape(10, 4)
    idx = Tensor([1, 2], dtype=dtype.int64) * 3  # Still being computed
    assert table[idx].tolist() == [[12.0, 13.0, 14.0, 15.0], [24.0, 25.0, 26.0, 27.0]]
    with pytest.raises(IndexError):
        table[Tensor([4], dtype=dtype.int64) * 3].tolist()
    assert table[Tensor([-1], dtype=dtype.int32)].tolist() == [[36.0, 37.0, 38.0, 39.0]]  # Cast index


def test_tensor_index_setitem() -> None:
    t = Tensor.zeros(5, 3)
    t[Tensor([0, -1], dtype=dtype.int64)] = 7.0
    assert t.sum(dim=1).tolist() == [21.0, 0.0, 0.0, 0.0, 21.0]
    with pytest.raises(IndexError):
        t[Tensor([5], dtype=dtype.int64)] = 1.0
        t.sync()
    with pytest.raises(IndexError):
        t[Tensor([0], dtype=dtype.int64), Tensor([3], dtype=dtype.int64)] = 1.0  # Valid flat offset, out of bounds on axis 1
        t.sync()
    assert t.sum(dim=1).tolist() == [21.0, 0.0, 0.0, 0.0, 21.0]  # Rejected writes left t unchanged


@pytest.mark.parametrize('op', ['gather', 'scatter', 'scatter_add', 'index_add', 'one_hot'])
def test_index_kernels_bounds_checked(op: str) -> None:
    x = Tensor.arange(0, 12, dtype=dtype.float32).reshape(3, 4)
    bad = Tensor([[0, 4]], dtype=dtype.int64)  # 4 is out of bounds for dim 1
    with pytest.raises(IndexError):
        if op == 'gather':
            x.gather(1, bad).tolist()
        elif op == 'scatter':
            x.scatter(1, bad, Tensor.ones(1, 2)).tolist()
        elif op == 'scatter_add':
            x.scatter_add(1, bad, Tensor.ones(1, 2)).tolist()
        elif op == 'index_add':
            x.index_add(1, Tensor([0, 4], dtype=dtype.int64), Tensor.ones(3, 2)).tolist()
        else:
            Tensor([0, 3], dtype=dtype.int64).one_hot(3).tolist()


@pytest.mark.parametrize('e', [0, 1, 2, 3, 4, 5, 0.5, -0.5, -1, 2.5, -3.0, True])
def test_pow_python_number_exponent(e: float) -> None:
    a = np.linspace(0.5, 3.0, 16, dtype=np.float32)
    x = Tensor(a)
    ref = a ** np.float32(e)
    np.testing.assert_allclose((x**e).numpy(), ref, rtol=1e-5)
    np.testing.assert_allclose(x.pow(e).numpy(), ref, rtol=1e-5)
    y = Tensor(a)
    y **= e
    np.testing.assert_allclose(y.numpy(), ref, rtol=1e-5)
    z = Tensor(a)
    z.pow_(e)
    np.testing.assert_allclose(z.numpy(), ref, rtol=1e-5)


def test_pow_python_number_exponent_dtypes() -> None:
    xi = Tensor([1, 2, 3], dtype=dtype.int32)
    assert (xi**2).dtype == dtype.int32 and (xi**2).tolist() == [1, 4, 9]
    assert (xi**3).tolist() == [1, 8, 27]
    r = xi**0.5  # Float exponent on int base promotes like a float scalar tensor would
    assert r.dtype == mag.context.get_default_dtype()
    np.testing.assert_allclose(r.numpy(), np.sqrt([1.0, 2.0, 3.0]), rtol=1e-6)
    assert (2**xi).tolist() == [2, 4, 8]  # Reflected: base is the number, unchanged path


def test_pow_python_number_exponent_grad() -> None:
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x**3).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [3.0, 12.0, 27.0], rtol=1e-6)
    x = Tensor([1.0, 4.0], requires_grad=True)
    (x**1.5).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [1.5, 3.0], rtol=1e-6)
