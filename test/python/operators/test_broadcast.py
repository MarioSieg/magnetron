# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch

from ..common import *

_PAIRS: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((3, 1), (1, 4)),
    ((2, 3, 4), (4,)),
    ((2, 3, 4), (3, 1)),
    ((5,), (2, 5)),
    ((1,), (3, 3)),
    ((), (2, 2)),
    ((2, 1, 3), (1, 4, 1)),
    ((4, 3), (3,)),
    ((2, 3), (1, 3)),
    ((1, 3, 1), (2, 1, 4)),
    ((2, 1, 1, 5), (3, 4, 1)),
    ((6, 1), (1,)),
    ((1, 33), (17, 1)),
    ((3, 1, 5), (1, 1, 5)),
)

_FLOORDIV = lambda a, b: a // b

_ARITH: tuple[tuple[str, Callable], ...] = (
    ('add', lambda a, b: a + b),
    ('sub', lambda a, b: a - b),
    ('mul', lambda a, b: a * b),
    ('truediv', lambda a, b: a / b),
    ('floordiv', _FLOORDIV),
    ('mod', lambda a, b: a % b),
    ('pow', lambda a, b: a**b),
)

_COMPARE: tuple[tuple[str, Callable], ...] = (
    ('eq', lambda a, b: a == b),
    ('ne', lambda a, b: a != b),
    ('lt', lambda a, b: a < b),
    ('le', lambda a, b: a <= b),
    ('gt', lambda a, b: a > b),
    ('ge', lambda a, b: a >= b),
)

_MINMAX: tuple[tuple[str, Callable, Callable], ...] = (
    ('min', lambda a, b: a.min(b), torch.minimum),
    ('max', lambda a, b: a.max(b), torch.maximum),
)

_DIV_LIKE = {'truediv', 'floordiv', 'mod'}
_WIDE_UNSIGNED = {dtype.uint16, dtype.uint32, dtype.uint64}
_INT_DTYPES = tuple(sorted(dtype.integer, key=lambda d: d.name))
_FLOAT_DTYPES = tuple(sorted(FLOATING_NO_FLOAT8, key=lambda d: d.name))


def _operand(shape: tuple[int, ...], dt: dtype.DType, op: str, is_rhs: bool, device: str) -> Tensor:
    if op == 'pow':
        if dt.is_integer():
            return uniform_tensor(shape, low=0, high=4, dtype=dt, device=device)
        return uniform_tensor(shape, low=0.5, high=2.0, dtype=dt, device=device) if not is_rhs else uniform_tensor(shape, low=-2.0, high=2.0, dtype=dt, device=device)
    if is_rhs and op in _DIV_LIKE:
        if dt.is_integer():
            y = random_tensor(shape, dt, device)
            return y + (y == 0).cast(dt)
        mag = uniform_tensor(shape, low=0.5, high=2.0, dtype=dt, device=device)
        sign = Tensor.bernoulli(shape, p=0.5, device=device).cast(dt) * 2.0 - 1.0
        return mag * sign
    return random_tensor(shape, dt, device)


def _torch_ref(fn: Callable, x: Tensor, y: Tensor, dt: dtype.DType) -> torch.Tensor:
    tx, ty = totorch(x), totorch(y)
    if fn is _FLOORDIV and dt.is_floating_point():
        return torch_floordiv(tx, ty)
    if dt in _WIDE_UNSIGNED:
        r = fn(tx.to(torch.int64), ty.to(torch.int64))
        return r if r.dtype == torch.bool else r.to(totorch_dtype(dt))
    return fn(tx, ty)


def _assert_same(r: Tensor, ref: torch.Tensor, dt: dtype.DType) -> None:
    assert r.shape == tuple(ref.shape)
    if ref.dtype == torch.bool:
        assert r.dtype == dtype.boolean
        assert r.tolist() == ref.tolist()
    elif dt.is_integer() and r.dtype.is_integer():
        assert r.dtype == dt
        assert r.tolist() == ref.tolist()
    else:
        assert_close_mag_torch(r, ref, dt)


def _run_forward(device: str, dt: dtype.DType, name: str, fn: Callable, ref_fn: Callable, xs: tuple[int, ...], ys: tuple[int, ...]) -> None:
    x = _operand(xs, dt, name, False, device)
    y = _operand(ys, dt, name, True, device)
    _assert_same(fn(x, y), _torch_ref(ref_fn, x, y, dt), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _FLOAT_DTYPES, ids=[d.name for d in _FLOAT_DTYPES])
@pytest.mark.parametrize('name, fn', _ARITH + _COMPARE, ids=[c[0] for c in _ARITH + _COMPARE])
@pytest.mark.parametrize('swap', [False, True])
def test_broadcast_float(device: str, dt: dtype.DType, name: str, fn: Callable, swap: bool) -> None:
    for xs, ys in _PAIRS:
        if swap:
            xs, ys = ys, xs
        _run_forward(device, dt, name, fn, fn, xs, ys)


_INT_OPS = tuple(c for c in _ARITH if c[0] != 'truediv') + _COMPARE


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _INT_DTYPES, ids=[d.name for d in _INT_DTYPES])
@pytest.mark.parametrize('name, fn', _INT_OPS, ids=[c[0] for c in _INT_OPS])
@pytest.mark.parametrize('swap', [False, True])
def test_broadcast_integer(device: str, dt: dtype.DType, name: str, fn: Callable, swap: bool) -> None:
    for xs, ys in _PAIRS:
        if swap:
            xs, ys = ys, xs
        _run_forward(device, dt, name, fn, fn, xs, ys)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _FLOAT_DTYPES + _INT_DTYPES, ids=[d.name for d in _FLOAT_DTYPES + _INT_DTYPES])
@pytest.mark.parametrize('name, fn, ref_fn', _MINMAX, ids=[c[0] for c in _MINMAX])
@pytest.mark.parametrize('swap', [False, True])
def test_broadcast_minmax(device: str, dt: dtype.DType, name: str, fn: Callable, ref_fn: Callable, swap: bool) -> None:
    for xs, ys in _PAIRS:
        if swap:
            xs, ys = ys, xs
        _run_forward(device, dt, name, fn, ref_fn, xs, ys)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16, dtype.int32, dtype.uint8), ids=['float32', 'float16', 'int32', 'uint8'])
@pytest.mark.parametrize('name, fn', _ARITH, ids=[c[0] for c in _ARITH])
def test_broadcast_inplace(device: str, dt: dtype.DType, name: str, fn: Callable) -> None:
    if dt.is_integer() and name == 'truediv':
        pytest.skip('integer true division produces a float result and cannot be applied in place')
    if dt.is_floating_point() and name == 'floordiv':
        pytest.skip('in-place float floor division is rejected; covered by test_inplace_float_floordiv_like_torch')
    dunder = f'__i{name}__'
    for xs, ys in _PAIRS:
        big = broadcast_shape(xs, ys)
        x = _operand(big, dt, name, False, device)
        y = _operand(ys, dt, name, True, device)
        expected = _torch_ref(fn, x, y, dt)
        z = x.clone()
        getattr(z, dunder)(y)
        _assert_same(z, expected, dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, fn', _ARITH + _COMPARE, ids=[c[0] for c in _ARITH + _COMPARE])
def test_broadcast_with_python_scalar(device: str, name: str, fn: Callable) -> None:
    for shape in ((), (3,), (2, 3), (2, 1, 4)):
        x = _operand(shape, dtype.float32, name, False, device)
        s = 1.75
        _assert_same(fn(x, s), fn(totorch(x), s), dtype.float32)
        if name in _DIV_LIKE:
            x = uniform_tensor(shape, low=0.5, high=2.0, dtype=dtype.float32, device=device)
        _assert_same(fn(s, x), fn(s, totorch(x)), dtype.float32)


_GRAD_OPS: tuple[tuple[str, Callable, Callable], ...] = (
    ('add', lambda a, b: a + b, lambda a, b: a + b),
    ('sub', lambda a, b: a - b, lambda a, b: a - b),
    ('mul', lambda a, b: a * b, lambda a, b: a * b),
    ('truediv', lambda a, b: a / b, lambda a, b: a / b),
    ('pow', lambda a, b: a**b, lambda a, b: a**b),
    ('min', lambda a, b: a.min(b), torch.minimum),
    ('max', lambda a, b: a.max(b), torch.maximum),
)


def _grad_leaf(shape: tuple[int, ...], name: str, is_rhs: bool, device: str) -> tuple[Tensor, torch.Tensor]:
    if name == 'pow':
        x = uniform_tensor(shape, low=0.5, high=2.0, device=device) if not is_rhs else uniform_tensor(shape, low=-2.0, high=2.0, device=device)
    elif name == 'truediv' and is_rhs:
        x = uniform_tensor(shape, low=0.5, high=2.0, device=device)
    else:
        x = uniform_tensor(shape, low=-2.0, high=2.0, device=device)
    tx = totorch(x).clone().requires_grad_(True)
    x.requires_grad = True
    return x, tx


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, fn, ref_fn', _GRAD_OPS, ids=[c[0] for c in _GRAD_OPS])
@pytest.mark.parametrize('swap', [False, True])
def test_broadcast_backward(device: str, name: str, fn: Callable, ref_fn: Callable, swap: bool) -> None:
    for xs, ys in _PAIRS:
        if swap:
            xs, ys = ys, xs
        x, tx = _grad_leaf(xs, name, False, device)
        y, ty = _grad_leaf(ys, name, True, device)
        r = fn(x, y)
        tr = ref_fn(tx, ty)
        assert r.shape == tuple(tr.shape)
        assert_close_mag_torch(r, tr.detach(), dtype.float32)
        w = random_tensor(r.shape, dtype.float32, device)
        (r * w).sum().backward()
        (tr * totorch(w)).sum().backward()
        assert x.grad.shape == xs
        assert y.grad.shape == ys
        assert_close_mag_torch(x.grad, tx.grad, dtype.float32)
        assert_close_mag_torch(y.grad, ty.grad, dtype.float32)
