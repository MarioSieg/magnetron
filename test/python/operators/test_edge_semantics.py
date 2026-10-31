# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>
# Edge-case semantics compared against torch: empty tensors, NaN propagation, scalar promotion, advanced indexing.

from __future__ import annotations

import torch

from ..common import *

_NAN = float('nan')
_INF = float('inf')
_HALF_FLOATS = (dtype.float32, dtype.float16, dtype.bfloat16)


def _i64(data: Any, device: str) -> Tensor:
    return Tensor(data, dtype=dtype.int64, device=device)


def _same(r: Tensor | tuple, ref: torch.Tensor | tuple) -> None:
    if isinstance(ref, tuple):  # (values, indices) results
        assert isinstance(r, tuple) and len(r) == len(ref)
        for a, b in zip(r, ref):
            _same(a, b)
        return
    assert r.shape == tuple(ref.shape)
    assert totorch_dtype(r.dtype) == ref.dtype
    if ref.dtype.is_floating_point:
        torch.testing.assert_close(totorch(r), ref, equal_nan=True)
    else:
        assert r.tolist() == ref.tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('op', ['sum', 'mean', 'prod'])
@pytest.mark.parametrize('shape, dim', [((0,), None), ((3, 0), 1), ((0, 3), 0), ((2, 0, 3), 1), ((0, 4), None)])
def test_empty_reduction_yields_identity_like_torch(device: str, op: str, shape: tuple[int, ...], dim: Any) -> None:
    x = Tensor.zeros(shape, device=device)
    tx = torch.zeros(shape)
    r = getattr(x, op)() if dim is None else getattr(x, op)(dim)
    _same(r, getattr(tx, op)() if dim is None else getattr(tx, op)(dim))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_empty_reduction_integer_and_logical(device: str) -> None:
    assert Tensor.zeros(0, dtype=dtype.int32, device=device).sum().item() == 0
    assert Tensor.zeros(0, dtype=dtype.int32, device=device).prod().item() == 1
    assert Tensor.zeros(0, dtype=dtype.boolean, device=device).any().item() is False
    assert Tensor.zeros(0, dtype=dtype.boolean, device=device).all().item() is True


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('op', ['max', 'min', 'amax', 'amin', 'argmax', 'argmin'])
def test_empty_extremum_reduction_raises_like_torch(device: str, op: str) -> None:
    with pytest.raises(RuntimeError):
        getattr(Tensor.zeros(0, device=device), op)()
    with pytest.raises(RuntimeError):
        getattr(Tensor.zeros(3, 0, device=device), op)(1)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_empty_reduction_backward(device: str) -> None:
    x = Tensor.zeros(0, device=device)
    x.requires_grad = True
    x.sum().backward()
    assert x.grad.shape == (0,)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _HALF_FLOATS, ids=[d.name for d in _HALF_FLOATS])
@pytest.mark.parametrize('op', ['max', 'min', 'amax', 'amin', 'argmax', 'argmin'])
def test_extremum_reduction_propagates_nan_like_torch(device: str, dt: dtype.DType, op: str) -> None:
    vals = [float(i % 7) - 3.0 for i in range(41)]  # Long enough for the SIMD paths
    vals[17] = _NAN
    vals[29] = _NAN
    for data in ([1.0, _NAN, 3.0, _NAN, -1.0], vals, [_NAN, _NAN]):
        x = Tensor(data, dtype=dt, device=device)
        _same(getattr(x, op)(), getattr(totorch(x), op)())
    rows = [[float((r * 5 + c * 7) % 11) for c in range(20)] for r in range(12)]
    rows[3][5] = _NAN
    rows[7][19] = _NAN
    x = Tensor(rows, dtype=dt, device=device)
    for dim in (0, 1):
        _same(getattr(x, op)(dim), getattr(totorch(x), op)(dim))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_max_dim_with_nan_returns_first_nan_like_torch(device: str) -> None:
    x = Tensor([[1.0, _NAN, 2.0, _NAN], [3.0, 1.0, 2.0, 0.0]], device=device)
    vals, idx = x.max(1)
    tvals, tidx = totorch(x).max(1)
    _same(vals, tvals)
    _same(idx, tidx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _HALF_FLOATS, ids=[d.name for d in _HALF_FLOATS])
@pytest.mark.parametrize('data', [[[-_INF, -_INF], [0.0, 1.0]], [[_NAN, 1.0, 2.0]], [[_INF, 1.0]], [[-_INF, 0.0, 0.0]], [[-_INF] * 37]])
def test_softmax_non_finite_rows_like_torch(device: str, dt: dtype.DType, data: list) -> None:
    x = Tensor(data, dtype=dt, device=device)
    _same(x.softmax(), totorch(x).softmax(-1))


_SCALAR_PROMOTION_CASES = (
    (dtype.int32, lambda a: a * 1.5),
    (dtype.int32, lambda a: 1.5 * a),
    (dtype.int32, lambda a: a + 1),
    (dtype.uint8, lambda a: a - 1.0),
    (dtype.int64, lambda a: a / 2),
    (dtype.int32, lambda a: a // 2.0),
    (dtype.int32, lambda a: a**0.5),
    (dtype.int32, lambda a: a > 1.5),
    (dtype.int32, lambda a: a == 2.0),
    (dtype.float16, lambda a: a * 2.5),
    (dtype.bfloat16, lambda a: a + 1),
    (dtype.boolean, lambda a: a + 1),
    (dtype.boolean, lambda a: a * 2.0),
    (dtype.int32, lambda a: a.clamp(1.5, 3.5)),
    (dtype.int32, lambda a: a.clamp_min(1.5)),
    (dtype.int32, lambda a: a.clamp_max(2.5)),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('case', range(len(_SCALAR_PROMOTION_CASES)))
def test_python_scalar_promotion_like_torch(device: str, case: int) -> None:
    dt, fn = _SCALAR_PROMOTION_CASES[case]
    data = [True, False, True] if dt == dtype.boolean else [1, 2, 4]
    x = Tensor(data, dtype=dt, device=device)
    _same(fn(x), fn(totorch(x)))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_where_promotes_scalar_like_torch(device: str) -> None:
    cond = Tensor([True, False], device=device)
    x = Tensor([1, 2], dtype=dtype.int32, device=device)
    _same(Tensor.where(cond, x, 1.5), torch.where(totorch(cond), totorch(x), 1.5))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_inplace_float_scalar_into_int_raises_like_torch(device: str) -> None:
    x = Tensor([1, 2], dtype=dtype.int32, device=device)
    with pytest.raises(RuntimeError):
        x *= 1.5


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('op', ['cusum', 'cuprod'])
@pytest.mark.parametrize('dt', (dtype.int8, dtype.uint8, dtype.int32, dtype.boolean), ids=['int8', 'uint8', 'int32', 'boolean'])
def test_integer_scans_widen_to_int64_like_torch(device: str, op: str, dt: dtype.DType) -> None:
    data = [True, True, False, True] if dt == dtype.boolean else [100, 100, 0, 3]
    x = Tensor(data, dtype=dt, device=device)
    _same(getattr(x, op)(0), getattr(totorch(x), 'cumsum' if op == 'cusum' else 'cumprod')(0))


_BIN_OPS = ('__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__', '__lt__', '__eq__')


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16, dtype.int64, dtype.int32, dtype.uint8), ids=['float32', 'float16', 'int64', 'int32', 'uint8'])
@pytest.mark.parametrize('op', _BIN_OPS)
def test_binary_expanded_with_scalar_operand(device: str, dt: dtype.DType, op: str) -> None:
    x = Tensor([[1], [2]], dtype=dt, device=device).expand(2, 3)
    tx = totorch(x)
    for rhs, trhs in ((Tensor(3, dtype=dt, device=device), torch.tensor(3, dtype=totorch_dtype(dt))), (3, 3)):
        _same(getattr(x, op)(rhs), getattr(tx, op)(trhs))
        if isinstance(rhs, Tensor):  # Scalar on the left hand side
            _same(getattr(rhs, op)(x), getattr(trhs, op)(tx))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_empty_slices_like_torch(device: str) -> None:
    x = Tensor.arange(12, device=device).reshape(3, 4)
    tx = torch.arange(12).reshape(3, 4)
    _same(x[1:1], tx[1:1])
    _same(x[:, 2:2], tx[:, 2:2])
    _same(x[5:], tx[5:])
    _same(x.narrow(1, 4, 0), tx.narrow(1, 4, 0))
    _same(x.cast(dtype.float32)[1:1].sum(0), tx.float()[1:1].sum(0))
    assert Tensor.zeros(3, 0, device=device).tolist() == [[], [], []]
    assert Tensor.zeros(2, 3, 0, device=device).tolist() == torch.zeros(2, 3, 0).tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_split_with_section_sizes_like_torch(device: str) -> None:
    x = Tensor.arange(12, device=device).reshape(2, 6)
    tx = torch.arange(12).reshape(2, 6)
    for sections, dim in (([2, 4], 1), ([1, 2, 3], 1), ([0, 6], 1), ([1, 1], 0), ((2, 0, 4), 1)):
        parts = x.split(sections, dim)
        tparts = tx.split(list(sections), dim)
        assert len(parts) == len(tparts)
        for p, tp in zip(parts, tparts):
            _same(p, tp)
    with pytest.raises(ValueError):
        x.split([2, 2], 1)


def _getitem_cases(t: Callable[[Any], Any], m: Callable[[Any], Any]) -> list[tuple[str, Any]]:
    return [
        ('pair', (t([0, 1]), t([1, 2]))),
        ('triple', (t([0, 1]), t([1, 2]), t([3, 0]))),
        ('slice_tensor', (slice(None), t([0, 2]))),
        ('slice_slice_tensor', (slice(None), slice(None), t([0, 3, 3]))),
        ('nonadjacent', (t([0, 1]), slice(None), t([1, 2]))),
        ('int_then_tensor', (slice(None), 1, t([0, 2]))),
        ('int_slice_tensor', (0, slice(None), t([0, 2]))),
        ('broadcast_2d', (t([[0], [1]]), t([[0, 1, 2]]))),
        ('negative', (t([-1, 0]), t([-3, 2]))),
        ('mask_slice', (m([True, False]), slice(1, None))),
        ('mask_2d', (m([[True, False, True], [False, False, True]]),)),
        ('slice_mask_2d', (slice(None), m([[True, False, True, False]] * 3))),
        ('mask_tensor', (m([True, True]), t([0, 2]))),
        ('ellipsis_tensor', (Ellipsis, t([1]))),
        ('none_tensor', (None, t([1, 0]))),
        ('single_axis0', (t([1, 0, 1]),)),
        ('scalar_tensor_tensor', (t(1), t([0, 2]))),
        ('none_between', (slice(None), t([0, 1]), None, t([1, 2]))),
    ]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('case', range(len(_getitem_cases(lambda v: v, lambda v: v))))
def test_advanced_getitem_like_torch(device: str, case: int) -> None:
    _, idx = _getitem_cases(lambda v: _i64(v, device), lambda v: Tensor(v, device=device))[case]
    _, tidx = _getitem_cases(torch.tensor, torch.tensor)[case]
    x = Tensor.arange(24, device=device).reshape(2, 3, 4)
    tx = torch.arange(24).reshape(2, 3, 4)
    _same(x[idx], tx[tidx])


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_advanced_getitem_rank4_placement_like_torch(device: str) -> None:
    x = Tensor.arange(360, device=device).reshape(6, 3, 4, 5)
    tx = torch.arange(360).reshape(6, 3, 4, 5)
    i, ti = _i64([0, 1], device), torch.tensor([0, 1])
    j, tj = _i64([1, 2], device), torch.tensor([1, 2])
    _same(x[:, i, 0, j], tx[:, ti, 0, tj])
    _same(x[:, i, None, j], tx[:, ti, None, tj])
    _same(x[:, i, :, j], tx[:, ti, :, tj])
    _same(x[-1, i, ..., j], tx[-1, ti, ..., tj])


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_advanced_getitem_errors_like_torch(device: str) -> None:
    x = Tensor.arange(24, device=device).reshape(2, 3, 4)
    with pytest.raises(IndexError):  # Raised by the kernel: on async devices when the result is read
        x[_i64([0], device), _i64([3], device)].tolist()  # Out of bounds on axis 1 must not wrap into the next row
    with pytest.raises(IndexError):
        x[_i64([0, 1], device), _i64([0, 1, 2], device)]  # Not broadcastable
    with pytest.raises(IndexError):
        x[Tensor([0.5], device=device)]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_advanced_getitem_backward_like_torch(device: str) -> None:
    for make_idx in (lambda t: (t([0, 2, 0]), t([1, 3, 1])), lambda t: (slice(None), t([0, 2, 2]))):
        w = Tensor.arange(12.0, device=device).reshape(3, 4)
        w.requires_grad = True
        tw = torch.arange(12.0).reshape(3, 4).requires_grad_()
        w[make_idx(lambda v: _i64(v, device))].sum().backward()
        tw[make_idx(torch.tensor)].sum().backward()
        _same(w.grad, tw.grad)


def _setitem_cases(t: Callable[[Any], Any], m: Callable[[Any], Any], z: Any) -> list[tuple[Any, Any]]:
    return [
        (m(z > 5), 0),
        (m([False, True]), -1),
        (t([0, 1]), 7),
        ([1], 9),
        ((t([0, 1]), t([2, 0])), -5),
        ((slice(None), t([0, 2])), 3),
        ((t([0, 1]), slice(None), t([1, 2])), t([[10, 11, 12], [20, 21, 22]])),
        ((t([-1]), t([-1]), t([-1])), 100),
        ((Ellipsis, m([True, False, False, True])), 0),
        ((slice(None), 0, t([1, 3])), t([[1, 2], [3, 4]])),
        ((t([0, 0]), t([1, 1])), 42),  # Duplicate indices
    ]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('case', range(11))
def test_advanced_setitem_like_torch(device: str, case: int) -> None:
    x = Tensor.arange(24, device=device).reshape(2, 3, 4)
    tx = torch.arange(24).reshape(2, 3, 4)
    idx, val = _setitem_cases(lambda v: _i64(v, device), lambda v: v if isinstance(v, Tensor) else Tensor(v, device=device), x)[case]
    tidx, tval = _setitem_cases(torch.tensor, lambda v: v if isinstance(v, torch.Tensor) else torch.tensor(v), tx)[case]
    x[idx] = val
    tx[tidx] = tval
    _same(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_advanced_setitem_writes_through_views_like_torch(device: str) -> None:
    x = Tensor.arange(24, device=device).reshape(2, 3, 4)
    tx = torch.arange(24).reshape(2, 3, 4)
    x.transpose(0, 2)[_i64([0, 3], device)] = -1
    tx.transpose(0, 2)[torch.tensor([0, 3])] = -1
    _same(x, tx)
    row, trow = x[1], tx[1]
    row[row > 15] = 0
    trow[trow > 15] = 0
    _same(x, tx)
    y = Tensor.arange(10.0, device=device)
    ty = torch.arange(10.0)
    y[y > 4] = y[y > 4] * 10
    ty[ty > 4] = ty[ty > 4] * 10
    _same(y, ty)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_advanced_setitem_out_of_bounds_raises(device: str) -> None:
    x = Tensor.arange(24, device=device).reshape(2, 3, 4)
    with pytest.raises(IndexError):  # Raised by the scatter kernel: on async devices at the next sync
        x[_i64([5], device)] = 0
        x.sync()
    with pytest.raises(IndexError):
        x[_i64([0], device), _i64([3], device)] = 0
        x.sync()
    assert x.tolist() == torch.arange(24).reshape(2, 3, 4).tolist()
