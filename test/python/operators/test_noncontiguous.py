# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..common import *

_BASE_SHAPE = (4, 6, 8)

_VIEW_NAMES = ('transpose01', 'transpose_last', 'permute', 'step_slice', 'sub_block', 'narrow', 'view_slice', 'row_step')


def _view(name: str, x: Tensor, tx: torch.Tensor) -> tuple[Tensor, torch.Tensor]:
    match name:
        case 'transpose01':
            return x.transpose(0, 1), tx.transpose(0, 1)
        case 'transpose_last':
            return x.transpose(-1, -2), tx.transpose(-1, -2)
        case 'permute':
            return x.permute((2, 1, 0)), tx.permute(2, 1, 0)
        case 'step_slice':
            return x[:, ::2], tx[:, ::2]
        case 'sub_block':
            return x[1:, 2:5, 3:7], tx[1:, 2:5, 3:7]
        case 'narrow':
            return x.narrow(-1, 1, 5), tx.narrow(-1, 1, 5)
        case 'view_slice':
            return x.view_slice(1, 1, 2, 2), tx[:, 1:4:2]
        case 'row_step':
            return x[::2, 1], tx[::2, 1]
    raise ValueError(name)


def _base(dt: dtype.DType, device: str, low: float, high: float) -> tuple[Tensor, torch.Tensor]:
    x = uniform_tensor(_BASE_SHAPE, low=low, high=high, dtype=dt, device=device)
    return x, totorch(x)


def _same(r: Tensor, ref: torch.Tensor, dt: dtype.DType) -> None:
    assert r.shape == tuple(ref.shape)
    if dt.is_integer() or ref.dtype == torch.bool:
        assert r.tolist() == ref.tolist()
    else:
        assert_close_mag_torch(r, ref, dt)


_UNARY: tuple[tuple[str, Callable], ...] = (
    ('neg', torch.neg),
    ('abs', torch.abs),
    ('exp', torch.exp),
    ('log', torch.log),
    ('sqrt', torch.sqrt),
    ('sqr', torch.square),
    ('tanh', torch.tanh),
    ('sigmoid', torch.sigmoid),
    ('relu', torch.relu),
    ('gelu', F.gelu),
    ('floor', torch.floor),
    ('rcp', torch.reciprocal),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('view', _VIEW_NAMES)
@pytest.mark.parametrize('name, ref', _UNARY, ids=[c[0] for c in _UNARY])
def test_unary_on_view(device: str, view: str, name: str, ref: Callable) -> None:
    x, tx = _base(dtype.float32, device, 0.1, 2.0)
    v, tv = _view(view, x, tx)
    assert not v.is_contiguous
    _same(getattr(v, name)(), ref(tv), dtype.float32)
    xb, tb = x.clone(), tx.clone()
    vb, tvb = _view(view, xb, tb)
    getattr(vb, f'{name}_')()
    tvb.copy_(ref(tvb))
    _same(xb, tb, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('view', _VIEW_NAMES)
@pytest.mark.parametrize('dt', [dtype.float32, dtype.float16, dtype.int32], ids=['float32', 'float16', 'int32'])
@pytest.mark.parametrize('name, fn', [('add', lambda a, b: a + b), ('sub', lambda a, b: a - b), ('mul', lambda a, b: a * b), ('truediv', lambda a, b: a / b)], ids=['add', 'sub', 'mul', 'truediv'])
def test_binary_on_views(device: str, view: str, dt: dtype.DType, name: str, fn: Callable) -> None:
    low, high = (1, 50) if dt.is_integer() else (0.5, 2.0)
    x, tx = _base(dt, device, low, high)
    y, ty = _base(dt, device, low, high)
    v, tv = _view(view, x, tx)
    u, tu = _view(view, y, ty)
    c = uniform_tensor(v.shape, low=low, high=high, dtype=dt, device=device)
    tc = totorch(c)
    row = uniform_tensor((v.shape[-1],), low=low, high=high, dtype=dt, device=device)
    trow = totorch(row)
    _same(fn(v, u), fn(tv, tu), dt)
    _same(fn(v, c), fn(tv, tc), dt)
    _same(fn(c, v), fn(tc, tv), dt)
    _same(fn(v, row), fn(tv, trow), dt)
    _same(fn(row, v), fn(trow, tv), dt)
    if name == 'truediv' and dt.is_integer():
        return
    xb, tb = x.clone(), tx.clone()
    vb, tvb = _view(view, xb, tb)
    getattr(vb, f'__i{name}__')(u)
    getattr(tvb, f'__i{name}__')(tu)
    _same(xb, tb, dt)


_REDUCE_DIMS = (None, 0, 1, 2, -1)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('view', _VIEW_NAMES)
@pytest.mark.parametrize('op', ['sum', 'mean', 'prod', 'max', 'min', 'argmax', 'argmin'])
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_on_view(device: str, view: str, op: str, keepdim: bool) -> None:
    x, tx = _base(dtype.float32, device, 0.8, 1.25)
    v, tv = _view(view, x, tx)
    for dim in _REDUCE_DIMS:
        if dim is not None and dim >= v.rank:
            continue
        r = call_reduction(v, op, dim, keepdim)
        if dim is None:
            t = getattr(tv, op)()
        else:
            t = getattr(tv, op)(dim=dim, keepdim=keepdim)
        if not isinstance(t, torch.Tensor):
            t = t[0]
        if op.startswith('arg'):
            assert r.shape == tuple(t.shape)
            assert r.tolist() == t.tolist()
        else:
            assert_close_mag_torch(r, t, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('view', _VIEW_NAMES)
@pytest.mark.parametrize('op', ['sum', 'max', 'min', 'argmax', 'argmin'])
def test_integer_reduction_on_view(device: str, view: str, op: str) -> None:
    x, tx = _base(dtype.int32, device, 0, 5)
    v, tv = _view(view, x, tx)
    tv64 = tv.to(torch.int64)
    for dim in _REDUCE_DIMS:
        if dim is not None and dim >= v.rank:
            continue
        for keepdim in (False, True):
            r = call_reduction(v, op, dim, keepdim)
            t = getattr(tv64, op)() if dim is None else getattr(tv64, op)(dim=dim, keepdim=keepdim)
            if not isinstance(t, torch.Tensor):
                t = t[0]
            assert r.shape == tuple(t.shape)
            assert r.tolist() == t.tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('view', _VIEW_NAMES)
def test_softmax_on_view(device: str, view: str) -> None:
    x, tx = _base(dtype.float32, device, -3.0, 3.0)
    v, tv = _view(view, x, tx)
    for dim in range(-v.rank, v.rank):
        assert_close_mag_torch(v.softmax(dim), torch.softmax(tv, dim), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('view', _VIEW_NAMES)
@pytest.mark.parametrize('op', ['cusum', 'cuprod', 'cumax', 'cumin'])
def test_cumulative_on_view(device: str, view: str, op: str) -> None:
    x, tx = _base(dtype.float32, device, 0.8, 1.25)
    v, tv = _view(view, x, tx)
    tname = {'cusum': 'cumsum', 'cuprod': 'cumprod', 'cumax': 'cummax', 'cumin': 'cummin'}[op]
    for dim in range(-v.rank, v.rank):
        r = getattr(v, op)(dim)
        t = getattr(tv, tname)(dim)
        if isinstance(r, tuple):
            assert_close_mag_torch(r[0], t.values, dtype.float32)
            assert r[1].tolist() == t.indices.tolist()
        else:
            assert_close_mag_torch(r, t, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('view', _VIEW_NAMES)
@pytest.mark.parametrize('dst', [dtype.float16, dtype.bfloat16, dtype.int32, dtype.boolean], ids=['float16', 'bfloat16', 'int32', 'boolean'])
def test_cast_on_view(device: str, view: str, dst: dtype.DType) -> None:
    x, tx = _base(dtype.float32, device, -3.0, 3.0)
    v, tv = _view(view, x, tx)
    r = v.cast(dst)
    t = tv.to(totorch_dtype(dst))
    assert r.dtype == dst
    _same(r, t, dst)
    xi, txi = _base(dtype.int32, device, -50, 50)
    vi, tvi = _view(view, xi, txi)
    _same(vi.cast(dtype.float32), tvi.to(torch.float32), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('view', _VIEW_NAMES)
@pytest.mark.parametrize('dt', [dtype.float32, dtype.int64, dtype.boolean], ids=['float32', 'int64', 'boolean'])
def test_contiguous_and_clone_of_view(device: str, view: str, dt: dtype.DType) -> None:
    if dt == dtype.boolean:
        x = Tensor.bernoulli(_BASE_SHAPE, p=0.5, device=device)
        tx = totorch(x)
    else:
        x, tx = _base(dt, device, -50, 50)
    v, tv = _view(view, x, tx)
    c = v.contiguous()
    assert c.is_contiguous
    assert c.strides == tuple(tv.contiguous().stride())
    _same(c, tv, dt)
    k = v.clone()
    assert k.is_contiguous
    _same(k, tv, dt)


_MATMUL_VIEWS: tuple[str, ...] = ('rhs_T', 'lhs_T', 'both_T', 'batched_permuted', 'rhs_step_slice', 'rhs_batched_T', 'lhs_narrow')


def _matmul_operands(name: str, device: str) -> tuple[Tensor, Tensor, torch.Tensor, torch.Tensor]:
    def u(*shape):
        m = uniform_tensor(shape, low=-1.0, high=1.0, device=device)
        return m, totorch(m)

    match name:
        case 'rhs_T':
            a, ta = u(3, 4)
            w, tw = u(5, 4)
            return a, w.T, ta, tw.T
        case 'lhs_T':
            a, ta = u(4, 3)
            b, tb = u(4, 5)
            return a.T, b, ta.T, tb
        case 'both_T':
            a, ta = u(4, 3)
            b, tb = u(5, 4)
            return a.T, b.T, ta.T, tb.T
        case 'batched_permuted':
            a, ta = u(3, 2, 4)
            b, tb = u(4, 5)
            return a.permute((1, 0, 2)), b, ta.permute(1, 0, 2), tb
        case 'rhs_step_slice':
            a, ta = u(3, 4)
            b, tb = u(4, 10)
            return a, b[:, ::2], ta, tb[:, ::2]
        case 'rhs_batched_T':
            a, ta = u(2, 3, 4)
            b, tb = u(2, 5, 4)
            return a, b.transpose(-1, -2), ta, tb.transpose(-1, -2)
        case 'lhs_narrow':
            a, ta = u(3, 9)
            b, tb = u(4, 5)
            return a.narrow(1, 2, 4), b, ta.narrow(1, 2, 4), tb
    raise ValueError(name)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name', _MATMUL_VIEWS)
def test_matmul_on_views(device: str, name: str) -> None:
    a, b, ta, tb = _matmul_operands(name, device)
    assert not (a.is_contiguous and b.is_contiguous)
    assert_close_mag_torch(a @ b, ta @ tb, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('x_shape, w_shape', [((3, 4), (5, 4)), ((2, 3, 4), (5, 4)), ((2, 3, 4), (2, 5, 4))])
def test_matmul_transposed_rhs_backward(device: str, x_shape, w_shape) -> None:
    x = uniform_tensor(x_shape, low=-1.0, high=1.0, device=device)
    w = uniform_tensor(w_shape, low=-1.0, high=1.0, device=device)
    tx, tw = totorch(x).clone().requires_grad_(True), totorch(w).clone().requires_grad_(True)
    x.requires_grad = True
    w.requires_grad = True
    y = x @ w.transpose(-1, -2)
    ty = tx @ tw.transpose(-1, -2)
    assert_close_mag_torch(y, ty.detach(), dtype.float32)
    s = random_tensor(y.shape, dtype.float32, device)
    (y * s).sum().backward()
    (ty * totorch(s)).sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)
    assert_close_mag_torch(w.grad, tw.grad, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_expanded_stride_zero_inputs(device: str) -> None:
    x = uniform_tensor((4, 1, 8), low=0.1, high=2.0, device=device)
    tx = totorch(x)
    e, te = x.expand(4, 6, 8), tx.expand(4, 6, 8)
    other = uniform_tensor((4, 6, 8), low=0.1, high=2.0, device=device)
    tother = totorch(other)
    w = uniform_tensor((8, 3), low=-1.0, high=1.0, device=device)
    tw = totorch(w)
    assert_close_mag_torch(e.exp(), te.exp(), dtype.float32)
    assert_close_mag_torch(e + other, te + tother, dtype.float32)
    assert_close_mag_torch(other * e, tother * te, dtype.float32)
    for dim in (None, 0, 1, 2):
        r = e.sum() if dim is None else e.sum(dim=dim)
        t = te.sum() if dim is None else te.sum(dim=dim)
        assert_close_mag_torch(r, t, dtype.float32)
    assert_close_mag_torch(e @ w, te @ tw, dtype.float32)
    assert_close_mag_torch(e.contiguous(), te.contiguous(), dtype.float32)
    assert_close_mag_torch(e.cast(dtype.float16), te.to(torch.float16), dtype.float16)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_softmax_on_expanded_input(device: str) -> None:
    x = uniform_tensor((4, 1, 8), low=-3.0, high=3.0, device=device)
    e = x.expand(4, 6, 8)
    assert_close_mag_torch(e.softmax(), torch.softmax(totorch(x).expand(4, 6, 8), -1), dtype.float32)
