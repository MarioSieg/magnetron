# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch
import torch.nn.functional as F

import magnetron as mag

from ..common import *

_SHAPES: tuple[tuple[int, ...], ...] = ((3,), (2, 5), (4, 3, 2), (1, 6, 1, 3), (2, 32, 33))


def _leaf(shape: tuple[int, ...], low: float, high: float, device: str) -> tuple[Tensor, torch.Tensor]:
    x = uniform_tensor(shape, low, high, dtype.float32, device)
    tx = totorch(x).clone().requires_grad_(True)
    x.requires_grad = True
    return x, tx


def _backward(y: Tensor, ty: torch.Tensor, device: str) -> None:
    assert y.shape == tuple(ty.shape)
    assert_close_mag_torch(y, ty.detach(), dtype.float32)
    w = random_tensor(y.shape, dtype.float32, device)
    (y * w).sum().backward()
    (ty * totorch(w)).sum().backward()


def _assert_grad(x: Tensor, tx: torch.Tensor) -> None:
    assert x.grad is not None
    assert tx.grad is not None
    assert x.grad.shape == tuple(tx.grad.shape)
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)


_UNARY: tuple[tuple[str, float, float, Callable[[torch.Tensor], torch.Tensor]], ...] = (
    ('abs', -2.0, 2.0, torch.abs),
    ('neg', -2.0, 2.0, torch.neg),
    ('sqr', -2.0, 2.0, torch.square),
    ('sin', -3.0, 3.0, torch.sin),
    ('cos', -3.0, 3.0, torch.cos),
    ('tan', -1.2, 1.2, torch.tan),
    ('sinh', -2.0, 2.0, torch.sinh),
    ('cosh', -2.0, 2.0, torch.cosh),
    ('tanh', -3.0, 3.0, torch.tanh),
    ('atan', -3.0, 3.0, torch.atan),
    ('asinh', -3.0, 3.0, torch.asinh),
    ('erf', -3.0, 3.0, torch.erf),
    ('erfc', -3.0, 3.0, torch.erfc),
    ('exp', -3.0, 3.0, torch.exp),
    ('exp2', -3.0, 3.0, torch.exp2),
    ('expm1', -3.0, 3.0, torch.expm1),
    ('log', 0.1, 4.0, torch.log),
    ('log10', 0.1, 4.0, torch.log10),
    ('log2', 0.1, 4.0, torch.log2),
    ('log1p', -0.5, 4.0, torch.log1p),
    ('sqrt', 0.1, 4.0, torch.sqrt),
    ('rsqrt', 0.1, 4.0, torch.rsqrt),
    ('rcp', 0.1, 4.0, torch.reciprocal),
    ('asin', -0.9, 0.9, torch.asin),
    ('acos', -0.9, 0.9, torch.acos),
    ('atanh', -0.9, 0.9, torch.atanh),
    ('acosh', 1.1, 4.0, torch.acosh),
    ('sigmoid', -5.0, 5.0, torch.sigmoid),
    ('hard_sigmoid', -5.0, 5.0, F.hardsigmoid),
    ('silu', -5.0, 5.0, F.silu),
    ('relu', -2.0, 2.0, torch.relu),
    ('gelu', -3.0, 3.0, F.gelu),
    ('gelu_approx', -3.0, 3.0, lambda t: F.gelu(t, approximate='tanh')),
    ('softmax', -3.0, 3.0, lambda t: torch.softmax(t, -1)),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape', _SHAPES)
@pytest.mark.parametrize('name, low, high, ref', _UNARY, ids=[c[0] for c in _UNARY])
def test_unary_backward(device: str, shape: tuple[int, ...], name: str, low: float, high: float, ref: Callable) -> None:
    x, tx = _leaf(shape, low, high, device)
    _backward(getattr(x, name)(), ref(tx), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_chained_unary_backward(device: str) -> None:
    x, tx = _leaf((4, 7), 0.2, 2.0, device)
    y = ((x.log().sqr() + x.sqrt().exp()).tanh() * x.sigmoid()).rsqrt()
    ty = ((tx.log().square() + tx.sqrt().exp()).tanh() * tx.sigmoid()).rsqrt()
    _backward(y, ty, device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_multi_path_backward(device: str) -> None:
    x, tx = _leaf((3, 5), -2.0, 2.0, device)
    y = (x.tanh() * x.sigmoid() + x.exp() / (1.0 + x.sqr())).sum(dim=-1).mean()
    ty = (tx.tanh() * tx.sigmoid() + tx.exp() / (1.0 + tx.square())).sum(dim=-1).mean()
    assert_close_mag_torch(y, ty.detach(), dtype.float32)
    y.backward()
    ty.backward()
    _assert_grad(x, tx)


_BINARY: tuple[tuple[str, Callable, tuple[float, float], tuple[float, float]], ...] = (
    ('add', lambda a, b: a + b, (-2.0, 2.0), (-2.0, 2.0)),
    ('sub', lambda a, b: a - b, (-2.0, 2.0), (-2.0, 2.0)),
    ('mul', lambda a, b: a * b, (-2.0, 2.0), (-2.0, 2.0)),
    ('truediv', lambda a, b: a / b, (-2.0, 2.0), (0.5, 2.0)),
    ('pow', lambda a, b: a**b, (0.5, 2.0), (-2.0, 2.0)),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name', ['min', 'max'])
@pytest.mark.parametrize('shape', _SHAPES)
def test_binary_minmax_backward(device: str, name: str, shape: tuple[int, ...]) -> None:
    x, tx = _leaf(shape, -2.0, 2.0, device)
    y, ty = _leaf(shape, -2.0, 2.0, device)
    ref = torch.minimum if name == 'min' else torch.maximum
    _backward(getattr(x, name)(y), ref(tx, ty), device)
    _assert_grad(x, tx)
    _assert_grad(y, ty)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape', _SHAPES)
@pytest.mark.parametrize('name, fn, xr, yr', _BINARY, ids=[c[0] for c in _BINARY])
def test_binary_backward(device: str, shape: tuple[int, ...], name: str, fn: Callable, xr, yr) -> None:
    x, tx = _leaf(shape, *xr, device)
    y, ty = _leaf(shape, *yr, device)
    _backward(fn(x, y), fn(tx, ty), device)
    _assert_grad(x, tx)
    _assert_grad(y, ty)


_SCALAR_BINARY: tuple[tuple[str, Callable, tuple[float, float]], ...] = (
    ('add', lambda a: a + 1.5, (-2.0, 2.0)),
    ('radd', lambda a: 1.5 + a, (-2.0, 2.0)),
    ('sub', lambda a: a - 1.5, (-2.0, 2.0)),
    ('rsub', lambda a: 1.5 - a, (-2.0, 2.0)),
    ('mul', lambda a: a * 1.5, (-2.0, 2.0)),
    ('rmul', lambda a: 1.5 * a, (-2.0, 2.0)),
    ('truediv', lambda a: a / 1.5, (-2.0, 2.0)),
    ('rtruediv', lambda a: 1.5 / a, (0.5, 2.0)),
    ('pow', lambda a: a**2.5, (0.5, 2.0)),
    ('rpow', lambda a: 1.5**a, (-2.0, 2.0)),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape', _SHAPES)
@pytest.mark.parametrize('name, fn, xr', _SCALAR_BINARY, ids=[c[0] for c in _SCALAR_BINARY])
def test_scalar_binary_backward(device: str, shape: tuple[int, ...], name: str, fn: Callable, xr) -> None:
    x, tx = _leaf(shape, *xr, device)
    _backward(fn(x), fn(tx), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('fn', [lambda a: a * a, lambda a: a + a, lambda a: a - a * a, lambda a: a / (a + 3.0)], ids=['mul', 'add', 'sub_mul', 'div'])
def test_same_operand_backward(device: str, fn: Callable) -> None:
    x, tx = _leaf((3, 4), -2.0, 2.0, device)
    _backward(fn(x), fn(tx), device)
    _assert_grad(x, tx)


_MATMUL: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((3, 4), (4, 5)),
    ((1, 4), (4, 1)),
    ((7, 16), (16, 9)),
    ((2, 3, 4), (2, 4, 5)),
    ((2, 3, 4), (4, 5)),
    ((1, 3, 4), (2, 4, 5)),
    ((2, 1, 3, 4), (1, 2, 4, 5)),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shapes', _MATMUL, ids=[f'{a}x{b}' for a, b in _MATMUL])
def test_matmul_backward(device: str, shapes) -> None:
    a, ta = _leaf(shapes[0], -1.0, 1.0, device)
    b, tb = _leaf(shapes[1], -1.0, 1.0, device)
    _backward(a @ b, ta @ tb, device)
    _assert_grad(a, ta)
    _assert_grad(b, tb)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('x_shape', [(3, 4), (2, 3, 4), (4,)])
def test_matmul_transposed_rhs_backward(device: str, x_shape) -> None:
    x, tx = _leaf(x_shape, -1.0, 1.0, device)
    w, tw = _leaf((6, 4), -1.0, 1.0, device)
    _backward(x @ w.T, tx @ tw.T, device)
    _assert_grad(x, tx)
    _assert_grad(w, tw)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_matmul_transposed_lhs_backward(device: str) -> None:
    a, ta = _leaf((4, 3), -1.0, 1.0, device)
    b, tb = _leaf((4, 5), -1.0, 1.0, device)
    _backward(a.T @ b, ta.T @ tb, device)
    _assert_grad(a, ta)
    _assert_grad(b, tb)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_matmul_batched_transposed_rhs_backward(device: str) -> None:
    x, tx = _leaf((2, 3, 4), -1.0, 1.0, device)
    w, tw = _leaf((2, 5, 4), -1.0, 1.0, device)
    _backward(x @ w.transpose(-1, -2), tx @ tw.transpose(-1, -2), device)
    _assert_grad(x, tx)
    _assert_grad(w, tw)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_linear_with_bias_backward(device: str) -> None:
    x, tx = _leaf((8, 16), -1.0, 1.0, device)
    w, tw = _leaf((4, 16), -1.0, 1.0, device)
    b, tb = _leaf((4,), -1.0, 1.0, device)
    _backward(x @ w.T + b, tx @ tw.T + tb, device)
    _assert_grad(x, tx)
    _assert_grad(w, tw)
    _assert_grad(b, tb)


_EINSUM: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = (
    ('ij,kj->ik', ((3, 4), (5, 4))),
    ('bij,bjk->bik', ((2, 3, 4), (2, 4, 5))),
    ('ij->', ((3, 4),)),
    ('ij->ji', ((3, 4),)),
    ('i,j->ij', ((3,), (4,))),
    ('bnd,dh->bnh', ((2, 3, 4), (4, 5))),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('equation, shapes', _EINSUM, ids=[c[0] for c in _EINSUM])
def test_einsum_backward(device: str, equation: str, shapes) -> None:
    leaves = [_leaf(s, -1.0, 1.0, device) for s in shapes]
    y = Tensor.einsum(equation, *[m for m, _ in leaves])
    ty = torch.einsum(equation, *[t for _, t in leaves])
    _backward(y, ty, device)
    for m, t in leaves:
        _assert_grad(m, t)


_REDUCE_DIMS = (None, 0, 1, -1, (0, 2), (1, 2), (0, 1, 2))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('op', ['sum', 'mean'])
@pytest.mark.parametrize('keepdim', [False, True])
@pytest.mark.parametrize('dim', _REDUCE_DIMS, ids=[str(d) for d in _REDUCE_DIMS])
def test_reduction_backward(device: str, op: str, keepdim: bool, dim) -> None:
    x, tx = _leaf((3, 4, 5), -2.0, 2.0, device)
    if dim is None:
        y = getattr(x, op)()
        ty = getattr(tx, op)()
    else:
        y = getattr(x, op)(dim=dim, keepdim=keepdim)
        ty = getattr(tx, op)(dim=dim, keepdim=keepdim)
    _backward(y, ty, device)
    _assert_grad(x, tx)


_VIEW_CASES: tuple[tuple[str, Callable], ...] = (
    ('transpose', lambda x: x.transpose(0, 1)),
    ('transpose_last', lambda x: x.transpose(-1, -2)),
    ('permute', lambda x: x.permute((2, 0, 1))),
    ('slice', lambda x: x[1:, ::2, 1:4]),
    ('narrow', lambda x: x.narrow(-1, 1, 3)),
    ('select', lambda x: x.select(1, 2)),
    ('index', lambda x: x[2]),
    ('view', lambda x: x.view(-1)),
    ('reshape', lambda x: x.reshape((5, -1))),
    ('flatten', lambda x: x.flatten(start_dim=1)),
    ('unsqueeze', lambda x: x.unsqueeze(1)),
    ('squeeze', lambda x: x.unsqueeze(0).squeeze(0)),
    ('movedim', lambda x: x.movedim(0, 2)),
    ('split', lambda x: x.split(2, dim=1)[1]),
    ('flip', lambda x: x.flip(0, 2)),
    ('repeat', lambda x: x.repeat(2, 1, 3)),
    ('repeat_leading', lambda x: x.repeat(2, 1, 1, 2)),
    ('repeat_interleave_int', lambda x: x.repeat_interleave(3, dim=1)),
    ('repeat_interleave_flat', lambda x: x.repeat_interleave(2)),
    ('cusum', lambda x: x.cumsum(1) if isinstance(x, torch.Tensor) else x.cusum(1)),
    ('cusum_last', lambda x: x.cumsum(-1) if isinstance(x, torch.Tensor) else x.cusum(-1)),
    ('pad_constant', lambda x: F.pad(x, [1, 2, 0, 1], value=0.5) if isinstance(x, torch.Tensor) else x.pad([1, 2, 0, 1], value=0.5)),
    ('pad_reflect', lambda x: F.pad(x, [2, 1, 1, 2], mode='reflect') if isinstance(x, torch.Tensor) else x.pad([2, 1, 1, 2], mode='reflect')),
    ('pad_replicate', lambda x: F.pad(x, [1, 3, 2, 0], mode='replicate') if isinstance(x, torch.Tensor) else x.pad([1, 3, 2, 0], mode='replicate')),
    ('topk', lambda x: x.topk(2, dim=1)[0]),
    ('topk_smallest', lambda x: x.topk(3, dim=-1, largest=False)[0]),
    ('sort', lambda x: x.sort(dim=1)[0]),
    ('sort_desc', lambda x: x.sort(dim=-1, descending=True)[0]),
    ('cummax', lambda x: x.cummax(2)[0] if isinstance(x, torch.Tensor) else x.cumax(2)[0]),
    ('cummin', lambda x: x.cummin(0)[0] if isinstance(x, torch.Tensor) else x.cumin(0)[0]),
    ('expand_leading', lambda x: x.expand(2, 3, 4, 5)),
    ('tril', lambda x: x.tril(1)),
    ('triu', lambda x: x.triu(-1)),
    ('clone', lambda x: x.clone()),
    ('contiguous', lambda x: x.transpose(0, 2).contiguous()),
    ('unbind', lambda x: x.unbind(1)[2]),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, fn', _VIEW_CASES, ids=[c[0] for c in _VIEW_CASES])
def test_view_backward(device: str, name: str, fn: Callable) -> None:
    x, tx = _leaf((3, 4, 5), -2.0, 2.0, device)
    _backward(fn(x), fn(tx), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_T_backward(device: str) -> None:
    x, tx = _leaf((4, 6), -2.0, 2.0, device)
    _backward(x.T, tx.T, device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('method', ['expand', 'broadcast'])
def test_expand_size_one_dims_backward(device: str, method: str) -> None:
    x, tx = _leaf((3, 1, 5), -2.0, 2.0, device)
    _backward(getattr(x, method)(3, 4, 5), tx.expand(3, 4, 5), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dim, start, length, step', [(1, 0, 2, 2), (0, 1, 2, 1), (-1, 1, 3, 1), (0, 0, 3, 2)])
def test_view_slice_backward(device: str, dim: int, start: int, length: int, step: int) -> None:
    x, tx = _leaf((5, 6), -2.0, 2.0, device)
    idx = [slice(None)] * 2
    idx[dim] = slice(start, start + (length - 1) * step + 1, step)
    _backward(x.view_slice(dim, start, length, step), tx[tuple(idx)], device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape, strides, offset', [((3, 3), (6, 1), 1), ((6, 4), (1, 6), 0), ((2, 4), (6, 2), 3), ((3, 4), (1, 1), 2)])
def test_strided_view_backward(device: str, shape, strides, offset: int) -> None:
    x, tx = _leaf((4, 6), -2.0, 2.0, device)
    _backward(x.strided_view(shape, strides, offset=offset), torch.as_strided(tx, shape, strides, offset), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dim', [0, 1, -1])
def test_cat_backward(device: str, dim: int) -> None:
    base = [3, 4]
    leaves = []
    for n in (1, 2, 3):
        s = list(base)
        s[dim] = n
        leaves.append(_leaf(tuple(s), -2.0, 2.0, device))
    _backward(Tensor.cat([m for m, _ in leaves], dim), torch.cat([t for _, t in leaves], dim), device)
    for m, t in leaves:
        _assert_grad(m, t)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dim', [0, 1, 2])
def test_stack_backward(device: str, dim: int) -> None:
    leaves = [_leaf((3, 4), -2.0, 2.0, device) for _ in range(3)]
    _backward(Tensor.stack([m for m, _ in leaves], dim), torch.stack([t for _, t in leaves], dim), device)
    for m, t in leaves:
        _assert_grad(m, t)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name', ['hstack', 'vstack', 'dstack'])
def test_stack_aliases_backward(device: str, name: str) -> None:
    leaves = [_leaf((3, 4), -2.0, 2.0, device) for _ in range(2)]
    _backward(getattr(Tensor, name)([m for m, _ in leaves]), getattr(torch, name)([t for _, t in leaves]), device)
    for m, t in leaves:
        _assert_grad(m, t)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape', _SHAPES)
def test_clamp_scalar_backward(device: str, shape) -> None:
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(x.clamp(-1.0, 0.5), torch.clamp(tx, -1.0, 0.5), device)
    _assert_grad(x, tx)
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(x.clamp_min(-0.5), torch.clamp_min(tx, -0.5), device)
    _assert_grad(x, tx)
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(x.clamp_max(0.5), torch.clamp_max(tx, 0.5), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_clamp_tensor_backward(device: str) -> None:
    x, tx = _leaf((4, 5), -2.0, 2.0, device)
    lo = uniform_tensor((4, 5), low=-1.5, high=-0.5, device=device)
    hi = uniform_tensor((4, 5), low=0.5, high=1.5, device=device)
    _backward(x.clamp(lo, hi), torch.clamp(tx, totorch(lo), totorch(hi)), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape', [(5,), (3, 4), (2, 3, 4)])
def test_masked_fill_backward(device: str, shape) -> None:
    x, tx = _leaf(shape, -2.0, 2.0, device)
    mask = Tensor.bernoulli(shape, p=0.4, device=device)
    _backward(x.masked_fill(mask, 0.5), tx.masked_fill(totorch(mask), 0.5), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape', [(5,), (3, 4), (2, 3, 4)])
def test_where_backward(device: str, shape) -> None:
    cond = Tensor.bernoulli(shape, p=0.5, device=device)
    tc = totorch(cond)
    x, tx = _leaf(shape, -2.0, 2.0, device)
    y, ty = _leaf(shape, -2.0, 2.0, device)
    _backward(Tensor.where(cond, x, y), torch.where(tc, tx, ty), device)
    _assert_grad(x, tx)
    _assert_grad(y, ty)
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(Tensor.where(cond, x, 0.25), torch.where(tc, tx, torch.tensor(0.25)), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dim', [0, 1, 2])
def test_gather_backward(device: str, dim: int) -> None:
    x, tx = _leaf((3, 4, 5), -2.0, 2.0, device)
    idx_shape = [3, 4, 5]
    idx_shape[dim] = 7
    tidx = torch.randint(0, tx.shape[dim], tuple(idx_shape))
    idx = Tensor(tidx.tolist(), device=device)
    _backward(x.gather(dim, idx), tx.gather(dim, tidx), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('idx_shape', [(4,), (2, 3), (2, 2, 2)])
def test_embedding_backward(device: str, idx_shape) -> None:
    w, tw = _leaf((7, 5), -2.0, 2.0, device)
    tidx = torch.randint(0, 7, idx_shape)
    idx = Tensor(tidx.tolist(), device=device)
    _backward(w.embedding(idx), F.embedding(tidx, tw), device)
    _assert_grad(w, tw)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('inner', [dtype.float16, dtype.bfloat16])
def test_cast_roundtrip_backward(device: str, inner: dtype.DType) -> None:
    x, tx = _leaf((4, 6), -2.0, 2.0, device)
    _backward(x.cast(inner).cast(dtype.float32), tx.to(totorch_dtype(inner)).to(torch.float32), device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_lerp_backward(device: str) -> None:
    a, ta = _leaf((3, 4), -2.0, 2.0, device)
    b, tb = _leaf((3, 4), -2.0, 2.0, device)
    _backward(a.lerp(b, 0.3), torch.lerp(ta, tb, 0.3), device)
    _assert_grad(a, ta)
    _assert_grad(b, tb)
    a, ta = _leaf((3, 4), -2.0, 2.0, device)
    b, tb = _leaf((3, 4), -2.0, 2.0, device)
    w, tw = _leaf((3, 4), 0.0, 1.0, device)
    _backward(a.lerp(b, w), torch.lerp(ta, tb, tw), device)
    _assert_grad(a, ta)
    _assert_grad(b, tb)
    _assert_grad(w, tw)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_outer_backward(device: str) -> None:
    a, ta = _leaf((5,), -2.0, 2.0, device)
    b, tb = _leaf((3,), -2.0, 2.0, device)
    _backward(a.outer(b), torch.outer(ta, tb), device)
    _assert_grad(a, ta)
    _assert_grad(b, tb)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_no_grad_blocks_recording(device: str) -> None:
    x, tx = _leaf((3, 4), -2.0, 2.0, device)
    with mag.no_grad():
        y = x.exp()
    assert not y.requires_grad
    z = (x.exp() * 2.0).sum()
    z.backward()
    tz = (tx.exp() * 2.0).sum()
    tz.backward()
    _assert_grad(x, tx)



@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shapes', [((4,), (4, 5)), ((3, 4), (4,)), ((4,), (4,)), ((1,), (1, 3)), ((6, 1), (1,))])
def test_matmul_vector_operands_backward(device: str, shapes) -> None:
    a, ta = _leaf(shapes[0], -1.0, 1.0, device)
    b, tb = _leaf(shapes[1], -1.0, 1.0, device)
    _backward(a @ b, ta @ tb, device)
    _assert_grad(a, ta)
    _assert_grad(b, tb)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name', ['min', 'max'])
def test_binary_minmax_ties_backward(device: str, name: str) -> None:
    x, tx = _leaf((3, 4), -2.0, 2.0, device)
    y = x.detach().clone()
    ty = tx.detach().clone().requires_grad_(True)
    y.requires_grad = True
    ref = torch.minimum if name == 'min' else torch.maximum
    _backward(getattr(x, name)(y), ref(tx, ty), device)
    _assert_grad(x, tx)
    _assert_grad(y, ty)
    x, tx = _leaf((3, 4), -2.0, 2.0, device)
    _backward(getattr(x, name)(x), ref(tx, tx), device)
    _assert_grad(x, tx)


_REDUCE_MINMAX_DIMS = (None, 0, 1, -1, (0, 2), (1, 2))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('op', ['max', 'min'])
@pytest.mark.parametrize('keepdim', [False, True])
@pytest.mark.parametrize('dim', _REDUCE_MINMAX_DIMS, ids=[str(d) for d in _REDUCE_MINMAX_DIMS])
def test_reduce_minmax_backward(device: str, op: str, keepdim: bool, dim) -> None:
    x, tx = _leaf((3, 4, 5), -2.0, 2.0, device)
    tref = torch.amax if op == 'max' else torch.amin
    if dim is None:
        y = getattr(x, op)()
        ty = tref(tx)
    else:
        y = getattr(x, op)(dim, keepdim=keepdim)
        ty = tref(tx, dim=dim, keepdim=keepdim)
    _backward(y, ty, device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('op', ['max', 'min'])
@pytest.mark.parametrize('dim', [None, 0, 1, -1])
def test_reduce_minmax_ties_split_evenly_backward(device: str, op: str, dim) -> None:
    tx = torch.tensor([[1.0, 3.0, 3.0, 0.0], [2.0, 2.0, 2.0, 2.0], [-1.0, 5.0, -1.0, 5.0]], requires_grad=True)
    x = Tensor(tx.tolist(), device=device)
    x.requires_grad = True
    tref = torch.amax if op == 'max' else torch.amin
    if dim is None:
        y, ty = getattr(x, op)(), tref(tx)
    else:
        y, ty = getattr(x, op)(dim), tref(tx, dim=dim)
    _backward(y, ty, device)
    _assert_grad(x, tx)


_PROD_DIMS = (None, 0, 1, -1, (0, 2), (1, 2), (0, 1, 2))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('keepdim', [False, True])
@pytest.mark.parametrize('dim', _PROD_DIMS, ids=[str(d) for d in _PROD_DIMS])
@pytest.mark.parametrize('zeros', ['none', 'single', 'double'])
def test_prod_backward(device: str, keepdim: bool, dim, zeros: str) -> None:
    tx = torch.rand(3, 4, 5, dtype=torch.float64) * 1.5 + 0.5
    if zeros != 'none':
        tx[1, 2, 3] = 0.0
        tx[2, 0, 0] = 0.0
    if zeros == 'double':
        tx[1, 2, 1] = 0.0
        tx[2, 3, 0] = 0.0
    tx = tx.to(torch.float32).requires_grad_(True)
    x = Tensor(tx.tolist(), device=device)
    x.requires_grad = True
    if dim is None:
        y, ty = x.prod(), tx.prod()
    elif isinstance(dim, int):
        y, ty = x.prod(dim, keepdim=keepdim), tx.prod(dim, keepdim=keepdim)
    else:
        y = x.prod(dim, keepdim=keepdim)
        ty = tx
        for d in sorted((d % 3 for d in dim), reverse=True):
            ty = ty.prod(d, keepdim=True)
        if not keepdim:
            ty = ty.squeeze(tuple(d % 3 for d in dim))
    _backward(y, ty, device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape, dim', [((7,), 0), ((3, 5), 0), ((3, 5), 1), ((2, 3, 4), -1), ((2, 3, 4), 1), ((2, 1, 5, 3), 2)])
def test_cumulative_backward(device: str, shape, dim: int) -> None:
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(x.cusum(dim), tx.cumsum(dim), device)
    _assert_grad(x, tx)
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(x.cumax(dim)[0], tx.cummax(dim).values, device)
    _assert_grad(x, tx)
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(x.cumin(dim)[0], tx.cummin(dim).values, device)
    _assert_grad(x, tx)


_PAD_BACKWARD_CASES = (
    ((3, 4), (1, 2), 'constant'),
    ((5,), (2, 3), 'constant'),
    ((2, 3, 4), (2, 1, 1, 1, 0, 2), 'constant'),
    ((2, 3, 4, 5), (1, 1, 2, 2), 'constant'),
    ((2, 3, 5), (2, 1), 'reflect'),
    ((2, 3, 4, 5), (1, 2, 2, 1), 'reflect'),
    ((1, 2, 3, 4, 5), (1, 1, 1, 1, 1, 1), 'reflect'),
    ((2, 3, 5), (2, 1), 'replicate'),
    ((2, 3, 4, 5), (1, 2, 2, 1), 'replicate'),
    ((1, 2, 3, 4, 5), (1, 1, 1, 1, 1, 1), 'replicate'),
    ((2, 3, 4, 5), (3, 3, 0, 0), 'replicate'),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape, pad, mode', _PAD_BACKWARD_CASES)
def test_pad_backward(device: str, shape, pad, mode: str) -> None:
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(x.pad(list(pad), mode=mode, value=1.5), F.pad(tx, list(pad), mode=mode, value=1.5 if mode == 'constant' else None), device)
    _assert_grad(x, tx)


_RI_BACKWARD_CASES = (((3,), 2, None), ((2, 3), 2, None), ((2, 3), 3, 1), ((2, 3), 2, 0), ((2, 3, 4), 3, -1), ((3,), [1, 2, 3], 0), ((2, 3), [2, 1], 0), ((2, 3), [1, 0, 2], 1), ((3,), [0, 2, 1], None), ((2, 3, 4), [2, 0, 1, 3], 2), ((4,), [0, 2, 0, 1], 0))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape, reps, dim', _RI_BACKWARD_CASES)
def test_repeat_interleave_backward(device: str, shape, reps, dim) -> None:
    x, tx = _leaf(shape, -2.0, 2.0, device)
    mreps = Tensor(reps, device=device) if isinstance(reps, list) else reps
    treps = torch.tensor(reps) if isinstance(reps, list) else reps
    if dim is None:
        y, ty = x.repeat_interleave(mreps), tx.repeat_interleave(treps)
    else:
        y, ty = x.repeat_interleave(mreps, dim=dim), tx.repeat_interleave(treps, dim=dim)
    _backward(y, ty, device)
    _assert_grad(x, tx)


_SCATTER_BACKWARD_CASES = (((3, 5), 0, (2, 5)), ((3, 5), 1, (3, 3)), ((2, 3, 4), 2, (2, 3, 2)), ((2, 3, 4), 0, (1, 3, 4)), ((4,), 0, (3,)))


def _unique_index(shape, dim: int, idx_shape) -> torch.Tensor:
    d = dim % len(shape)
    full = list(idx_shape)
    full[d] = shape[d]
    return torch.rand(full).argsort(dim=d).narrow(d, 0, idx_shape[d])


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape, dim, idx_shape', _SCATTER_BACKWARD_CASES)
@pytest.mark.parametrize('name', ['scatter', 'scatter_add'])
def test_scatter_backward(device: str, shape, dim: int, idx_shape, name: str) -> None:
    base, tbase = _leaf(shape, -2.0, 2.0, device)
    src, tsrc = _leaf(idx_shape, -2.0, 2.0, device)
    tidx = _unique_index(shape, dim, idx_shape) if name == 'scatter' else torch.randint(0, shape[dim], idx_shape)
    idx = Tensor(tidx.tolist(), device=device)
    _backward(getattr(base, name)(dim, idx, src), getattr(tbase, name)(dim, tidx, tsrc), device)
    _assert_grad(base, tbase)
    _assert_grad(src, tsrc)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape, k, dim', [((7,), 3, 0), ((3, 5), 2, 1), ((3, 5), 3, 0), ((2, 3, 4), 4, -1), ((2, 3, 4), 1, 1)])
@pytest.mark.parametrize('largest', [True, False])
def test_topk_backward(device: str, shape, k: int, dim: int, largest: bool) -> None:
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(x.topk(k, dim=dim, largest=largest)[0], tx.topk(k, dim=dim, largest=largest).values, device)
    _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape, dim', [((7,), 0), ((3, 5), 1), ((3, 5), 0), ((2, 3, 4), -1), ((2, 3, 4), 1)])
@pytest.mark.parametrize('descending', [False, True])
def test_sort_backward(device: str, shape, dim: int, descending: bool) -> None:
    x, tx = _leaf(shape, -2.0, 2.0, device)
    _backward(x.sort(dim=dim, descending=descending)[0], tx.sort(dim=dim, descending=descending).values, device)
    _assert_grad(x, tx)
