# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import *

_STRICT = dict(strict=True)


@pytest.mark.xfail(reason='cumax/cumin return the first index of a run of equal values; torch returns the last', **_STRICT)
@pytest.mark.parametrize('op', ['cumax', 'cumin'])
def test_cummax_cummin_tie_index_like_torch(op: str) -> None:
    values = [1, 3, 3, 2, 3] if op == 'cumax' else [3, 1, 1, 2, 1]
    x = Tensor(values, dtype=dtype.int32)
    tname = 'cummax' if op == 'cumax' else 'cummin'
    _, indices = getattr(x, op)(0)
    assert indices.tolist() == getattr(torch.tensor(values), tname)(0).indices.tolist()


@pytest.mark.xfail(reason='elementwise min/max drop NaN operands; torch.minimum/maximum propagate NaN', **_STRICT)
@pytest.mark.parametrize('name', ['min', 'max'])
def test_binary_minmax_propagates_nan_like_torch(name: str) -> None:
    a = Tensor([float('nan'), 1.0, 2.0])
    b = Tensor([0.0, float('nan'), 3.0])
    ref = torch.minimum if name == 'min' else torch.maximum
    got = getattr(a, name)(b)
    expected = ref(totorch(a), totorch(b))
    torch.testing.assert_close(totorch(got), expected, equal_nan=True)


@pytest.mark.xfail(reason='pad rejects negative padding; torch crops with negative constant padding', **_STRICT)
def test_negative_constant_pad_like_torch() -> None:
    x = uniform_tensor((2, 3, 6))
    assert_close_mag_torch(x.pad([-1, 2]), F.pad(totorch(x), [-1, 2]), dtype.float32)


@pytest.mark.xfail(reason='stack rejects negative dims; torch accepts them', **_STRICT)
def test_stack_negative_dim_like_torch() -> None:
    xs = [uniform_tensor((2, 3)) for _ in range(3)]
    assert_close_mag_torch(Tensor.stack(xs, -1), torch.stack([totorch(x) for x in xs], -1), dtype.float32)


@pytest.mark.xfail(reason="meshgrid only supports indexing='ij'; torch also supports 'xy'", **_STRICT)
def test_meshgrid_xy_like_torch() -> None:
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0])
    got = Tensor.meshgrid(a, b, indexing='xy')
    expected = torch.meshgrid(totorch(a), totorch(b), indexing='xy')
    for g, e in zip(got, expected):
        assert g.tolist() == e.tolist()


@pytest.mark.xfail(reason='softmax(dim) ignores dim and always normalizes over the last axis', **_STRICT)
@pytest.mark.parametrize('dim', [0, 1, -3])
def test_softmax_over_non_last_dim_like_torch(dim: int) -> None:
    x = uniform_tensor((3, 4, 5), low=-3.0, high=3.0)
    assert_close_mag_torch(x.softmax(dim), torch.softmax(totorch(x), dim), dtype.float32)


@pytest.mark.xfail(reason='sum/prod reject boolean tensors; torch sums booleans as int64', **_STRICT)
def test_bool_sum_like_torch() -> None:
    x = Tensor.bernoulli((4, 5), p=0.5)
    assert x.sum().tolist() == totorch(x).sum().tolist()


_MISSING_BACKWARD = (
    ('max_dim', lambda x: x.max(1), lambda t: t.max(1).values),
    ('min_dim', lambda x: x.min(0, keepdim=True), lambda t: t.min(0, keepdim=True).values),
    ('max_full', lambda x: x.max(), lambda t: t.max()),
    ('prod', lambda x: x.prod(), lambda t: t.prod()),
    ('cusum', lambda x: x.cusum(1), lambda t: t.cumsum(1)),
    ('pad', lambda x: x.pad([1, 1]), lambda t: F.pad(t, [1, 1])),
    ('repeat_interleave', lambda x: x.repeat_interleave(2, dim=0), lambda t: t.repeat_interleave(2, dim=0)),
    ('scatter_add', lambda x: Tensor.zeros(3, 4).scatter_add(0, Tensor([[0, 1, 2, 0]]), x[:1]), lambda t: torch.zeros(3, 4).scatter_add(0, torch.tensor([[0, 1, 2, 0]]), t[:1])),
    ('topk', lambda x: x.topk(2)[0], lambda t: t.topk(2).values),
    ('sort', lambda x: x.sort()[0], lambda t: t.sort().values),
)


@pytest.mark.xfail(reason='no backward implemented for this operator; torch differentiates it', **_STRICT)
@pytest.mark.parametrize('name, fn, ref', _MISSING_BACKWARD, ids=[c[0] for c in _MISSING_BACKWARD])
def test_backward_exists_like_torch(name: str, fn: Callable, ref: Callable) -> None:
    x = uniform_tensor((3, 4), low=0.5, high=1.5)
    tx = totorch(x).clone().requires_grad_(True)
    x.requires_grad = True
    y = fn(x)
    ty = ref(tx)
    w = random_tensor(y.shape, dtype.float32)
    (y * w).sum().backward()
    (ty * totorch(w)).sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)


@pytest.mark.xfail(reason='elementwise min/max backward gives the whole gradient to the left operand on ties; torch splits it in half', **_STRICT)
@pytest.mark.parametrize('name', ['min', 'max'])
def test_binary_minmax_backward_on_ties_like_torch(name: str) -> None:
    x = uniform_tensor((3, 4), low=-2.0, high=2.0)
    y = x.clone()
    tx, ty = totorch(x).clone().requires_grad_(True), totorch(y).clone().requires_grad_(True)
    x.requires_grad = True
    y.requires_grad = True
    r = getattr(x, name)(y)
    tr = (torch.minimum if name == 'min' else torch.maximum)(tx, ty)
    w = random_tensor(r.shape, dtype.float32)
    (r * w).sum().backward()
    (tr * totorch(w)).sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)
    assert_close_mag_torch(y.grad, ty.grad, dtype.float32)


@pytest.mark.xfail(reason='repeat backward does not accumulate the tiled gradients like torch', **_STRICT)
def test_repeat_backward_like_torch() -> None:
    x = uniform_tensor((3, 4, 5), low=-2.0, high=2.0)
    tx = totorch(x).clone().requires_grad_(True)
    x.requires_grad = True
    r = x.repeat(2, 1, 3)
    tr = tx.repeat(2, 1, 3)
    w = random_tensor(r.shape, dtype.float32)
    (r * w).sum().backward()
    (tr * totorch(w)).sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)


@pytest.mark.xfail(reason='gelu_approx backward differs from the derivative of the tanh approximation by up to ~5e-4', **_STRICT)
def test_gelu_approx_backward_like_torch() -> None:
    x = uniform_tensor((64, 65), low=-3.0, high=3.0)
    tx = totorch(x).clone().requires_grad_(True)
    x.requires_grad = True
    r = x.gelu_approx()
    tr = F.gelu(tx, approximate='tanh')
    assert_close_mag_torch(r, tr.detach(), dtype.float32)
    w = random_tensor(r.shape, dtype.float32)
    (r * w).sum().backward()
    (tr * totorch(w)).sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)


@pytest.mark.xfail(reason='matmul backward fails for a rank-1 left operand; torch differentiates vector @ matrix', **_STRICT)
def test_matmul_vector_lhs_backward_like_torch() -> None:
    x = uniform_tensor((4,), low=-1.0, high=1.0)
    w = uniform_tensor((6, 4), low=-1.0, high=1.0)
    tx, tw = totorch(x).clone().requires_grad_(True), totorch(w).clone().requires_grad_(True)
    x.requires_grad = True
    w.requires_grad = True
    r = x @ w.T
    tr = tx @ tw.T
    assert_close_mag_torch(r, tr.detach(), dtype.float32)
    s = random_tensor(r.shape, dtype.float32)
    (r * s).sum().backward()
    (tr * totorch(s)).sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)
    assert_close_mag_torch(w.grad, tw.grad, dtype.float32)


@pytest.mark.xfail(reason='gather rejects negative dims; torch accepts them', **_STRICT)
def test_gather_negative_dim_like_torch() -> None:
    x = uniform_tensor((2, 3, 4))
    tidx = torch.randint(0, 4, (2, 3, 6))
    assert_close_mag_torch(x.gather(-1, Tensor(tidx.tolist())), totorch(x).gather(-1, tidx), dtype.float32)


@pytest.mark.xfail(reason='clamp with min > max keeps min; torch sets every element to max', **_STRICT)
@pytest.mark.parametrize('dt, lo, hi', [(dtype.float32, 0.5, -0.5), (dtype.int32, 10, -10)], ids=['float32', 'int32'])
def test_clamp_inverted_bounds_like_torch(dt: dtype.DType, lo, hi) -> None:
    x = random_tensor((3, 4), dt)
    assert x.clamp(lo, hi).tolist() == torch.clamp(totorch(x), lo, hi).tolist()


@pytest.mark.xfail(reason='embedding requires int64 indices; torch also accepts int32', **_STRICT)
def test_embedding_int32_indices_like_torch() -> None:
    w = uniform_tensor((6, 3))
    tidx = torch.randint(0, 6, (2, 3), dtype=torch.int32)
    idx = Tensor(tidx.tolist(), dtype=dtype.int32)
    assert_close_mag_torch(w.embedding(idx), F.embedding(tidx, totorch(w)), dtype.float32)


@pytest.mark.xfail(reason='Tensor() rejects numpy scalar objects such as np.float32(1.5); torch.tensor accepts them', **_STRICT)
@pytest.mark.parametrize('value', [np.float32(1.5), np.bool_(True), np.int64(7)], ids=['float32', 'bool', 'int64'])
def test_numpy_scalar_constructor_like_torch(value) -> None:
    t = Tensor(value)
    assert t.shape == ()
    assert t.item() == torch.tensor(value).item()


@pytest.mark.xfail(reason='gather requires the index to match the input in every non-gather dim; torch allows a smaller index', **_STRICT)
@pytest.mark.parametrize('dim, idx_shape', [(1, (2, 5, 3)), (2, (2, 2, 2)), (0, (3, 1, 1)), (1, (2, 1, 1))])
def test_gather_with_smaller_index_like_torch(dim: int, idx_shape) -> None:
    x = uniform_tensor((2, 3, 4))
    tidx = torch.randint(0, x.shape[dim], idx_shape)
    assert x.gather(dim, Tensor(tidx.tolist())).tolist() == totorch(x).gather(dim, tidx).tolist()


@pytest.mark.xfail(reason='in-place floor division is rejected for float tensors; torch supports //= on floats', **_STRICT)
@pytest.mark.parametrize('dt', [dtype.float32, dtype.float16], ids=['float32', 'float16'])
def test_inplace_float_floordiv_like_torch(dt: dtype.DType) -> None:
    x = uniform_tensor((3, 4), -5.0, 5.0, dt)
    y = uniform_tensor((3, 4), 0.5, 2.0, dt)
    tx, ty = totorch(x), totorch(y)
    x //= y
    tx //= ty
    assert_close_mag_torch(x, tx, dt)


@pytest.mark.xfail(reason='in-place binary ops accept a left operand smaller than the broadcast shape and write into it; torch raises', **_STRICT)
@pytest.mark.parametrize('name', ['__iadd__', '__isub__', '__imul__', '__itruediv__'])
def test_inplace_with_smaller_lhs_raises_like_torch(name: str) -> None:
    small = uniform_tensor((3, 1))
    big = uniform_tensor((3, 4), low=0.5, high=1.5)
    with pytest.raises(RuntimeError):
        getattr(totorch(small), name)(totorch(big))
    with pytest.raises(Exception):
        getattr(small, name)(big)


@pytest.mark.xfail(reason='uniform/normal return the same values on every call; torch advances the generator', **_STRICT)
@pytest.mark.parametrize('factory', ['uniform', 'normal'])
def test_random_factories_advance_generator_like_torch(factory: str) -> None:
    a = getattr(Tensor, factory)((64,))
    b = getattr(Tensor, factory)((64,))
    assert a.tolist() != b.tolist()
    c = Tensor.zeros(64)
    getattr(c, f'{factory}_')()
    d = Tensor.zeros(64)
    getattr(d, f'{factory}_')()
    assert c.tolist() != d.tolist()
    ta = getattr(torch, 'rand' if factory == 'uniform' else 'randn')(64)
    tb = getattr(torch, 'rand' if factory == 'uniform' else 'randn')(64)
    assert not torch.equal(ta, tb)
