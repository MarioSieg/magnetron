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
