# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch.nn.functional

from ..common import *

_ALL_DTYPE_REDUCES = (
    'sum',
    'prod',
    'min',
    'max',
    'argmin',
    'argmax',
    'all',
    'any',
)


@pytest.mark.parametrize('dtype', FLOATING_NO_FLOAT8)
@pytest.mark.parametrize('op', _ALL_DTYPE_REDUCES)
@pytest.mark.parametrize('keepdim', [True, False])
def test_reduce_op(dtype: dtype.DType, op: str, keepdim: bool) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=dtype)
        dim = random_dim(shape)
        tx = totorch(x)
        if dim is None:
            r = call_reduction(x, op, None, keepdim)
            t = getattr(tx, op)()
        else:
            r = call_reduction(x, op, dim, keepdim)
            t = getattr(tx, op)(dim=dim, keepdim=keepdim)

        if not isinstance(t, torch.Tensor):
            t = t[0]  # min, max, argmin, argmax return (values, indices)

        assert_close_mag_torch(r, t, dtype, equal_nan=True)

    for_all_shapes(test)


@pytest.mark.parametrize('dtype', FLOATING_NO_FLOAT8)
@pytest.mark.parametrize('keepdim', [True, False])
def test_reduce_op_mean(dtype: dtype.DType, keepdim: bool) -> None:  # Mean is only for floating point
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=dtype)
        dim = random_dim(shape)
        if dim is None:
            r = x.mean()
            t = totorch(x).mean()
        else:
            r = x.mean(dim=dim, keepdim=keepdim)
            t = totorch(x).mean(dim=dim, keepdim=keepdim)

        assert_close_mag_torch(r, t, dtype, equal_nan=True)

    for_all_shapes(test)


@pytest.mark.parametrize('dtype', FLOATING_NO_FLOAT8)
@pytest.mark.parametrize('largest', [True, False])
def test_reduce_op_topk(dtype: dtype.DType, largest: bool) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0:  # topk not defined for 0-dim tensors
            return
        x = random_tensor(shape, dt=dtype)
        k = random.randint(1, max(1, min(shape)))
        dim = random_dim(shape)
        tx = totorch(x)
        if dim is None:
            rv, ri = x.topk(k, largest=largest)
            tv, ti = tx.topk(k, largest=largest)
        else:
            rv, ri = x.topk(k, dim=dim, largest=largest)
            tv, ti = tx.topk(k, dim=dim, largest=largest)

        assert_close_mag_torch(rv, tv, dtype, equal_nan=True)
        np.testing.assert_array_equal(tonumpy(ri), tonumpy(ti))

    for_all_shapes(test)
