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

@pytest.mark.parametrize('dtype', FLOATING_POINT_DTYPES)
@pytest.mark.parametrize('op', _ALL_DTYPE_REDUCES)
@pytest.mark.parametrize('keepdim', [True, False])
def test_reduce_op(dtype: DataType, op: str, keepdim: bool) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dtype=dtype)
        dim = random_dim(shape)
        tx = totorch(x)
        op_mag = getattr(x, op)
        op_torch = getattr(tx, op)
        if dim is None:
            r = op_mag()
            t = op_torch()
        else:
            r = op_mag(dim=dim, keepdim=keepdim)
            t = op_torch(dim=dim, keepdim=keepdim)

        if not isinstance(t, torch.Tensor):
            t = t[0]  # min, max, argmin, argmax return (values, indices)

        torch.testing.assert_close(totorch(r), t, equal_nan=True)

    for_all_shapes(test)


@pytest.mark.parametrize('dtype', FLOATING_POINT_DTYPES)
@pytest.mark.parametrize('keepdim', [True, False])
def test_reduce_op_mean(dtype: DataType, keepdim: bool) -> None: # Mean is only for floating point
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dtype=dtype)
        dim = random_dim(shape)
        if dim is None:
            r = x.mean()
            t = totorch(x).mean()
        else:
            r = x.mean(dim=dim, keepdim=keepdim)
            t = totorch(x).mean(dim=dim, keepdim=keepdim)

        torch.testing.assert_close(totorch(r), t, equal_nan=True)

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', FLOATING_POINT_DTYPES)
@pytest.mark.parametrize('largest', [True, False])
def test_reduce_op_topk(dtype: DataType, largest: bool) -> None: # Mean is only for floating point
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0: # topk not defined for 0-dim tensors
            return
        x = random_tensor(shape, dtype=dtype)
        k = random.randint(1, max(1, min(shape )))
        dim = random_dim(shape)
        if dim is None:
            rv, ri = x.topk(k, largest=largest)
            tv, ti = totorch(x).topk(k, largest=largest)
        else:
            rv, ri = x.topk(k, dim=dim, largest=largest)
            tv, ti = totorch(x).topk(k, dim=dim, largest=largest)

        torch.testing.assert_close(totorch(rv), tv, equal_nan=True)
        torch.testing.assert_close(totorch(ri), ti)

    for_all_shapes(test)
