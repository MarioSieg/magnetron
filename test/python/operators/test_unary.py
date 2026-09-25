# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations
from dataclasses import dataclass

import pytest
import torch.nn.functional

from ..common import *


@dataclass
class UnaryOpTestCase:
    name: str
    torch_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor]
    rank_min: int = 0
    inplace: bool = True


_UNARY_OPS: tuple[UnaryOpTestCase, ...] = (
    UnaryOpTestCase('clone', None, 0, False),
    UnaryOpTestCase('abs', None),
    UnaryOpTestCase('neg', None),
    UnaryOpTestCase('log', None),
    UnaryOpTestCase('log10', None),
    UnaryOpTestCase('log1p', None),
    UnaryOpTestCase('log2', None),
    UnaryOpTestCase('sqr', lambda x: x * x),
    UnaryOpTestCase('rcp', lambda x: torch.reciprocal(x)),
    UnaryOpTestCase('sqrt', None),
    UnaryOpTestCase('rsqrt', None),
    UnaryOpTestCase('sin', None),
    UnaryOpTestCase('asin', None),
    UnaryOpTestCase('sinh', None),
    UnaryOpTestCase('asinh', None),
    UnaryOpTestCase('cos', None),
    UnaryOpTestCase('acos', None),
    UnaryOpTestCase('cosh', None),
    UnaryOpTestCase('acosh', None),
    UnaryOpTestCase('tan', None),
    UnaryOpTestCase('atan', None),
    UnaryOpTestCase('tanh', None),
    UnaryOpTestCase('atanh', None),
    UnaryOpTestCase('step', lambda x: torch.where(x >= 0, torch.tensor(1, dtype=x.dtype), torch.tensor(0, dtype=x.dtype))),
    UnaryOpTestCase('erf', None),
    UnaryOpTestCase('erfc', None),
    UnaryOpTestCase('exp', None),
    UnaryOpTestCase('expm1', None),
    UnaryOpTestCase('exp2', None),
    UnaryOpTestCase('floor', None),
    UnaryOpTestCase('ceil', None),
    UnaryOpTestCase('round', None),
    UnaryOpTestCase('trunc', None),
    UnaryOpTestCase('softmax', lambda x: torch.nn.functional.softmax(x, dim=-1)),
    UnaryOpTestCase('sigmoid', None),
    UnaryOpTestCase('hard_sigmoid', lambda x: torch.nn.functional.hardsigmoid(x)),
    UnaryOpTestCase('silu', None),
    UnaryOpTestCase('gelu', None),
    UnaryOpTestCase('tril', None, 2),
    UnaryOpTestCase('triu', None, 2),
)


_UNARY_TOLS: dict[dtype.DType, tuple[float, float]] = {
    dtype.float32: (1e-5, 1e-5),
    dtype.float16: (1e-3, 1e-5),
    dtype.bfloat16: (1.6e-2, 1e-5),
}


def _unary_tol(device: str, dt: dtype.DType, op_name: str) -> tuple[float, float]:
    return _UNARY_TOLS[dt]


def unary_op(
    device: str,
    dtype: dtype.DType,
    rank_min: int,
    op_name: str,
    mag_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor],
    torch_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor],
) -> None:
    rtol, atol = _unary_tol(device, dtype, op_name)

    def test(shape: tuple[int, ...]) -> None:
        if len(shape) < rank_min:
            return
        x = random_tensor(shape, dtype, device=device)
        r = mag_callback(x.clone())
        torch.testing.assert_close(totorch(r), torch_callback(totorch(x)), equal_nan=True, rtol=rtol, atol=atol)

    for_all_shapes(test)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dtype', dtype.floating - {dtype.float8_e4m3fn})
@pytest.mark.parametrize('op', _UNARY_OPS)
def test_unary_op(device: str, dtype: dtype.DType, op: UnaryOpTestCase) -> None:
    name = op.name
    if op.torch_callback is not None:
        torch_op = op.torch_callback
    elif hasattr(torch, name):
        torch_op = getattr(torch, name)
    elif hasattr(torch.nn.functional, name):
        torch_op = getattr(torch.nn.functional, name)
    else:
        raise RuntimeError(f'No reference torch op found for unary op {name!r}')
    unary_op(device, dtype, op.rank_min, name, lambda x: getattr(x, name)(), lambda x: torch_op(x))
    if op.inplace:
        unary_op(device, dtype, op.rank_min, name, lambda x: getattr(x, name + '_')(), lambda x: torch_op(x))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', dtype.integer)
def test_unary_abs_integer(device: str, dt: dtype.DType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=dt, device=device)
        r = x.clone().abs()
        np.testing.assert_array_equal(tonumpy(r), np.abs(tonumpy(x)))

    for_all_shapes(test)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', dtype.integral)
def test_unary_logical_not_integral(device: str, dt: dtype.DType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=dt, device=device)
        expected = np.logical_not(tonumpy(x)) if dt == dtype.boolean else np.bitwise_not(tonumpy(x))
        np.testing.assert_array_equal(tonumpy(~x), expected)
        np.testing.assert_array_equal(tonumpy(x.logical_not()), expected)
        y = x.clone()
        y.logical_not_()
        np.testing.assert_array_equal(tonumpy(y), expected)

    for_all_shapes(test)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', [dtype.float32, dtype.float16, dtype.bfloat16], ids=['float32', 'float16', 'bfloat16'])
def test_unary_round_ties_to_even(device: str, dt: dtype.DType) -> None:
    ties = [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 0.0, -0.0, 4.5, 1024.5, -1024.5]
    x = Tensor(ties, dtype=dt, device=device)
    torch.testing.assert_close(totorch(x.round()), totorch(x).round(), rtol=0, atol=0)
    y = x.clone()
    y.round_()
    torch.testing.assert_close(totorch(y), totorch(x).round(), rtol=0, atol=0)

    def test(shape: tuple[int, ...]) -> None:
        x = uniform_tensor(shape, -8.0, 8.0, dt, device)
        torch.testing.assert_close(totorch(x.round()), totorch(x).round(), rtol=0, atol=0)

    for_all_shapes(test)
