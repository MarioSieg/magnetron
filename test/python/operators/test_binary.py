# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>
from typing import Callable

import torch.testing

from ..common import *

# We test integers against numpy here as torch has some issues with certain dtypes.
# Float binary ops required touch tough as numpy does not support bfloat16 😾
# Torch's unsigned types are shell types and do not support key operations properly

_BINARY_OPS_NUMERIC: tuple[tuple[str, Callable], ...] = (
    ('add', lambda x, y: x + y),
    ('sub', lambda x, y: x - y),
    ('mul', lambda x, y: x * y),
    ('truediv', lambda x, y: x / y),
    ('floordiv', lambda x, y: x // y),
    ('mod', lambda x, y: x % y),
    ('pow', lambda x, y: x**y),
    ('eq', lambda x, y: x == y),
    ('ne', lambda x, y: x != y),
    ('lt', lambda x, y: x < y),
    ('le', lambda x, y: x <= y),
    ('gt', lambda x, y: x > y),
    ('ge', lambda x, y: x >= y),
)

_BINARY_OPS_BITWISE_INTEGRAL: tuple[tuple[str, Callable], ...] = (
    ('bitwise_and', lambda x, y: x & y),
    ('bitwise_or', lambda x, y: x | y),
    ('bitwise_xor', lambda x, y: x ^ y),
)

_BINARY_OPS_INTEGER: tuple[tuple[str, Callable], ...] = (
    ('lshift', lambda x, y: x << y),
    ('rshift', lambda x, y: x >> y),
)


def binary_unary_op_np(
    device: str,
    dtype: dtype.DType,
    op_name: str,
    avoid_zero_in_y: bool,
    mag_callback: Callable[[Tensor | np.ndarray, Tensor | np.ndarray], Tensor | np.ndarray],
    np_callback: Callable[[Tensor | np.ndarray, Tensor | np.ndarray], Tensor | np.ndarray],
) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if op_name in ('lshift', 'rshift'):
            # Keep values small so shifts match NumPy overflow rules.
            x = Tensor.uniform(shape, low=0, high=8, dtype=dtype, device=device)
            y = clamp_shift_amount(random_tensor(shape, dt=dtype, device=device), dtype)
        else:
            x = random_tensor(shape, dt=dtype, device=device)
            y = random_tensor(shape, dt=dtype, device=device)
        if avoid_zero_in_y:  # For division and similar operations
            y = y + (y == 0).cast(dtype)  # Removes zeros from y to avoid division by zero
        if op_name == 'pow':
            y = y.abs()  # Avoid negative powers (numpy rejects those for integers)
        r = mag_callback(x.clone(), y.clone())
        expected = np_callback(tonumpy(x), tonumpy(y))
        if op_name in {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}:
            np.testing.assert_array_equal(tonumpy(r), expected)
        else:
            np.testing.assert_allclose(tonumpy(r), expected, equal_nan=True)

    for_all_shapes(test)


def binary_unary_op_torch(
    device: str,
    dtype: dtype.DType,
    op: str,
    mag_callback: Callable[[Tensor | np.ndarray, Tensor | np.ndarray], Tensor | np.ndarray],
    np_callback: Callable[[Tensor | np.ndarray, Tensor | np.ndarray], Tensor | np.ndarray],
) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=dtype, device=device)
        y = random_tensor(shape, dt=dtype, device=device)
        if op in ('div', 'mod'):  # For division and similar operations
            y = y + (y == 0).cast(dtype)  # Removes zeros from y to avoid division by zero
        elif op == 'pow':
            y = y.abs()  # Avoid negative powers
        r = mag_callback(x.clone(), y.clone())
        kwargs: dict[str, Any] = {'equal_nan': True}
        torch.testing.assert_close(totorch(r), np_callback(totorch(x), totorch(y)), **kwargs)

    for_all_shapes(test)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', FLOATING_NO_FLOAT8)
@pytest.mark.parametrize('op', _BINARY_OPS_NUMERIC)
def test_binary_op_numeric_fp(device: str, dt: dtype.DType, op: tuple[str, Callable]) -> None:
    callback = op[1]
    binary_unary_op_torch(device, dt, op[0], callback, callback)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', dtype.integer)
@pytest.mark.parametrize('op', _BINARY_OPS_NUMERIC)
def test_binary_op_numeric_integers(device: str, dt: dtype.DType, op: tuple[str, Callable]) -> None:
    callback = op[1]
    binary_unary_op_np(device, dt, op[0], True, callback, callback)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', dtype.integral)
@pytest.mark.parametrize('op', _BINARY_OPS_BITWISE_INTEGRAL)
def test_binary_op_integral(device: str, dt: dtype.DType, op: tuple[str, Callable]) -> None:
    callback = op[1]
    binary_unary_op_np(device, dt, op[0], False, callback, callback)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', dtype.integer)
@pytest.mark.parametrize('op', _BINARY_OPS_INTEGER)
def test_binary_op_integer(device: str, dt: dtype.DType, op: tuple[str, Callable]) -> None:
    callback = op[1]
    binary_unary_op_np(device, dt, op[0], False, callback, callback)
