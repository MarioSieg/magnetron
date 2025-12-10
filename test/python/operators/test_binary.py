# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from ..common import *

# We test against numpy here as torch has some issues with certain dtypes
# Torch's unsigned types are shell types and do not support key operations properly

_BINARY_OPS_NUMERIC: tuple[tuple[str, Callable], ...] = (
    ('add', lambda x, y: x + y),
    ('sub', lambda x, y: x - y),
    ('mul', lambda x, y: x * y),
    ('truediv', lambda x, y: x / y),
    ('floordiv', lambda x, y: x // y),
    ('mod', lambda x, y: x % y),
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

def binary_unary_op(
    dtype: DataType,
    avoid_zero_in_y: bool,
    mag_callback: Callable[[Tensor | np.ndarray, Tensor | np.ndarray], Tensor | np.ndarray],
    np_callback: Callable[[Tensor | np.ndarray, Tensor | np.ndarray], Tensor | np.ndarray]
) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dtype)
        y = random_tensor(shape, dtype)
        if avoid_zero_in_y: # For division and similar operations
            y = y + (y == 0).cast(dtype) # Removes zeros from y to avoid division by zero
        r = mag_callback(x.clone(), y.clone())
        np.testing.assert_allclose(tonumpy(r), np_callback(tonumpy(x), tonumpy(y)), equal_nan=True)

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', NUMERIC_DTYPES)
@pytest.mark.parametrize('op', _BINARY_OPS_NUMERIC)
def test_binary_op_numeric(dtype: DataType, op: tuple[str, Callable]) -> None:
    callback = op[1]
    binary_unary_op(dtype, True, callback, callback)


@pytest.mark.parametrize('dtype', INTEGRAL_DTYPES)
@pytest.mark.parametrize('op', _BINARY_OPS_BITWISE_INTEGRAL)
def test_binary_op_integral(dtype: DataType, op: tuple[str, Callable]) -> None:
    callback = op[1]
    binary_unary_op(dtype, False, callback, callback)

@pytest.mark.parametrize('dtype', INTEGER_DTYPES)
@pytest.mark.parametrize('op', _BINARY_OPS_INTEGER)
def test_binary_op_integer(dtype: DataType, op: tuple[str, Callable]) -> None:
    callback = op[1]
    binary_unary_op(dtype, True, callback, callback)
