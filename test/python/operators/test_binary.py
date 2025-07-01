# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from ..common import *

@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_add(dtype: DataType) -> None:
    binary_op_square(dtype, lambda x, y: x + y)

@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_sub(dtype: DataType) -> None:
    binary_op_square(dtype, lambda x, y: x + y)

@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_mul(dtype: DataType) -> None:
    binary_op_square(dtype, lambda x, y: x * y)

@pytest.mark.parametrize('dtype', [float16, float32])
def test_binary_op_div_fp(dtype: DataType) -> None:
    binary_op_square(dtype, lambda x, y: x / y)

def test_binary_op_div_int32() -> None:
    binary_op_square(int32, lambda x, y: x // y, from_=1, to=10000)

@pytest.mark.parametrize('dtype', [boolean, int32])
def test_binary_op_and(dtype: DataType) -> None:
    binary_op_square(dtype, lambda x, y: x & y)

@pytest.mark.parametrize('dtype', [boolean, int32])
def test_binary_op_or(dtype: DataType) -> None:
    binary_op_square(dtype, lambda x, y: x | y)

@pytest.mark.parametrize('dtype', [boolean, int32])
def test_binary_op_xor(dtype: DataType) -> None:
    binary_op_square(dtype, lambda x, y: x ^ y)

def test_binary_op_shl() -> None:
    binary_op_square(int32, lambda x, y: x << y, from_=0, to=31)

def test_binary_op_shr() -> None:
    binary_op_square(int32, lambda x, y: x >> y, from_=0, to=31)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_binary_op_eq(dtype: DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x == y)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_binary_op_ne(dtype: DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x != y)

@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_le(dtype: DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x <= y)

@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_ge(dtype: DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x >= y)

@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_lt(dtype: DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x < y)

@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_gt(dtype: DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x > y)
