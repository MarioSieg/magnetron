# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from ..common import *

def test_binary_op_broadcast_shape() -> None:
    cm = Tensor.uniform(4) * Tensor.uniform(2, 3, 4)
    ct = torch.rand(4) * torch.rand(2, 3, 4)
    assert cm.shape == ct.shape, f"Expected shape {ct.shape}, got {cm.shape}"

def test_binary_op_broadcast_shape_cmp() -> None:
    cm = Tensor.uniform(4) == Tensor.uniform(2, 3, 4)
    ct = torch.rand(4) == torch.rand(2, 3, 4)
    assert cm.shape == ct.shape, f"Expected shape {ct.shape}, got {cm.shape}"

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_add(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_op_square(dtype, lambda x, y: x + y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_sub(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_op_square(dtype, lambda x, y: x + y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_mul(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_op_square(dtype, lambda x, y: x * y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [float16, float32])
def test_binary_op_div_fp(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_op_square(dtype, lambda x, y: x / y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
def test_binary_op_div_int32(kind: BinaryOpParamKind) -> None:
    binary_op_square(int32, lambda x, y: x // y, kind=kind, low=1, high=10000)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [boolean, int32])
def test_binary_op_and(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_op_square(dtype, lambda x, y: x & y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [boolean, int32])
def test_binary_op_or(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_op_square(dtype, lambda x, y: x | y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [boolean, int32])
def test_binary_op_xor(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_op_square(dtype, lambda x, y: x ^ y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
def test_binary_op_shl(kind: BinaryOpParamKind) -> None:
    binary_op_square(int32, lambda x, y: x << y, kind=kind, low=0, high=31)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
def test_binary_op_shr(kind: BinaryOpParamKind) -> None:
    binary_op_square(int32, lambda x, y: x >> y, kind=kind, low=0, high=31)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR, BinaryOpParamKind.LIST])
@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_binary_op_eq(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_cmp_op(dtype, lambda x, y: x == y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR, BinaryOpParamKind.LIST])
@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_binary_op_ne(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_cmp_op(dtype, lambda x, y: x != y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_le(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_cmp_op(dtype, lambda x, y: x <= y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_ge(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_cmp_op(dtype, lambda x, y: x >= y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_lt(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_cmp_op(dtype, lambda x, y: x < y, kind=kind)

@pytest.mark.parametrize('kind', [BinaryOpParamKind.TENSOR, BinaryOpParamKind.SCALAR])
@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_binary_op_gt(dtype: DataType, kind: BinaryOpParamKind) -> None:
    binary_cmp_op(dtype, lambda x, y: x > y, kind=kind)
