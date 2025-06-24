# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import random
import pytest
import torch

import magnetron as mag

DTYPE_TORCH_MAP: dict[mag.DataType, torch.dtype] = {
    mag.float16: torch.float16,
    mag.float32: torch.float32,
    mag.int32: torch.int32,
    mag.boolean: torch.bool
}


def totorch(t: mag.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(t.tolist(), dtype=dtype).reshape(t.shape)


def square_shape_permutations(f: callable, lim: int) -> None:
    lim += 1
    for i0 in range(1, lim):
        for i1 in range(1, lim):
            for i2 in range(1, lim):
                for i3 in range(1, lim):
                    for i4 in range(1, lim):
                        for i5 in range(1, lim):
                            f((i0, i1, i2, i3, i4, i5))


def binary_op_square(dtype: mag.DataType, f: callable, lim: int = 4, from_: float | int | None = None, to: float | int | None = None) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def compute(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
            y = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
            y = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
        r = f(x, y)
        torch.testing.assert_close(
            totorch(r, torch_dt),
            f(totorch(x, torch_dt), totorch(y, torch_dt))
        )

    square_shape_permutations(compute, lim)

def binary_cmp_op(dtype: mag.DataType, f: callable, lim: int = 4, from_: float | int | None = None, to: float | int | None = None) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def compute(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
            y = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
            y = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
        r = f(x, y)
        assert r.dtype == mag.boolean
        torch.testing.assert_close(
            totorch(r, torch.bool),
            f(totorch(x, torch_dt), totorch(y, torch_dt))
        )

    square_shape_permutations(compute, lim)

def unary_op(
    dtype: mag.DataType,
    magf: callable,
    torchf: callable,
    lim: int = 4,
    from_: float | int | None = None,
    to: float | int | None = None
) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def compute(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
        r = magf(x.clone())
        torch.testing.assert_close(totorch(r, torch_dt), torchf(totorch(x, torch_dt)))

    square_shape_permutations(compute, lim)


def scalar_op(dtype: mag.DataType, f: callable, rhs: bool = True, lim: int = 4) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def compute(shape: tuple[int, ...]) -> None:  # x op scalar
        xi: float = random.uniform(-1.0, 1.0)
        x = mag.Tensor.uniform(shape, dtype=dtype)
        r = f(x, xi)
        torch.testing.assert_close(totorch(r, torch_dt), f(totorch(x, torch_dt), xi))

    square_shape_permutations(compute, lim)

    if not rhs:
        return

    def compute(shape: tuple[int, ...]) -> None:  # scalar op x
        xi: float = random.uniform(-1.0, 1.0)
        x = mag.Tensor.uniform(shape)
        r = f(xi, x)
        torch.testing.assert_close(totorch(r, torch_dt), f(xi, totorch(x, torch_dt)))

    square_shape_permutations(compute, lim)


@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_abs(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.abs(), lambda x: torch.abs(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.abs_(), lambda x: torch.abs(x))


@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_neg(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: -x, lambda x: -x)


@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_log(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.log(), lambda x: torch.log(x), from_=0, to=1000)
    with mag.no_grad():
        unary_op(dtype, lambda x: x.log_(), lambda x: torch.log(x), from_=0, to=1000)


@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_sqr(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.sqr(), lambda x: x * x)
    with mag.no_grad():
        unary_op(dtype, lambda x: x.sqr_(), lambda x: x * x)


@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_sqrt(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.sqrt(), lambda x: torch.sqrt(x), from_=0, to=1000)
    with mag.no_grad():
        unary_op(dtype, lambda x: x.sqrt_(), lambda x: torch.sqrt(x), from_=0, to=1000)


@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_sin(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.sin(), lambda x: torch.sin(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.sin_(), lambda x: torch.sin(x))


@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_cos(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.cos(), lambda x: torch.cos(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.cos_(), lambda x: torch.cos(x))


@pytest.mark.parametrize('dtype', [mag.float32]) # Heaviside is not supported for fp16 in Torch, magnetron supports it tough
def test_unary_op_step(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.step(), lambda x: torch.heaviside(x, torch.tensor([0.0])))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.step_(), lambda x: torch.heaviside(x, torch.tensor([0.0])))


@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_exp(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.exp(), lambda x: torch.exp(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.exp_(), lambda x: torch.exp(x))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_floor(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.floor(), lambda x: torch.floor(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.floor_(), lambda x: torch.floor(x))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_ceil(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.ceil(), lambda x: torch.ceil(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.ceil_(), lambda x: torch.ceil(x))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_round(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.round(), lambda x: torch.round(x), from_=0, to=100)
    with mag.no_grad():
        unary_op(dtype, lambda x: x.round_(), lambda x: torch.round(x), from_=0, to=100)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_softmax(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.softmax(), lambda x: torch.softmax(x, dim=-1))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.softmax_(), lambda x: torch.softmax(x, dim=-1))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_sigmoid(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.sigmoid(), lambda x: torch.sigmoid(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.sigmoid_(), lambda x: torch.sigmoid(x))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_hard_sigmoid(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.hardsigmoid(), lambda x: torch.nn.functional.hardsigmoid(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.hardsigmoid_(), lambda x: torch.nn.functional.hardsigmoid(x))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_silu(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.silu(), lambda x: torch.nn.functional.silu(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.silu_(), lambda x: torch.nn.functional.silu(x))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_tanh(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.tanh(), lambda x: torch.tanh(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.tanh_(), lambda x: torch.tanh(x))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_relu(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.relu(), lambda x: torch.relu(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.relu_(), lambda x: torch.relu(x))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_unary_op_gelu(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.gelu(), lambda x: torch.nn.functional.gelu(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.gelu_(), lambda x: torch.nn.functional.gelu(x))

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.int32, mag.boolean])
def test_unary_op_tril(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.tril(), lambda x: torch.tril(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.tril_(), lambda x: x.tril_())

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.int32, mag.boolean])
def test_unary_op_triu(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: x.triu(), lambda x: torch.triu(x))
    with mag.no_grad():
        unary_op(dtype, lambda x: x.triu_(), lambda x: x.triu_())

@pytest.mark.parametrize('dtype', [mag.boolean, mag.int32])
def test_unary_op_not(dtype: mag.DataType) -> None:
    unary_op(dtype, lambda x: ~x, lambda x: ~x)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.int32])
def test_binary_op_add(dtype: mag.DataType) -> None:
    binary_op_square(dtype, lambda x, y: x + y)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.int32])
def test_binary_op_sub(dtype: mag.DataType) -> None:
    binary_op_square(dtype, lambda x, y: x + y)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.int32])
def test_binary_op_mul(dtype: mag.DataType) -> None:
    binary_op_square(dtype, lambda x, y: x * y)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32])
def test_binary_op_div_fp(dtype: mag.DataType) -> None:
    binary_op_square(dtype, lambda x, y: x / y)

def test_binary_op_div_int32() -> None:
    binary_op_square(mag.int32, lambda x, y: x // y, from_=1, to=10000)

@pytest.mark.parametrize('dtype', [mag.boolean, mag.int32])
def test_binary_op_and(dtype: mag.DataType) -> None:
    binary_op_square(dtype, lambda x, y: x & y)

@pytest.mark.parametrize('dtype', [mag.boolean, mag.int32])
def test_binary_op_or(dtype: mag.DataType) -> None:
    binary_op_square(dtype, lambda x, y: x | y)

@pytest.mark.parametrize('dtype', [mag.boolean, mag.int32])
def test_binary_op_xor(dtype: mag.DataType) -> None:
    binary_op_square(dtype, lambda x, y: x ^ y)

def test_binary_op_shl() -> None:
    binary_op_square(mag.int32, lambda x, y: x << y, from_=0, to=31)

def test_binary_op_shr() -> None:
    binary_op_square(mag.int32, lambda x, y: x >> y, from_=0, to=31)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.boolean, mag.int32])
def test_binary_op_eq(dtype: mag.DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x == y)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.boolean, mag.int32])
def test_binary_op_ne(dtype: mag.DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x != y)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.int32])
def test_binary_op_le(dtype: mag.DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x <= y)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.int32])
def test_binary_op_ge(dtype: mag.DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x >= y)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.int32])
def test_binary_op_lt(dtype: mag.DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x < y)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.int32])
def test_binary_op_gt(dtype: mag.DataType) -> None:
    binary_cmp_op(dtype, lambda x, y: x > y)
