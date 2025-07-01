# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from ..common import *

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_abs(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.abs(), lambda x: torch.abs(x))
    with no_grad():
        unary_op(dtype, lambda x: x.abs_(), lambda x: torch.abs(x))


@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_neg(dtype: DataType) -> None:
    unary_op(dtype, lambda x: -x, lambda x: -x)


@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_log(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.log(), lambda x: torch.log(x), low=0, high=1000)
    with no_grad():
        unary_op(dtype, lambda x: x.log_(), lambda x: torch.log(x), low=0, high=1000)


@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_sqr(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.sqr(), lambda x: x * x)
    with no_grad():
        unary_op(dtype, lambda x: x.sqr_(), lambda x: x * x)


@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_sqrt(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.sqrt(), lambda x: torch.sqrt(x), low=0, high=1000)
    with no_grad():
        unary_op(dtype, lambda x: x.sqrt_(), lambda x: torch.sqrt(x), low=0, high=1000)


@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_sin(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.sin(), lambda x: torch.sin(x))
    with no_grad():
        unary_op(dtype, lambda x: x.sin_(), lambda x: torch.sin(x))


@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_cos(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.cos(), lambda x: torch.cos(x))
    with no_grad():
        unary_op(dtype, lambda x: x.cos_(), lambda x: torch.cos(x))


@pytest.mark.parametrize('dtype', [float32]) # Heaviside is not supported for fp16 in Torch, magnetron supports it tough
def test_unary_op_step(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.step(), lambda x: torch.heaviside(x, torch.tensor([0.0])))
    with no_grad():
        unary_op(dtype, lambda x: x.step_(), lambda x: torch.heaviside(x, torch.tensor([0.0])))


@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_exp(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.exp(), lambda x: torch.exp(x))
    with no_grad():
        unary_op(dtype, lambda x: x.exp_(), lambda x: torch.exp(x))

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_floor(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.floor(), lambda x: torch.floor(x))
    with no_grad():
        unary_op(dtype, lambda x: x.floor_(), lambda x: torch.floor(x))

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_ceil(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.ceil(), lambda x: torch.ceil(x))
    with no_grad():
        unary_op(dtype, lambda x: x.ceil_(), lambda x: torch.ceil(x))

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_round(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.round(), lambda x: torch.round(x), low=0, high=100)
    with no_grad():
        unary_op(dtype, lambda x: x.round_(), lambda x: torch.round(x), low=0, high=100)

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_softmax(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.softmax(), lambda x: torch.softmax(x, dim=-1))
    with no_grad():
        unary_op(dtype, lambda x: x.softmax_(), lambda x: torch.softmax(x, dim=-1))

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_sigmoid(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.sigmoid(), lambda x: torch.sigmoid(x))
    with no_grad():
        unary_op(dtype, lambda x: x.sigmoid_(), lambda x: torch.sigmoid(x))

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_hard_sigmoid(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.hardsigmoid(), lambda x: torch.nn.functional.hardsigmoid(x))
    with no_grad():
        unary_op(dtype, lambda x: x.hardsigmoid_(), lambda x: torch.nn.functional.hardsigmoid(x))

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_silu(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.silu(), lambda x: torch.nn.functional.silu(x))
    with no_grad():
        unary_op(dtype, lambda x: x.silu_(), lambda x: torch.nn.functional.silu(x))

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_tanh(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.tanh(), lambda x: torch.tanh(x))
    with no_grad():
        unary_op(dtype, lambda x: x.tanh_(), lambda x: torch.tanh(x))

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_relu(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.relu(), lambda x: torch.relu(x))
    with no_grad():
        unary_op(dtype, lambda x: x.relu_(), lambda x: torch.relu(x))

@pytest.mark.parametrize('dtype', [float16, float32])
def test_unary_op_gelu(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.gelu(), lambda x: torch.nn.functional.gelu(x))
    with no_grad():
        unary_op(dtype, lambda x: x.gelu_(), lambda x: torch.nn.functional.gelu(x))

@pytest.mark.parametrize('dtype', [float16, float32, int32, boolean])
def test_unary_op_tril(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.tril(), lambda x: torch.tril(x))
    with no_grad():
        unary_op(dtype, lambda x: x.tril_(), lambda x: x.tril_())

@pytest.mark.parametrize('dtype', [float16, float32, int32, boolean])
def test_unary_op_triu(dtype: DataType) -> None:
    unary_op(dtype, lambda x: x.triu(), lambda x: torch.triu(x))
    with no_grad():
        unary_op(dtype, lambda x: x.triu_(), lambda x: x.triu_())
@pytest.mark.parametrize('dtype', [boolean, int32])
def test_unary_op_not(dtype: DataType) -> None:
    unary_op(dtype, lambda x: ~x, lambda x: ~x)