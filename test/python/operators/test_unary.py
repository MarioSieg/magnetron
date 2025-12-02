# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

from ..common import *

_UNARY_OPS: tuple[str, ...] = (
    'not',
    'abs',
    'neg',
    'log',
    'log10',
    'log1p',
    'log2',
    'sqr',
    'sqrt',
    'sin',
    'asin',
    'sinh',
    'asinh',
    'cos',
    'acos',
    'cosh',
    'acosh',
    'tan',
    'atan',
    'tanh',
    'atanh',
    'step',
    'exp',
    'expm1',
    'exp2',
    'floor',
    'ceil',
    'round',
    'trunc',
    'softmax',
    'sigmoid',
    'hardsigmoid',
    'silu',
    'tanh',
    'gelu',
    'tril',
    'triu',
)


def unary_op(
    dtype: DataType,
    mag_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor],
    torch_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor],
) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = Tensor.bernoulli(shape) if dtype == boolean else Tensor.uniform(shape, dtype=dtype)
        r = mag_callback(x.clone())
        torch.testing.assert_close(totorch(r), torch_callback(totorch(x)).squeeze())

    for_all_shapes(test)


@pytest.mark.parametrize('dtype', FLOATING_POINT_DTYPES)
@pytest.mark.parametrize('op', _UNARY_OPS)
def test_unary_operators(op: str, dtype: DataType) -> None:
    unary_op(dtype, lambda x: getattr(x, op)(), lambda x: getattr(torch, op)(x) if hasattr(torch, op) else getattr(torch.functional.F, op)(x))
    unary_op(dtype, lambda x: getattr(x, op + '_')(), lambda x: getattr(torch, op)(x) if hasattr(torch, op) else getattr(torch.functional.F, op)(x))
