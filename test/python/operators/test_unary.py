# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

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
    #UnaryOpTestCase('not', None),
    UnaryOpTestCase('abs', None),
    UnaryOpTestCase('neg', None),
    UnaryOpTestCase('log', None),
    UnaryOpTestCase('log10', None),
    UnaryOpTestCase('log1p', None),
    UnaryOpTestCase('log2', None),
    UnaryOpTestCase('sqr', lambda x: x*x),
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
    UnaryOpTestCase('tanh', None),
    UnaryOpTestCase('gelu', None),
    UnaryOpTestCase('tril', None, 2),
    UnaryOpTestCase('triu', None, 2)
)

def unary_op(
    dtype: DataType,
    rank_min: int,
    mag_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor],
    torch_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor]
) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) < rank_min:
            return
        x = random_tensor(shape, dtype)
        r = mag_callback(x.clone())
        torch.testing.assert_close(totorch(r), torch_callback(totorch(x)), equal_nan=True)

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', FLOATING_POINT_DTYPES)
@pytest.mark.parametrize('op', _UNARY_OPS)
def test_unary_op(op: UnaryOpTestCase, dtype: DataType) -> None:
    name = op.name
    if op.torch_callback is not None:
        torch_op = op.torch_callback
    elif hasattr(torch, name):
        torch_op = getattr(torch, name)
    elif hasattr(torch.nn.functional, name):
        torch_op = getattr(torch.nn.functional, name)
    else:
        raise RuntimeError(f"No reference torch op found for unary op {name!r}")
    unary_op(dtype, op.rank_min, lambda x: getattr(x, name)(), lambda x: torch_op(x))
    if op.inplace:
        unary_op(dtype, op.rank_min, lambda x: getattr(x, name + '_')(), lambda x: torch_op(x))
