"""One value, one answer, whichever kernel path it takes.

A CPU kernel reaches an element through one of three paths: the vectorized body, the scalar tail
that handles the remainder, or the generic strided walk. When those paths do not compute the same
function, the result depends on where the element happened to land - so tanh(0.7) came out
differently for a length-3 tensor than a length-4 one, and differently again through a transposed
view. The vector forms of tanh, sigmoid, silu, sin and tan approximate; the scalar ones called libm.

This sweeps every float unary op across lengths that straddle the vector width, and compares a
strided view against a contiguous tensor. It is the general form of the check that found the same
class of bug in the bfloat16 store.
"""

import numpy as np
import pytest

from .common import *

# Lengths chosen to land elements in the vectorized body, in the scalar remainder, and in both.
LENGTHS = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 32, 64, 65)

# Every float unary op reachable from the tensor API. softmax is excluded on purpose: it normalizes
# over the whole tensor, so depending on the length is its definition, not a bug.
UNARY_OPS = (
    'abs', 'sgn', 'neg', 'sqr', 'sqrt', 'rsqrt', 'rcp',
    'log', 'log2', 'log10', 'log1p', 'exp', 'exp2', 'expm1',
    'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'atan', 'asinh', 'atanh',
    'erf', 'erfc', 'step', 'floor', 'ceil', 'round', 'trunc',
    'sigmoid', 'hard_sigmoid', 'silu', 'relu', 'gelu', 'gelu_approx',
)


def _bits(t: Tensor, index: int = 0) -> int:
    flat = t.cast(dtype.float32).numpy().ravel().astype(np.float32)
    return int(flat[index].view(np.uint32))


@pytest.mark.parametrize('op', UNARY_OPS)
@pytest.mark.parametrize('value', [0.7, 1.3, 0.25])
def test_result_does_not_depend_on_tensor_length(op: str, value: float) -> None:
    """An element must compute the same whether it lands in the vector body or the remainder."""
    seen = {}
    for n in LENGTHS:
        t = Tensor(np.full((n,), value, dtype=np.float32))
        seen.setdefault(_bits(getattr(t, op)()), []).append(n)
    assert len(seen) == 1, f'{op}({value}) varies with length: ' + str(
        {hex(bits): lens for bits, lens in seen.items()}
    )


@pytest.mark.parametrize('op', UNARY_OPS)
def test_strided_matches_contiguous(op: str) -> None:
    """The generic strided walk must compute the same function as the vectorized path."""
    base = np.full((8, 8), 0.7, dtype=np.float32)
    t = Tensor(base)
    assert _bits(getattr(t, op)()) == _bits(getattr(t.T, op)()), f'{op} differs through a strided view'
