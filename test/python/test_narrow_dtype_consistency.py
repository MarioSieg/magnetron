"""A narrow-dtype result must not depend on how many elements the tensor has.

The CPU kernels process a vector body and then a scalar tail. If the two disagree about rounding,
the same value computes differently depending on where it lands, and a length-3 tensor disagrees
with a length-4 one. That happened: the NEON bfloat16 store truncated with a bare shift while the
scalar path rounded to nearest even.

The lengths below straddle every vector width and tail size, so a value is forced through both
paths, and the last test pins the rounding rule itself rather than only its consistency - a store
that truncated everywhere would be perfectly consistent and still wrong.
"""

import operator

import numpy as np
import pytest

from .common import *

BINARY_OPS = (operator.add, operator.sub, operator.mul, operator.truediv)

# Lengths that put elements in the vector body, in the scalar remainder, and in both.
LENGTHS = (1, 2, 3, 4, 5, 8, 16, 64, 257)

# Parametrized by name rather than by DType: pytest keeps its parameter table alive past the
# point where the bindings are torn down, and holding binding objects there is reported as a leak.
NARROW = ('bfloat16', 'float16')


def _first_element_bits(t: Tensor) -> int:
    """Raw storage bits of element 0, read back losslessly by widening to float32."""
    return int(t.cast(dtype.float32).numpy().ravel()[0].view(np.uint32))


def _filled(value: float, n: int, dt: dtype.DType) -> Tensor:
    return Tensor(np.full((n,), value, dtype=np.float32)).cast(dt)


def _truncated_to_bfloat16(x: float) -> float:
    """What a bare shift would store: the low 16 bits dropped, never rounded up."""
    u = np.float32(x).view(np.uint32)
    return float((u & np.uint32(0xFFFF0000)).view(np.float32))


def _nearest_bfloat16(x: float) -> float:
    """Round to nearest, ties to even: add half a unit in the last place, plus one when the value
    kept is odd, then drop the low 16 bits."""
    u = np.uint64(np.float32(x).view(np.uint32))
    u = np.uint32((u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) & np.uint64(0xFFFF0000))
    return float(u.view(np.float32))


@pytest.mark.parametrize('name', NARROW)
@pytest.mark.parametrize('value', [0.012, 0.7, 1.3, 3.14159, -0.012, 1e-5, 12345.0])
def test_scalar_multiply_is_length_independent(name: str, value: float) -> None:
    dt = getattr(dtype, name)
    seen = {_first_element_bits(0.9 * _filled(value, n, dt)) for n in LENGTHS}
    assert len(seen) == 1, f'{name} scalar multiply of {value} varies with length: {[hex(b) for b in seen]}'


@pytest.mark.parametrize('name', NARROW)
@pytest.mark.parametrize('op', BINARY_OPS, ids=lambda f: f.__name__)
def test_elementwise_ops_are_length_independent(name: str, op) -> None:
    dt = getattr(dtype, name)
    seen = {_first_element_bits(op(_filled(1.4711, n, dt), _filled(0.3137, n, dt))) for n in LENGTHS}
    assert len(seen) == 1, f'{name} {op.__name__} varies with length: {[hex(b) for b in seen]}'


@pytest.mark.parametrize('name', NARROW)
def test_store_rounds_to_nearest(name: str) -> None:
    """0.012 sits between two representable values, closer to the upper one, so truncating and
    rounding give different answers here. Consistency alone would not catch a store that truncated
    everywhere."""
    dt = getattr(dtype, name)
    if name == 'bfloat16':
        nearest, truncated = _nearest_bfloat16(0.012), _truncated_to_bfloat16(0.012)
    else:
        nearest = float(np.float32(np.float16(np.float32(0.012))))  # numpy rounds to nearest even
        truncated = None
    assert truncated != nearest, 'pick a constant where the two rules disagree'
    for n in LENGTHS:
        stored = float(_filled(0.012, n, dt).cast(dtype.float32).numpy().ravel()[0])
        assert stored == nearest, f'{name} length {n}: stored {stored}, nearest is {nearest}'
