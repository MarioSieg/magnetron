"""Float to integer casts follow one rule, for every input.

Converting a float to an integer type in C is undefined once the value does not fit, and that is
what the cast used to do, so the answer depended on which kernel handled the element: -2.0 to uint8
was 254 in a short tensor and 0 in a long one. NaN now becomes zero, the value saturates into the
32-bit range of the target's signedness, then truncates to the target width.

The two invariance tests are the ones that matter: implementations of this commonly answer
differently depending on the source dtype or on how many elements the tensor has, and torch 2.14
does both.
"""

import numpy as np
import pytest

from .common import *

INT_TARGETS = ('uint8', 'int8', 'uint16', 'int16', 'int32')

# value, target dtype name, expected. Named, not DType objects: pytest holds its parameter
# table past the point the bindings are torn down, and binding objects there are reported as leaks.
CASES = [
    (float('nan'), 'int32', 0),
    (float('inf'), 'int32', 2147483647),
    (float('-inf'), 'int32', -2147483648),
    (float('inf'), 'uint8', 255),
    (1.5, 'int8', 1),
    (-1.5, 'int8', -1),
    (-0.9, 'int8', 0),
    (200.7, 'uint8', 200),
    (-1.0, 'uint16', 0),  # saturates into the unsigned range; wrapping would give 65535
    (-1e10, 'uint16', 0),
    (70000.0, 'uint16', 70000 % 65536),  # inside 32 bits, so it narrows
    (1e10, 'uint16', 65535),  # saturated first, so the low bits are all ones
    (1e10, 'int32', 2147483647),
    (-1e10, 'int32', -2147483648),
]


def _cast(values: list[float], dt: dtype.DType, src: dtype.DType = dtype.float32) -> list[int]:
    t = Tensor(np.asarray(values, dtype=np.float32))
    if src != dtype.float32:
        t = t.cast(src)
    return [int(v) for v in t.cast(dt).numpy().ravel()]


@pytest.mark.parametrize('value,name,expected', CASES, ids=lambda v: str(v))
def test_rule(value: float, name: str, expected: int) -> None:
    assert _cast([value], getattr(dtype, name)) == [expected]


def test_answer_does_not_depend_on_the_source_float_type() -> None:
    values = [-1.0, -256.0, 0.0, 1.0, 200.0]
    for name in INT_TARGETS:
        dt = getattr(dtype, name)
        assert _cast(values, dt, src=dtype.float32) == _cast(values, dt, src=dtype.bfloat16), name


def test_answer_does_not_depend_on_tensor_length() -> None:
    for name in INT_TARGETS:
        dt = getattr(dtype, name)
        seen = {_cast([-1.0] * n, dt)[0] for n in (1, 2, 3, 4, 6, 8, 16, 17, 64)}
        assert len(seen) == 1, f'{name}: -1.0 cast differently depending on tensor length: {seen}'
