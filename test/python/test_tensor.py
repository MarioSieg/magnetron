# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from .common import *


def test_tensor_creation() -> None:
    tensor = Tensor.empty(1, 2, 3, 4, 5, 6)
    assert tensor.shape == (1, 2, 3, 4, 5, 6)
    assert tensor.numel == (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.numbytes == 4 * (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.data_ptr != 0
    assert tensor.is_contiguous is True
    assert tensor.dtype == dtype.float32


def test_tensor_numpy_roundtrip() -> None:
    pass  # TODO
