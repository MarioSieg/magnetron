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


def test_numbytes_is_the_extent_not_the_storage() -> None:
    # numbytes is numel*itemsize, so it shrinks with a view; storage_numbytes is the buffer the
    # view shares with its base and does not. Conflating the two makes a view's transfer, copy or
    # bounds check reach past the end of the tensor.
    base = Tensor.empty(64, dtype=dtype.uint8)
    assert base.numbytes == 64
    assert base.storage_numbytes == 64

    v = base.view_slice(0, 32, 16, 1)
    assert v.numbytes == 16
    assert v.storage_numbytes == 64
    assert v.data_ptr == base.data_ptr + 32

    f32 = Tensor.empty(8, dtype=dtype.float32)
    assert f32.numbytes == 32
    assert f32.view(2, 4).numbytes == 32, 'a reshape spans the same bytes'
