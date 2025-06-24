# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import *

def test_tensor_clone() -> None:
    a = Tensor.of([[1, 2], [3, 4]])
    b = a.clone()
    assert a.shape == b.shape
    assert a.numel == b.numel
    assert a.rank == b.rank
    assert a.tolist() == b.tolist()
    assert a.is_contiguous == b.is_contiguous

def test_tensor_transpose() -> None:
    a = Tensor.full(2, 3, fill_value=1)
    b = a.transpose()
    assert a.shape == (2, 3)
    assert b.shape == (3, 2)
    assert a.numel == 6
    assert b.numel == 6
    assert a.rank == 2
    assert b.rank == 2
    assert a.tolist() == [1, 1, 1, 1, 1, 1]
    assert b.tolist() == [1, 1, 1, 1, 1, 1]
    assert a.is_contiguous
    assert not b.is_contiguous

def test_tensor_permute() -> None:
    a = Tensor.full(2, 3, fill_value=1)
    b = a.permute((1, 0))
    assert a.shape == (2, 3)
    assert b.shape == (3, 2)
    assert a.numel == 6
    assert b.numel == 6
    assert a.rank == 2
    assert b.rank == 2
    assert a.tolist() == [1, 1, 1, 1, 1, 1]
    assert b.tolist() == [1, 1, 1, 1, 1, 1]
    assert a.is_contiguous
    assert not b.is_contiguous

def test_tensor_permute_6d() -> None:
    a = Tensor.full(1, 2, 3, 4, 5, 6, fill_value=1)
    b = a.permute((5, 4, 3, 2, 1, 0))
    assert a.shape == (1, 2, 3, 4, 5, 6)
    assert b.shape == (6, 5, 4, 3, 2, 1)
    assert a.numel == 720
    assert b.numel == 720
    assert a.rank == 6
    assert b.rank == 6
    assert a.tolist() == [1] * 720
    assert b.tolist() == [1] * 720
    assert a.is_contiguous
    assert not b.is_contiguous
