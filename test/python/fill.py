# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import *
from .common import *

def test_tensor_fill_zero() -> None:
    tensor = Tensor.zeros(1, 2, 3, 4, 5, 6)
    data = tensor.tolist()
    assert nested_len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([x == 0 for x in flatten(data)])


def test_tensor_fill_x() -> None:
    tensor = Tensor.full(1, 2, 3, 4, 5, 6, fill_value=-22)
    data = tensor.tolist()
    assert nested_len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([x == -22 for x in flatten(data)])


def test_tensor_fill_uniform() -> None:
    tensor = Tensor.uniform(1, 2, 3, 4, 5, 6, from_=-1, to=1)
    data = tensor.tolist()
    assert nested_len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([-1 <= x <= 1 for x in flatten(data)])


def test_tensor_fill_uniform2() -> None:
    tensor = Tensor.uniform(1, 2, 3, 4, 5, 6, from_=0, to=100)
    data = tensor.tolist()
    assert nested_len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([0 <= x <= 100 for x in flatten(data)])


def test_tensor_fill_uniform3() -> None:
    tensor = Tensor.uniform(1, 2, 3, 4, 5, 6, from_=-1000, to=-20)
    data = tensor.tolist()
    assert nested_len(data) == 1 * 2 * 3 * 4 * 5 * 6
    assert all([-1000 <= x <= -20 for x in flatten(data)])


def test_tensor_fill_normal() -> None:
    mean = 0.0
    stddev = 1
    tensor = Tensor.normal(1, 2, 3, 4, 5, 6, mean=mean, std=stddev)
    data = tensor.tolist()
    assert nested_len(data) == 1 * 2 * 3 * 4 * 5 * 6
    # assert all([abs(x - mean) <= 3 * stddev for x in data]) TODO


def test_tensor_fill_const_1d() -> None:
    init = [1, 2, 3, 4]
    tensor = Tensor.of(init)
    assert tensor.shape == (4,)
    assert tensor.numel == 4
    assert tensor.rank == 1
    data = tensor.tolist()
    assert data == init


def test_tensor_fill_const_2d() -> None:
    init = [[1, 2], [3, 4]]
    tensor = Tensor.of(init)
    assert tensor.shape == (2, 2)
    assert tensor.numel == 4
    assert tensor.rank == 2
    data = tensor.tolist()
    assert data == init


def test_tensor_fill_const_3d() -> None:
    init = [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
    tensor = Tensor.of(init)
    assert tensor.shape == (2, 2, 2)
    assert tensor.numel == 8
    assert tensor.rank == 3
    data = tensor.tolist()
    assert data == init


def test_tensor_fill_const_4d() -> None:
    init = [[[[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]]
    tensor = Tensor.of(init)
    assert tensor.shape == (2, 2, 2, 2)
    assert tensor.numel == 16
    assert tensor.rank == 4
    data = tensor.tolist()
    assert data == init

def test_tensor_fill_const_5d() -> None:
    init = [
        [[[[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]],
        [[[[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]],
    ]
    tensor = Tensor.of(init)
    assert tensor.shape == (2, 2, 2, 2, 2)
    assert tensor.numel == 32
    assert tensor.rank == 5
    data = tensor.tolist()
    assert data == init


def test_tensor_fill_const_6d() -> None:
    init = [
        [
            [
                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
            ],
            [
                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
            ],
        ],
        [
            [
                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
            ],
            [
                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
            ],
        ],
    ]
    tensor = Tensor.of(init)
    assert tensor.shape == (2, 2, 2, 2, 2, 2)
    assert tensor.numel == 64
    assert tensor.rank == 6
    data = tensor.tolist()
    assert data == init
