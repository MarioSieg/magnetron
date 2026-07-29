# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from .common import *


def test_negative_strides_reverse() -> None:
    x = Tensor.arange(5)
    assert not x.is_view
    y = x.strided_view((5,), (-1,), offset=4)
    assert y.is_view
    assert y.tolist() == [4, 3, 2, 1, 0]


def test_negative_strides_2d() -> None:
    x = Tensor.arange(12).reshape(3, 4)
    y = x.strided_view((3, 4), (4, -1), offset=3)
    assert y.tolist() == [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]


def test_negative_strides_mutation() -> None:
    x = Tensor.arange(5)
    y = x.strided_view((5,), (-1,), offset=4)
    y[0] = 99
    assert x.tolist() == [0, 1, 2, 3, 99]


def test_negative_strides_base_mutation() -> None:
    x = Tensor.arange(5)
    y = x.strided_view((5,), (-1,), offset=4)
    x[4] = 123
    assert y.tolist() == [123, 3, 2, 1, 0]


def test_negative_strides_slice() -> None:
    x = Tensor.arange(5)
    y = x.strided_view((5,), (-1,), offset=4)
    assert y[1:4].tolist() == [3, 2, 1]


def test_negative_strides_compose() -> None:
    x = Tensor.arange(12).reshape(3, 4)
    y = x.strided_view((3, 4), (4, -1), offset=3)
    z = y + 1
    assert z.tolist() == [
        [4, 3, 2, 1],
        [8, 7, 6, 5],
        [12, 11, 10, 9],
    ]


def test_negative_strides_double_reverse() -> None:
    x = Tensor.arange(5)
    y = x.strided_view((5,), (-1,), offset=4)
    z = y.strided_view((5,), (1,), offset=0)
    assert z.tolist() == [0, 1, 2, 3, 4]


def test_negative_strides_clone_removes_view() -> None:
    x = Tensor.arange(5)

    y = x.strided_view((5,), (-1,), offset=4)
    z = y.clone()

    assert y.is_view
    assert not z.is_view
    assert z.tolist() == [4, 3, 2, 1, 0]


def test_negative_strides_contiguous_normalizes_layout() -> None:
    x = Tensor.arange(5)

    y = x.strided_view((5,), (-1,), offset=4)
    c = y.contiguous()

    assert c.tolist() == [4, 3, 2, 1, 0]
    assert not c.is_view
    assert c.strides == (1,)


def test_negative_strides_transpose_then_reverse_rows() -> None:
    x = Tensor.arange(12).reshape(3, 4)

    xt = x.T
    y = xt.strided_view((4, 3), (-1, 4), offset=3)

    assert y.tolist() == [
        [3, 7, 11],
        [2, 6, 10],
        [1, 5, 9],
        [0, 4, 8],
    ]


def test_negative_strides_elementwise_after_transpose() -> None:
    x = Tensor.arange(12).reshape(3, 4)

    y = x.strided_view((3, 4), (4, -1), offset=3)
    z = y + 10

    assert z.tolist() == [
        [13, 12, 11, 10],
        [17, 16, 15, 14],
        [21, 20, 19, 18],
    ]
