# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>
# This file tests each operator once against a manual, correct result

import pytest

from magnetron import dtype, Tensor
from .common import AVAILABLE_DEVICES


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_empty(device: str) -> None:
    x = Tensor.empty((2, 3), dtype=dtype.int64, device=device)
    assert x.shape == (2, 3)
    assert x.dtype == dtype.int64


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_strided_view(device: str) -> None:
    x = Tensor.arange(9, device=device).reshape(3, 3)
    t = Tensor.strided_view(x, (2, 2), (1, 2))
    assert t.shape == (2, 2)
    assert t.tolist() == [
        [0, 2],
        [1, 3],
    ]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_broadcast(device: str) -> None:
    x = Tensor([1, 2, 3], device=device)
    y = x.broadcast((3, 3))
    assert y.tolist() == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_expand(device: str) -> None:
    x = Tensor([[1], [2], [3]], device=device)
    assert x.shape == (3, 1)
    y = x.expand((3, 4))
    assert y.tolist() == [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
    y = x.expand(-1, 4)
    assert y.tolist() == [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_empty_like(device: str) -> None:
    x = Tensor.empty(2, 4, dtype=dtype.float8_e4m3fn, device=device)
    y = Tensor.empty_like(x)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() == y.tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_scalar(device: str) -> None:
    x = Tensor.scalar(2.5, device=device)
    assert x.rank == 0
    assert x.shape == ()
    assert x.strides == ()
    assert x.numel == 1
    assert x.item() == 2.5


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_full(device: str) -> None:
    x = Tensor.full(2, 3, fill_value=3.141592, device=device)
    assert x.shape == (2, 3)
    assert x.numel == 2 * 3
    assert x.rank == 2
    assert (x == 3.141592).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_full_like(device: str) -> None:
    x = Tensor.full(2, 4, fill_value=3.141592, device=device)
    y = Tensor.full_like(x, -3.141592)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() != y.tolist()
    assert (y == -3.141592).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_zeros(device: str) -> None:
    x = Tensor.zeros(2, 3, device=device)
    assert x.shape == (2, 3)
    assert x.numel == 2 * 3
    assert x.rank == 2
    assert (x == 0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_zeros_like(device: str) -> None:
    x = Tensor.uniform(2, 4, fill_value=3.141592, device=device)
    y = Tensor.zeros_like(x)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() != y.tolist()
    assert (y == 0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_ones(device: str) -> None:
    x = Tensor.ones(2, 3, device=device)
    assert x.shape == (2, 3)
    assert x.numel == 2 * 3
    assert x.rank == 2
    assert (x == 1).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_ones_like(device: str) -> None:
    x = Tensor.uniform(2, 4, device=device)
    y = Tensor.ones_like(x)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() != y.tolist()
    assert (y == 1).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_uniform(device: str) -> None:
    x = Tensor.uniform(2, 4, low=-10.0, hi=13.0, device=device)
    assert x.shape == (2, 4)
    assert x.numel == 2 * 4
    assert x.rank == 2
    assert (x <= 13.0).all() and (x >= -10.0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_uniform_like(device: str) -> None:
    x = Tensor.uniform(2, 4, low=-10.0, hi=13.0, device=device)
    y = Tensor.uniform_like(x, low=-10.0, hi=13.0)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert (x <= 13.0).all() and (x >= -10.0).all()
    assert (y <= 13.0).all() and (y >= -10.0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_normal(device: str) -> None:
    x = Tensor.normal(2, 4, mean=0.3, std=0.6, device=device)
    assert x.shape == (2, 4)
    assert x.numel == 2 * 4
    assert x.rank == 2


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_normal_like(device: str) -> None:
    x = Tensor.normal(2, 4, mean=0.3, std=0.6, device=device)
    y = Tensor.normal_like(x, mean=0.3, std=0.6)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_bernoulli(device: str) -> None:
    x = Tensor.bernoulli(2, 4, p=0.5, device=device)
    assert x.dtype == dtype.boolean
    assert x.shape == (2, 4)
    assert x.numel == 2 * 4
    assert x.rank == 2


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_bernoulli_like(device: str) -> None:
    x = Tensor.bernoulli(2, 4, p=0.5, device=device)
    y = Tensor.bernoulli_like(x, p=0.5)
    assert x.dtype == dtype.boolean
    assert y.dtype == dtype.boolean
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_arange(device: str) -> None:
    x = Tensor.arange(5, device=device)
    assert x.dtype == dtype.int64
    assert x.tolist() == [0, 1, 2, 3, 4]
    x = Tensor.arange(1, 4)
    assert x.tolist() == [1, 2, 3]
    x = Tensor.arange(1.0, 2.5, 0.5)
    assert x.tolist() == [1.0000, 1.5000, 2.0000]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_linspace(device: str) -> None:
    x = Tensor.linspace(3, 10, steps=5, device=device)
    assert x.tolist() == [3.0000, 4.7500, 6.5000, 8.2500, 10.0000]
    x = Tensor.linspace(-10, 10, steps=5)
    assert x.tolist() == [-10.0, -5.0, 0.0, 5.0, 10.0]
    x = Tensor.linspace(start=-10, end=10, steps=5)
    assert x.tolist() == [-10.0, -5.0, 0.0, 5.0, 10.0]
    x = Tensor.linspace(start=-10, end=10, steps=1)
    assert x.tolist() == [-10]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_meshgrid(device: str) -> None:
    x = Tensor([1, 2, 3], device=device)
    y = Tensor([4, 5, 6], device=device)
    gx, gy = Tensor.meshgrid(x, y, indexing='ij')
    assert gx.tolist() == [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    assert gy.tolist() == [[4, 5, 6], [4, 5, 6], [4, 5, 6]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_one_hot(device: str) -> None:
    x = (Tensor.arange(0, 5, device=device) % 3).one_hot()
    assert x.dtype == dtype.int64
    assert x.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    x = (Tensor.arange(0, 5, device=device) % 3).one_hot(num_classes=5)
    assert x.tolist() == [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
    x = (Tensor.arange(0, 6, device=device).view(3, 2) % 3).one_hot()
    assert x.tolist() == [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_rand_perm(device: str) -> None:
    x = Tensor.rand_perm(4, device=device)
    assert x.dtype == dtype.int64
    for i in range(x.numel):
        assert i in x.tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_inplace_copy(device: str) -> None:
    x = Tensor.uniform(2, 4, 6, device=device)
    y = Tensor.uniform_like(x)
    x.copy_(y)
    assert (x == y).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_inplace_zeros(device: str) -> None:
    x = Tensor.uniform(2, 4, 6, device=device)
    x.zero_()
    assert (x == 0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_inplace_ones(device: str) -> None:
    x = Tensor.uniform(2, 4, 6, device=device)
    x.one_()
    assert (x == 1).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_inplace_fill(device: str) -> None:
    x = Tensor.uniform(2, 4, 6, device=device)
    x.fill_(3.1415)
    assert (x == 3.1415).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_masked_fill_outplace(device: str) -> None:
    x = Tensor.uniform(2, 4, 6, device=device)
    mask = Tensor.ones_like(x).tril().cast(dtype.boolean)
    x = x.masked_fill(mask, 3.1415)
    assert (x == 3.1415).any()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_masked_fill_inplace(device: str) -> None:
    x = Tensor.uniform(2, 4, 6, device=device)
    mask = Tensor.ones_like(x).tril().cast(dtype.boolean)
    x.masked_fill_(mask, 3.1415)
    assert (x == 3.1415).any()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_inplace_uniform(device: str) -> None:
    x = Tensor.zeros(2, 4, 6, device=device)
    x.uniform_(low=-1.0, high=1.0)
    assert (x != 0).all()
    assert (x >= -1.0).all() and (x <= 1.0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_inplace_normal(device: str) -> None:
    x = Tensor.zeros(2, 4, 6, device=device)
    x.normal_(mean=0.5, std=1.0)
    assert (x != 0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_inplace_bernoulli(device: str) -> None:
    x = Tensor.ones(2, 4, 6, device=device).cast(dtype.boolean)
    x.bernoulli_(p=0.5)
    assert ((x ^ x) == 0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_clone(device: str) -> None:
    x = Tensor.uniform(2, 4, 8, 3, device=device)
    y = x.clone()
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() == y.tolist()
    assert x.data_storage_ptr != y.data_storage_ptr
    assert x.is_contiguous
    assert y.is_contiguous
    assert not x.is_view
    assert not y.is_view


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_cast(device: str) -> None:
    x = Tensor.uniform(2, 4, 8, 3, device=device, low=-10.0, high=10.0, dtype=dtype.float32)
    vals = x.flatten().tolist()
    y = x.cast(dtype.int16)
    assert y.dtype == dtype.int16
    assert y.data_storage_ptr != x.data_storage_ptr
    for i, v in enumerate(y.flatten().tolist()):
        assert v == int(vals[i])


# TODO: transfer


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_view(device: str) -> None:
    x = Tensor.normal(4, 4, device=device)
    assert not x.is_view
    assert x.shape == (4, 4)
    y = x.view(16)
    assert y.is_view
    assert y.shape == (16,)
    z = x.view(-1, 8)
    assert z.is_view
    assert z.shape == (2, 8)
    a = Tensor.normal(1, 2, 3, 4)
    assert a.shape == (1, 2, 3, 4)
    b = a.transpose(1, 2)
    assert b.shape == (1, 3, 2, 4)
    c = a.view(1, 2, 3, 4)
    assert a.shape == c.shape
    assert b.tolist() != c.tolist()


# TODO: view_slice


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_reshape(device: str) -> None:
    a = Tensor.arange(4.0, device=device)
    b = a.reshape((2, 2))
    assert b.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    b = Tensor([[0, 1], [2, 3]], device=device)
    c = b.reshape((-1,))
    assert c.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_transpose(device: str) -> None:
    x = Tensor([[33, 4, -10], [12, 100, -666]], device=device)
    assert x.shape == (2, 3)
    assert x.tolist() == [[33, 4, -10], [12, 100, -666]]
    y = x.transpose(0, 1)
    assert y.shape == (3, 2)
    assert y.tolist() == [[33, 12], [4, 100], [-10, -666]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_T(device: str) -> None:
    x = Tensor.normal((), device=device)
    x.shape == ()
    x.numel == 1
    assert x.T.shape == x.shape
    assert x.T.numel == x.numel
    assert x.T == x
    x = Tensor([[33, 4, -10], [12, 100, -666]], device=device)
    assert x.shape == (2, 3)
    assert x.tolist() == [[33, 4, -10], [12, 100, -666]]
    y = x.T
    assert y.shape == (3, 2)
    assert y.tolist() == [[33, 12], [4, 100], [-10, -666]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_permute(device: str) -> None:
    x = Tensor.normal(2, 3, 5, device=device)
    assert x.shape == (2, 3, 5)
    y = x.permute((2, 0, 1))
    assert y.shape == (5, 2, 3)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_contiguous(device: str) -> None:
    x = Tensor.normal(2, 3, 5, device=device)
    assert x.is_contiguous
    y = x.permute((2, 0, 1))
    assert not y.is_contiguous
    assert (y.contiguous() == y).all()
    assert y.contiguous().is_contiguous
    assert (y.contiguous() == y).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_squeeze(device: str) -> None:
    x = Tensor.zeros(2, 1, 2, 1, 2, device=device)
    assert x.shape == (2, 1, 2, 1, 2)
    y = x.squeeze()
    assert y.shape == (2, 2, 2)
    y = x.squeeze(0)
    assert y.shape == (2, 1, 2, 1, 2)
    y = x.squeeze(1)
    assert y.shape == (2, 2, 1, 2)
    # TODO: squeeze with tuple


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_unsqueeze(device: str) -> None:
    x = Tensor([1, 2, 3, 4], device=device)
    y = x.unsqueeze(0)
    assert y.tolist() == [[1, 2, 3, 4]]
    y = x.unsqueeze(1)
    assert y.tolist() == [[1], [2], [3], [4]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_flatten(device: str) -> None:
    x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=device)
    y = x.flatten()
    assert y.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    y = x.flatten(start_dim=1)
    assert y.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_unflatten(device: str) -> None:
    x = Tensor.normal(3, 4, 1, device=device)
    y = x.unflatten(1, (2, 2))
    assert y.shape == (3, 2, 2, 1)
    y = x.unflatten(1, (-1, 2))
    assert y.shape == (3, 2, 2, 1)
    x = Tensor.normal(5, 12, 3, device=device)
    y = x.unflatten(-2, (2, 2, 3, 1, 1))
    assert y.shape == (5, 2, 2, 3, 1, 1, 3)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_narrow(device: str) -> None:
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
    assert x.narrow(0, 0, 2).tolist() == [[1, 2, 3], [4, 5, 6]]
    assert x.narrow(1, 1, 2).tolist() == [[2, 3], [5, 6], [8, 9]]
    assert x.narrow(-1, -1, 1).tolist() == [[3], [6], [9]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_movedim(device: str) -> None:
    x = Tensor.normal(3, 2, 1, device=device)
    assert x.shape == (3, 2, 1)
    y = x.movedim(1, 0)
    assert y.shape == (2, 3, 1)
    # TODO: tuple movedim
    # y = x.movedim((1, 2), (0, 1))
    # assert y.shape == (2, 1, 3)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_select(device: str) -> None:
    pass  # TODO


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_split(device: str) -> None:
    x = Tensor.arange(10, device=device).reshape(5, 2)
    assert x.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    y: tuple[Tensor] = x.split(2)
    should: tuple[Tensor] = (Tensor([[0, 1], [2, 3]], device=device), Tensor([[4, 5], [6, 7]], device=device), Tensor([[8, 9]], device=device))
    assert len(y) == len(should)
    for i in range(len(y)):
        assert (y[i] == should[i]).all()
    # TODO: tuple split


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_eye(device: str) -> None:
    pass  # TODO


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_pad(device: str) -> None:
    pass  # TODO


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_cumsum(device: str) -> None:
    x = Tensor([13, 7, 3, 10, 13, 3, 15, 10, 9, 10], device=device)
    y = x.cusum(dim=0)
    assert y.tolist() == [13, 20, 23, 33, 46, 49, 64, 74, 83, 93]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_cumprod(device: str) -> None:
    x = Tensor([0.6001, 0.2069, -0.1919, 0.9792, 0.6727, 1.0062, 0.4126, -0.2129, -0.4206, 0.1968], device=device)
    y = x.cuprod(dim=0)
    should = Tensor([0.6001, 0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065, 0.0014, -0.0006, -0.0001], device=device)
    assert ((y - should).abs().max()) < 1e-2
    y[5] = 0.0
    should = Tensor([0.6001, 0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000, 0.0000, -0.0000, -0.0000], device=device)
    assert ((y - should).abs().max()) < 1e-2


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_cummax(device: str) -> None:
    x = Tensor([-0.3449, -1.5447, 0.0685, -1.5104, -1.1706, 0.2259, 1.4696, -1.3284, 1.9946, -0.8209], device=device)
    values, indices = x.cumax(dim=0)
    should = Tensor([-0.3449, -0.3449, 0.0685, 0.0685, 0.0685, 0.2259, 1.4696, 1.4696, 1.9946, 1.9946], device=device)
    assert ((values - should).abs().max()) < 1e-2
    assert indices.tolist() == [0, 0, 2, 2, 2, 5, 6, 6, 8, 8]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_cummin(device: str) -> None:
    x = Tensor([-0.2284, -0.6628, 0.0975, 0.2680, -1.3298, -0.4220, -0.3885, 1.1762, 0.9165, 1.6684], device=device)
    values, indices = x.cumin(dim=0)
    should = Tensor([-0.2284, -0.6628, -0.6628, -0.6628, -1.3298, -1.3298, -1.3298, -1.3298, -1.3298, -1.3298], device=device)
    assert ((values - should).abs().max()) < 1e-2
    assert indices.tolist() == [0, 1, 1, 1, 4, 4, 4, 4, 4, 4]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_repeat(device: str) -> None:
    x = Tensor([1, 2, 3], device=device)
    y = x.repeat(4, 2)
    assert y.tolist() == [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]
    assert x.repeat(4, 2, 1).shape == (4, 2, 3)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_repeat_interleave(device: str) -> None:
    x = Tensor([1, 2, 3], device=device)
    y = x.repeat_interleave(2)
    assert y.tolist() == [1, 1, 2, 2, 3, 3]
    x = Tensor([[1, 2], [3, 4]], device=device)
    y = x.repeat_interleave(2)
    assert y.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    y = x.repeat_interleave(3, dim=1)
    assert y.tolist() == [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]
    assert x.repeat_interleave(Tensor([1, 2], device=device), dim=0).tolist() == [[1, 2], [3, 4], [3, 4]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_index_add_(device: str) -> None:
    x = Tensor.ones(5, 3, device=device)
    t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype.float32, device=device)
    idx = Tensor([0, 4, 2], device=device)
    x.index_add_(0, idx, t)
    assert x.tolist() == [[2.0, 3.0, 4.0], [1.0, 1.0, 1.0], [8.0, 9.0, 10.0], [1.0, 1.0, 1.0], [5.0, 6.0, 7.0]]
    x.index_add_(0, idx, t, alpha=-1)
    assert x.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_outer(device: str) -> None:
    x = Tensor.arange(1.0, 5.0, device=device)
    y = Tensor.arange(1.0, 4.0, device=device)
    assert x.outer(y).tolist() == [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [4.0, 8.0, 12.0]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_lerp_outplace(device: str) -> None:
    start = Tensor.arange(1.0, 5.0, device=device)
    end = Tensor.full(4, fill_value=10.0, device=device)
    y = start.lerp(end, 0.5)
    assert y.tolist() == [5.5000, 6.0000, 6.5000, 7.0000]
    y = start.lerp(end, Tensor.full_like(start, 0.5))
    assert y.tolist() == [5.5000, 6.0000, 6.5000, 7.0000]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_index_add(device: str) -> None:
    x = Tensor.ones(5, 3)
    t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype.float32)
    index = Tensor([0, 4, 2])
    x.index_add_(0, index, t)
    assert x.tolist() == [[2.0, 3.0, 4.0], [1.0, 1.0, 1.0], [8.0, 9.0, 10.0], [1.0, 1.0, 1.0], [5.0, 6.0, 7.0]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_gather(device: str) -> None:
    x = Tensor([[1, 2], [3, 4]])
    ga = x.gather(1, Tensor([[0, 0], [1, 0]]))
    assert ga.tolist() == [[1, 1], [4, 3]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_scatter(device: str) -> None:
    x = Tensor.arange(1, 11).reshape(2, 5)
    assert x.tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    idx = Tensor([[0, 1, 2, 0]])
    y = Tensor.zeros(3, 5, dtype=x.dtype).scatter_(0, idx, x)
    assert y.tolist() == [[1, 0, 0, 4, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0]]
    idx = Tensor([[0, 1, 2], [0, 1, 4]])
    y = Tensor.zeros(3, 5, dtype=x.dtype).scatter_(1, idx, x)
    assert y.tolist() == [[1, 2, 3, 0, 0], [6, 7, 0, 0, 8], [0, 0, 0, 0, 0]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_scatter_add(device: str) -> None:
    src = Tensor.ones(2, 5, device=device)
    idx = Tensor([[0, 1, 2, 0, 0]], device=device)
    y = Tensor.zeros(3, 5, dtype=src.dtype, device=device).scatter_add_(0, idx, src)
    assert y.tolist() == [[1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]
    idx = Tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]], device=device)
    y = Tensor.zeros(3, 5, dtype=src.dtype, device=device).scatter_add_(0, idx, src)
    assert y.tolist() == [[2.0, 0.0, 0.0, 1.0, 1.0], [0.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0, 1.0]]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_flip(device: str) -> None:
    x = Tensor.arange(8).view(2, 2, 2)
    assert x.tolist() == [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    y = x.flip([0, 1])
    assert y.tolist() == [[[6, 7], [4, 5]], [[2, 3], [0, 1]]]
