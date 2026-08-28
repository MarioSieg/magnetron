# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>
# This file tests each operator once against a manual, correct result

from collections.abc import Callable

import pytest
import torch

from magnetron import dtype, Tensor
from .common import AVAILABLE_DEVICES, assert_close_mag_torch, totorch


def skip_if_op_missing_on_cuda(device: str, op: str) -> None:
    if device != 'cpu':
        pytest.skip(f'operator {op} is not implemented in the CUDA backend')


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


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_transfer(device: str) -> None:
    x = Tensor.uniform(2, 4, 3, device='cpu')
    y = x.transfer(device)
    assert y.device.startswith(device)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert_close_mag_torch(y.transfer('cpu'), totorch(x), dtype.float32)


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


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_view_slice(device: str) -> None:
    x = Tensor.arange(12, device=device).reshape(3, 4)
    t = totorch(x)
    y = x.view_slice(0, 1, 2, 1)
    assert y.is_view
    assert y.tolist() == t[1:3].tolist()
    assert x.view_slice(1, 0, 2, 2).tolist() == t[:, 0:4:2].tolist()
    assert x.view_slice(-1, 1, 3, 1).tolist() == t[:, 1:4].tolist()


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
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    t = totorch(x)
    assert_close_mag_torch(x.select(0, 1), t.select(0, 1), dtype.float32)
    assert_close_mag_torch(x.select(1, 2), t.select(1, 2), dtype.float32)
    assert_close_mag_torch(x.select(-1, 0), t.select(-1, 0), dtype.float32)


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
    assert_close_mag_torch(Tensor.eye(3, device=device), torch.eye(3), dtype.float32)
    assert_close_mag_torch(Tensor.eye(2, 4, device=device), torch.eye(2, 4), dtype.float32)
    x = Tensor.eye(3, dtype=dtype.int32, device=device)
    assert x.dtype == dtype.int32
    assert x.tolist() == torch.eye(3, dtype=torch.int32).tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_pad(device: str) -> None:
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    t = totorch(x)
    assert_close_mag_torch(x.pad([1, 2]), torch.nn.functional.pad(t, [1, 2]), dtype.float32)
    assert_close_mag_torch(x.pad([1, 1], value=-1.0), torch.nn.functional.pad(t, [1, 1], value=-1.0), dtype.float32)
    assert_close_mag_torch(x.pad([2, 1], mode='reflect'), torch.nn.functional.pad(t.unsqueeze(0), [2, 1], mode='reflect').squeeze(0), dtype.float32)
    assert_close_mag_torch(
        x.pad([2, 1], mode='replicate'), torch.nn.functional.pad(t.unsqueeze(0), [2, 1], mode='replicate').squeeze(0), dtype.float32
    )


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


_UNARY_GENERAL: list[float] = [-2.75, -1.25, -0.5, 0.0, 0.5, 1.25, 2.75]
_UNARY_POSITIVE: list[float] = [0.125, 0.5, 1.0, 2.0, 3.5, 7.25]
_UNARY_UNIT: list[float] = [-0.875, -0.5, -0.125, 0.0, 0.125, 0.5, 0.875]
_UNARY_ABOVE_ONE: list[float] = [1.0, 1.5, 2.0, 4.0, 9.0]
_UNARY_NO_TIES: list[float] = [-2.7, -1.2, -0.4, 0.0, 0.4, 1.2, 2.7]

_UNARY_OPS: tuple[tuple[str, list[float], Callable[[torch.Tensor], torch.Tensor]], ...] = (
    ('abs', _UNARY_GENERAL, torch.abs),
    ('sgn', _UNARY_GENERAL, torch.sign),
    ('neg', _UNARY_GENERAL, torch.neg),
    ('log', _UNARY_POSITIVE, torch.log),
    ('log10', _UNARY_POSITIVE, torch.log10),
    ('log1p', _UNARY_POSITIVE, torch.log1p),
    ('log2', _UNARY_POSITIVE, torch.log2),
    ('sqr', _UNARY_GENERAL, torch.square),
    ('rcp', _UNARY_POSITIVE, torch.reciprocal),
    ('sqrt', _UNARY_POSITIVE, torch.sqrt),
    ('rsqrt', _UNARY_POSITIVE, torch.rsqrt),
    ('sin', _UNARY_GENERAL, torch.sin),
    ('cos', _UNARY_GENERAL, torch.cos),
    ('tan', _UNARY_GENERAL, torch.tan),
    ('sinh', _UNARY_GENERAL, torch.sinh),
    ('cosh', _UNARY_GENERAL, torch.cosh),
    ('tanh', _UNARY_GENERAL, torch.tanh),
    ('asin', _UNARY_UNIT, torch.asin),
    ('acos', _UNARY_UNIT, torch.acos),
    ('atan', _UNARY_GENERAL, torch.atan),
    ('asinh', _UNARY_GENERAL, torch.asinh),
    ('acosh', _UNARY_ABOVE_ONE, torch.acosh),
    ('atanh', _UNARY_UNIT, torch.atanh),
    ('step', _UNARY_GENERAL, lambda t: (t > 0.0).to(t.dtype)),
    ('erf', _UNARY_GENERAL, torch.erf),
    ('erfc', _UNARY_GENERAL, torch.erfc),
    ('exp', _UNARY_GENERAL, torch.exp),
    ('exp2', _UNARY_GENERAL, torch.exp2),
    ('expm1', _UNARY_GENERAL, torch.expm1),
    ('floor', _UNARY_GENERAL, torch.floor),
    ('ceil', _UNARY_GENERAL, torch.ceil),
    ('round', _UNARY_NO_TIES, torch.round),
    ('trunc', _UNARY_GENERAL, torch.trunc),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, values, ref', _UNARY_OPS, ids=[op[0] for op in _UNARY_OPS])
def test_unary_op(device: str, name: str, values: list[float], ref: Callable[[torch.Tensor], torch.Tensor]) -> None:
    x = Tensor(values, device=device)
    expected = ref(totorch(x))
    assert_close_mag_torch(getattr(x, name)(), expected, dtype.float32)
    y = x.clone()
    getattr(y, f'{name}_')()
    assert_close_mag_torch(y, expected, dtype.float32)


_ACTIVATION_OPS: tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor]], ...] = (
    ('softmax', lambda t: torch.softmax(t, -1)),
    ('sigmoid', torch.sigmoid),
    ('hard_sigmoid', torch.nn.functional.hardsigmoid),
    ('silu', torch.nn.functional.silu),
    ('relu', torch.relu),
    ('gelu', lambda t: torch.nn.functional.gelu(t, approximate='none')),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, ref', _ACTIVATION_OPS, ids=[op[0] for op in _ACTIVATION_OPS])
def test_activation_op(device: str, name: str, ref: Callable[[torch.Tensor], torch.Tensor]) -> None:
    x = Tensor(_UNARY_GENERAL, device=device)
    expected = ref(totorch(x))
    assert_close_mag_torch(getattr(x, name)(), expected, dtype.float32)
    y = x.clone()
    getattr(y, f'{name}_')()
    assert_close_mag_torch(y, expected, dtype.float32)


def _torch_grad(ref: Callable[[torch.Tensor], torch.Tensor], values: list[float]) -> torch.Tensor:
    t = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    ref(t).sum().backward()
    return t.grad


_ACTIVATION_DV_OPS: tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor]], ...] = (
    ('sigmoid_dv', torch.sigmoid),
    ('silu_dv', torch.nn.functional.silu),
    ('tanh_dv', torch.tanh),
    ('relu_dv', torch.relu),
)

@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, ref', _ACTIVATION_DV_OPS, ids=[op[0] for op in _ACTIVATION_DV_OPS])
def test_activation_dv_op(device: str, name: str, ref: Callable[[torch.Tensor], torch.Tensor]) -> None:
    x = Tensor(_UNARY_GENERAL, device=device)
    expected = _torch_grad(ref, _UNARY_GENERAL)
    assert_close_mag_torch(getattr(x, name)(), expected, dtype.float32)
    y = x.clone()
    getattr(y, f'{name}_')()
    assert_close_mag_torch(y, expected, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_gelu_approx(device: str) -> None:
    x = Tensor(_UNARY_GENERAL, device=device)
    expected = torch.nn.functional.gelu(totorch(x), approximate='tanh')
    assert_close_mag_torch(x.gelu_approx(), expected, dtype.float32)
    y = x.clone()
    y.gelu_approx_()
    assert_close_mag_torch(y, expected, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_gelu_dv(device: str) -> None:
    x = Tensor(_UNARY_GENERAL, device=device)
    expected = _torch_grad(lambda t: torch.nn.functional.gelu(t, approximate='none'), _UNARY_GENERAL)
    assert_close_mag_torch(x.gelu_dv(), expected, dtype.float32)
    y = x.clone()
    y.gelu_dv_()
    assert_close_mag_torch(y, expected, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_softmax_dv(device: str) -> None:
    x = Tensor(_UNARY_GENERAL, device=device)
    s = torch.softmax(totorch(x), -1)
    expected = s * (1.0 - s)
    assert_close_mag_torch(x.softmax_dv(), expected, dtype.float32)
    y = x.clone()
    y.softmax_dv_()
    assert_close_mag_torch(y, expected, dtype.float32)


_BIN_LHS: list[list[float]] = [[-3.0, 2.0, 5.5], [7.0, -4.25, 1.5]]
_BIN_RHS: list[list[float]] = [[2.0, -3.0, 1.25], [-5.0, 3.0, 2.5]]
_BIN_POS: list[list[float]] = [[3.0, 2.0, 5.5], [7.0, 4.25, 1.5]]

_BINARY_OPS: tuple[tuple[str, Callable, list[list[float]], dtype.DType], ...] = (
    ('add', lambda a, b: a + b, _BIN_LHS, dtype.float32),
    ('sub', lambda a, b: a - b, _BIN_LHS, dtype.float32),
    ('mul', lambda a, b: a * b, _BIN_LHS, dtype.float32),
    ('truediv', lambda a, b: a / b, _BIN_LHS, dtype.float32),
    ('floordiv', lambda a, b: a // b, _BIN_LHS, dtype.int32),
    ('mod', lambda a, b: a % b, _BIN_LHS, dtype.float32),
    ('pow', lambda a, b: a**b, _BIN_POS, dtype.float32),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, ref, lhs, inplace_dt', _BINARY_OPS, ids=[op[0] for op in _BINARY_OPS])
def test_binary_op(device: str, name: str, ref: Callable, lhs: list[list[float]], inplace_dt: dtype.DType) -> None:
    x = Tensor(lhs, device=device)
    y = Tensor(_BIN_RHS, device=device)
    tx, ty = totorch(x), totorch(y)
    expected = ref(tx, ty)
    assert_close_mag_torch(ref(x, y), expected, dtype.float32)
    assert_close_mag_torch(getattr(x, name)(y), expected, dtype.float32)
    assert_close_mag_torch(ref(x, 2.0), ref(tx, 2.0), dtype.float32)
    assert_close_mag_torch(ref(2.0, x), ref(2.0, tx), dtype.float32)
    z = Tensor(lhs, device=device).cast(inplace_dt)
    w = Tensor(_BIN_RHS, device=device).cast(inplace_dt)
    inplace_expected = ref(totorch(z), totorch(w))
    getattr(z, f'{name}_')(w)
    assert_close_mag_torch(z, inplace_expected, inplace_dt)


_COMPARE_OPS: tuple[tuple[str, Callable], ...] = (
    ('eq', lambda a, b: a == b),
    ('ne', lambda a, b: a != b),
    ('lt', lambda a, b: a < b),
    ('le', lambda a, b: a <= b),
    ('gt', lambda a, b: a > b),
    ('ge', lambda a, b: a >= b),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, ref', _COMPARE_OPS, ids=[op[0] for op in _COMPARE_OPS])
def test_compare_op(device: str, name: str, ref: Callable) -> None:
    x = Tensor([[-3.0, 2.0, 5.5], [7.0, -4.25, 1.5]], device=device)
    y = Tensor([[-3.0, -3.0, 1.25], [7.0, 3.0, 2.5]], device=device)
    tx, ty = totorch(x), totorch(y)
    got = ref(x, y)
    assert got.dtype == dtype.boolean
    assert got.tolist() == ref(tx, ty).tolist()
    assert getattr(x, name)(y).tolist() == ref(tx, ty).tolist()
    assert ref(x, 2.0).tolist() == ref(tx, 2.0).tolist()


_BITWISE_OPS: tuple[tuple[str, Callable], ...] = (
    ('logical_and', lambda a, b: a & b),
    ('logical_or', lambda a, b: a | b),
    ('logical_xor', lambda a, b: a ^ b),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, ref', _BITWISE_OPS, ids=[op[0] for op in _BITWISE_OPS])
def test_bitwise_op(device: str, name: str, ref: Callable) -> None:
    x = Tensor([[6, 3, 12], [255, 0, 9]], dtype=dtype.int32, device=device)
    y = Tensor([[3, 5, 10], [15, 7, 9]], dtype=dtype.int32, device=device)
    expected = ref(totorch(x), totorch(y))
    assert ref(x, y).tolist() == expected.tolist()
    assert getattr(x, name)(y).tolist() == expected.tolist()
    z = x.clone()
    getattr(z, f'{name}_')(y)
    assert z.tolist() == expected.tolist()
    a = Tensor([True, True, False, False], device=device)
    b = Tensor([True, False, True, False], device=device)
    assert ref(a, b).tolist() == ref(totorch(a), totorch(b)).tolist()


_SHIFT_OPS: tuple[tuple[str, Callable], ...] = (
    ('lshift', lambda a, b: a << b),
    ('rshift', lambda a, b: a >> b),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, ref', _SHIFT_OPS, ids=[op[0] for op in _SHIFT_OPS])
def test_shift_op(device: str, name: str, ref: Callable) -> None:
    x = Tensor([[-3, 2, 12], [7, -4, 1]], dtype=dtype.int32, device=device)
    y = Tensor([[2, 3, 1], [5, 3, 2]], dtype=dtype.int32, device=device)
    expected = ref(totorch(x), totorch(y))
    assert ref(x, y).tolist() == expected.tolist()
    assert getattr(x, name)(y).tolist() == expected.tolist()
    z = x.clone()
    getattr(z, f'{name}_')(y)
    assert z.tolist() == expected.tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_matmul(device: str) -> None:
    x = Tensor.uniform(3, 4, device=device)
    y = Tensor.uniform(4, 5, device=device)
    expected = totorch(x) @ totorch(y)
    assert_close_mag_torch(x @ y, expected, dtype.float32)
    assert_close_mag_torch(x.matmul(y), expected, dtype.float32)
    a = Tensor.uniform(2, 3, 4, device=device)
    b = Tensor.uniform(2, 4, 6, device=device)
    assert_close_mag_torch(a @ b, totorch(a) @ totorch(b), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_mean(device: str) -> None:
    x = Tensor.uniform(2, 3, 4, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.mean(), t.mean(), dtype.float32)
    assert_close_mag_torch(x.mean(1), t.mean(1), dtype.float32)
    assert_close_mag_torch(x.mean(1, keepdim=True), t.mean(1, keepdim=True), dtype.float32)
    assert_close_mag_torch(x.mean((0, 2)), t.mean((0, 2)), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_sum(device: str) -> None:
    x = Tensor.uniform(2, 3, 4, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.sum(), t.sum(), dtype.float32)
    assert_close_mag_torch(x.sum(1), t.sum(1), dtype.float32)
    assert_close_mag_torch(x.sum(1, keepdim=True), t.sum(1, keepdim=True), dtype.float32)
    assert_close_mag_torch(x.sum((0, 2)), t.sum((0, 2)), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_prod(device: str) -> None:
    x = Tensor.uniform(2, 3, 4, low=0.5, high=1.5, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.prod(), t.prod(), dtype.float32)
    assert_close_mag_torch(x.prod(1), t.prod(1), dtype.float32)
    assert_close_mag_torch(x.prod(1, keepdim=True), t.prod(1, keepdim=True), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_minima(device: str) -> None:
    x = Tensor.uniform(2, 3, 4, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.min(), t.min(), dtype.float32)
    assert_close_mag_torch(x.min(1), t.min(1).values, dtype.float32)
    assert_close_mag_torch(x.min(1, keepdim=True), t.min(1, keepdim=True).values, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_maxima(device: str) -> None:
    x = Tensor.uniform(2, 3, 4, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.max(), t.max(), dtype.float32)
    assert_close_mag_torch(x.max(1), t.max(1).values, dtype.float32)
    assert_close_mag_torch(x.max(1, keepdim=True), t.max(1, keepdim=True).values, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_min(device: str) -> None:
    skip_if_op_missing_on_cuda(device, 'MIN')
    x = Tensor.uniform(2, 3, 4, device=device)
    y = Tensor.uniform(2, 3, 4, device=device)
    assert_close_mag_torch(x.min(y), torch.minimum(totorch(x), totorch(y)), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_max(device: str) -> None:
    skip_if_op_missing_on_cuda(device, 'MAX')
    x = Tensor.uniform(2, 3, 4, device=device)
    y = Tensor.uniform(2, 3, 4, device=device)
    assert_close_mag_torch(x.max(y), torch.maximum(totorch(x), totorch(y)), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_argmin(device: str) -> None:
    x = Tensor.uniform(2, 3, 4, device=device)
    t = totorch(x)
    assert x.argmin().dtype == dtype.int64
    assert x.argmin().tolist() == t.argmin().tolist()
    assert x.argmin(1).tolist() == t.argmin(1).tolist()
    assert x.argmin(1, keepdim=True).tolist() == t.argmin(1, keepdim=True).tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_argmax(device: str) -> None:
    x = Tensor.uniform(2, 3, 4, device=device)
    t = totorch(x)
    assert x.argmax().dtype == dtype.int64
    assert x.argmax().tolist() == t.argmax().tolist()
    assert x.argmax(1).tolist() == t.argmax(1).tolist()
    assert x.argmax(1, keepdim=True).tolist() == t.argmax(1, keepdim=True).tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_all(device: str) -> None:
    x = Tensor.bernoulli(2, 3, 4, p=0.5, device=device)
    t = totorch(x)
    assert x.all().tolist() == t.all().tolist()
    assert x.all(1).tolist() == t.all(1).tolist()
    assert x.all(1, keepdim=True).tolist() == t.all(1, keepdim=True).tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_any(device: str) -> None:
    x = Tensor.bernoulli(2, 3, 4, p=0.5, device=device)
    t = totorch(x)
    assert x.any().tolist() == t.any().tolist()
    assert x.any(1).tolist() == t.any(1).tolist()
    assert x.any(1, keepdim=True).tolist() == t.any(1, keepdim=True).tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_topk(device: str) -> None:
    x = Tensor([[3.0, 1.0, 4.0, 1.5], [9.0, 2.0, 6.0, 5.0]], device=device)
    t = totorch(x)
    values, indices = x.topk(2, dim=1)
    expected_values, expected_indices = torch.topk(t, 2, dim=1)
    assert_close_mag_torch(values, expected_values, dtype.float32)
    assert indices.tolist() == expected_indices.tolist()
    values, indices = x.topk(3, dim=1, largest=False)
    expected_values, expected_indices = torch.topk(t, 3, dim=1, largest=False)
    assert_close_mag_torch(values, expected_values, dtype.float32)
    assert indices.tolist() == expected_indices.tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_where(device: str) -> None:
    cond = Tensor([[True, False], [False, True]], device=device)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    y = Tensor([[-1.0, -2.0], [-3.0, -4.0]], device=device)
    tc, tx, ty = totorch(cond), totorch(x), totorch(y)
    assert_close_mag_torch(Tensor.where(cond, x, y), torch.where(tc, tx, ty), dtype.float32)
    assert_close_mag_torch(Tensor.where(cond, x, 0.0), torch.where(tc, tx, torch.tensor(0.0)), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_clamp(device: str) -> None:
    skip_if_op_missing_on_cuda(device, 'CLAMP')
    x = Tensor(_UNARY_GENERAL, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.clamp(-1.0, 1.0), torch.clamp(t, -1.0, 1.0), dtype.float32)
    lo = Tensor([-1.0] * len(_UNARY_GENERAL), device=device)
    hi = Tensor([2.0] * len(_UNARY_GENERAL), device=device)
    assert_close_mag_torch(x.clamp(lo, hi), torch.clamp(t, totorch(lo), totorch(hi)), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_clamp_min(device: str) -> None:
    skip_if_op_missing_on_cuda(device, 'MAX')
    x = Tensor(_UNARY_GENERAL, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.clamp_min(-1.0), torch.clamp_min(t, -1.0), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_clamp_max(device: str) -> None:
    skip_if_op_missing_on_cuda(device, 'MIN')
    x = Tensor(_UNARY_GENERAL, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.clamp_max(1.0), torch.clamp_max(t, 1.0), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_tril(device: str) -> None:
    x = Tensor.uniform(4, 4, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.tril(), torch.tril(t), dtype.float32)
    assert_close_mag_torch(x.tril(1), torch.tril(t, 1), dtype.float32)
    assert_close_mag_torch(x.tril(-1), torch.tril(t, -1), dtype.float32)
    y = x.clone()
    y.tril_()
    assert_close_mag_torch(y, torch.tril(t), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_triu(device: str) -> None:
    x = Tensor.uniform(4, 4, device=device)
    t = totorch(x)
    assert_close_mag_torch(x.triu(), torch.triu(t), dtype.float32)
    assert_close_mag_torch(x.triu(1), torch.triu(t, 1), dtype.float32)
    assert_close_mag_torch(x.triu(-1), torch.triu(t, -1), dtype.float32)
    y = x.clone()
    y.triu_()
    assert_close_mag_torch(y, torch.triu(t), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_multinomial(device: str) -> None:
    p = Tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], device=device)
    idx = p.multinomial(num_samples=1)
    assert idx.dtype == dtype.int64
    assert idx.shape == (2, 1)
    assert idx.tolist() == torch.multinomial(totorch(p), 1).tolist()
    w = Tensor([0.25, 0.25, 0.25, 0.25], device=device)
    samples = w.multinomial(num_samples=4).tolist()
    assert sorted(samples) == sorted(torch.multinomial(totorch(w), 4).tolist())


@pytest.mark.xfail(reason='multinomial ignores replacement=True and always samples without replacement', strict=True)
@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_multinomial_with_replacement(device: str) -> None:
    p = Tensor([0.0, 1.0, 0.0, 0.0], device=device)
    assert p.multinomial(num_samples=4, replacement=True).tolist() == torch.multinomial(totorch(p), 4, replacement=True).tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_embedding(device: str) -> None:
    w = Tensor.uniform(6, 3, device=device)
    idx = Tensor([[0, 2], [5, 1]], device=device)
    expected = torch.nn.functional.embedding(totorch(idx), totorch(w))
    assert_close_mag_torch(w.embedding(idx), expected, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_cat(device: str) -> None:
    x = Tensor.uniform(2, 3, device=device)
    y = Tensor.uniform(4, 3, device=device)
    assert_close_mag_torch(Tensor.cat([x, y], 0), torch.cat([totorch(x), totorch(y)], 0), dtype.float32)
    z = Tensor.uniform(2, 5, device=device)
    assert_close_mag_torch(Tensor.cat([x, z], 1), torch.cat([totorch(x), totorch(z)], 1), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_stack(device: str) -> None:
    x = Tensor.uniform(2, 3, device=device)
    y = Tensor.uniform(2, 3, device=device)
    assert_close_mag_torch(Tensor.stack([x, y], 0), torch.stack([totorch(x), totorch(y)], 0), dtype.float32)
    assert_close_mag_torch(Tensor.stack([x, y], 2), torch.stack([totorch(x), totorch(y)], 2), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_hstack(device: str) -> None:
    x = Tensor.uniform(2, 3, device=device)
    y = Tensor.uniform(2, 4, device=device)
    assert_close_mag_torch(Tensor.hstack([x, y]), torch.hstack([totorch(x), totorch(y)]), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_vstack(device: str) -> None:
    x = Tensor.uniform(2, 3, device=device)
    y = Tensor.uniform(4, 3, device=device)
    assert_close_mag_torch(Tensor.vstack([x, y]), torch.vstack([totorch(x), totorch(y)]), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_dstack(device: str) -> None:
    x = Tensor.uniform(2, 3, device=device)
    y = Tensor.uniform(2, 3, device=device)
    assert_close_mag_torch(Tensor.dstack([x, y]), torch.dstack([totorch(x), totorch(y)]), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_einsum(device: str) -> None:
    a = Tensor.uniform(2, 3, device=device)
    b = Tensor.uniform(3, 4, device=device)
    ta, tb = totorch(a), totorch(b)
    assert_close_mag_torch(Tensor.einsum('ij,jk->ik', a, b), torch.einsum('ij,jk->ik', ta, tb), dtype.float32)
    assert_close_mag_torch(Tensor.einsum('ij->ji', a), torch.einsum('ij->ji', ta), dtype.float32)
    assert_close_mag_torch(Tensor.einsum('ij->', a), torch.einsum('ij->', ta), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_detach(device: str) -> None:
    x = Tensor.uniform(2, 3, device=device, requires_grad=True)
    assert x.requires_grad
    y = x.detach()
    assert not y.requires_grad
    assert_close_mag_torch(y, totorch(x), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_reinterpret_view(device: str) -> None:
    x = Tensor([1.0, -2.0, 3.5, 0.0], device=device)
    y = x.reinterpret_view(dtype.int32, 4)
    assert y.dtype == dtype.int32
    assert y.tolist() == totorch(x).view(torch.int32).tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_scatter_outplace(device: str) -> None:
    base = Tensor.zeros(3, 5, dtype=dtype.float32, device=device)
    src = Tensor.arange(1.0, 11.0, device=device).reshape(2, 5)
    idx = Tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], device=device)
    expected = totorch(base).scatter(0, totorch(idx), totorch(src))
    assert_close_mag_torch(base.scatter(0, idx, src), expected, dtype.float32)
    assert (base == 0.0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_scatter_add_outplace(device: str) -> None:
    base = Tensor.zeros(3, 5, dtype=dtype.float32, device=device)
    src = Tensor.arange(1.0, 11.0, device=device).reshape(2, 5)
    idx = Tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], device=device)
    expected = totorch(base).scatter_add(0, totorch(idx), totorch(src))
    assert_close_mag_torch(base.scatter_add(0, idx, src), expected, dtype.float32)
    assert (base == 0.0).all()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_lerp_inplace(device: str) -> None:
    start = Tensor.arange(1.0, 5.0, device=device)
    end = Tensor.full(4, fill_value=10.0, device=device)
    expected = torch.lerp(totorch(start), totorch(end), 0.5)
    x = start.clone()
    x.lerp_(end, 0.5)
    assert_close_mag_torch(x, expected, dtype.float32)
