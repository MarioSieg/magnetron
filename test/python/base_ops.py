# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import random
import pytest
from magnetron import *
from .common import *

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

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.boolean, mag.int32])
def test_tensor_view(dtype: mag.DataType) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def func(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype)
        shape = list(shape)
        random.shuffle(shape) # Shuffle view shape
        shape = tuple(shape)
        y = x.view(*shape)
        torch.testing.assert_close(totorch(y, torch_dt), totorch(y, torch_dt).view(shape))

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.boolean, mag.int32])
def test_tensor_view_infer_axis(dtype: mag.DataType) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def func(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype)
        shape = list(shape)
        random.shuffle(shape) # Shuffle view shape
        shape[random.randint(0, len(shape)-1)] = -1 # Set inferred axis randomly
        shape = tuple(shape)
        y = x.view(*shape)
        torch.testing.assert_close(totorch(y, torch_dt), totorch(y, torch_dt).view(shape))

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.boolean, mag.int32])
def test_tensor_reshape(dtype: mag.DataType) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def func(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype)
        shape = list(shape)
        random.shuffle(shape) # Shuffle reshape shape
        shape = tuple(shape)
        y = x.T.reshape(*shape)
        torch.testing.assert_close(totorch(y, torch_dt), totorch(y, torch_dt).reshape(shape))

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [mag.float16, mag.float32, mag.boolean, mag.int32])
def test_tensor_reshape_infer_axis(dtype: mag.DataType) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def func(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype)
        shape = list(shape)
        random.shuffle(shape) # Shuffle reshape shape
        shape[random.randint(0, len(shape)-1)] = -1 # Set inferred axis randomly
        shape = tuple(shape)
        y = x.T.reshape(*shape)
        torch.testing.assert_close(totorch(y, torch_dt), totorch(y, torch_dt).reshape(shape))

    square_shape_permutations(func, 4)

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
    b = a.permute(5, 4, 3, 2, 1, 0)
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
