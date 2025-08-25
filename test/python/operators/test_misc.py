# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>
import torch

from ..common import *

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_clone(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        a = x
        b = a.clone()
        assert a.shape == b.shape
        assert a.strides == b.strides
        assert a.numel == b.numel
        assert a.rank == b.rank
        assert a.is_contiguous == b.is_contiguous
        assert a.tolist() == b.tolist()

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_gather(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        torch_x = totorch(x)
        rank = len(shape)
        for dim in range(rank):
            index_torch = torch.randint(0, shape[dim], size=shape, dtype=torch.int64)
            index_own = Tensor.of(index_torch.tolist(), dtype=int32)
            out_own = x.gather(dim, index_own)
            out_torch = torch.gather(torch_x, dim, index_torch)
            torch.testing.assert_close(totorch(out_own), out_torch)

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_split(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        if len(shape) <= 1:
            return
        dim = random.randint(0, len(shape)-1)
        split_size = random.randint(1, shape[dim])
        a = x.split(split_size, dim)
        b = totorch(x).split(split_size, dim)
        assert len(a) == len(b)
        for i in range(len(a)):
            torch.testing.assert_close(totorch(a[i]), b[i])

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_tolist(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        assert x.tolist() == totorch(x).tolist()

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_transpose(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if len(shape) <= 1: return
        if dtype == boolean: x = Tensor.bernoulli(shape)
        else: x = Tensor.uniform(shape, dtype=dtype)
        sample = lambda: random.randint(-len(shape)+1, len(shape)-1)
        dim1: int = sample()
        dim2: int = dim1
        while dim2 == dim1: # Reject equal transposition axes
            dim2 = sample()
        a = totorch(x.transpose(dim1, dim2))
        b = totorch(x).transpose(dim1, dim2)
        if not torch.allclose(a, b):
            print('M='+str(a))
            print('T='+str(b))
            print(f'axes: {dim1} {dim2}')
        torch.testing.assert_close(a, b)

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_view(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        shape = list(shape)
        random.shuffle(shape) # Shuffle view shape
        shape = tuple(shape)
        y = x.view(*shape)
        torch.testing.assert_close(totorch(y), totorch(x).view(shape))

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_view_infer_axis(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        shape = list(shape)
        random.shuffle(shape) # Shuffle view shape
        shape[random.randint(0, len(shape)-1)] = -1 # Set inferred axis randomly
        shape = tuple(shape)
        y = x.view(*shape)
        torch.testing.assert_close(totorch(y), totorch(x).view(shape))

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_reshape(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        shape = list(shape)
        random.shuffle(shape) # Shuffle reshape shape
        shape = tuple(shape)
        y = x.T.reshape(*shape)
        torch.testing.assert_close(totorch(y), totorch(y).reshape(shape))

    square_shape_permutations(func, 4)

@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_reshape_infer_axis(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        shape = list(shape)
        random.shuffle(shape) # Shuffle reshape shape
        shape[random.randint(0, len(shape)-1)] = -1 # Set inferred axis randomly
        shape = tuple(shape)
        y = x.T.reshape(*shape)
        torch.testing.assert_close(totorch(y), totorch(y).reshape(shape))

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
    assert a.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert b.tolist() == [[1, 1], [1, 1], [1, 1]]
    assert a.is_contiguous
    assert not b.is_contiguous
