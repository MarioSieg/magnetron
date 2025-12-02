# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

import torch

from ..common import *


def _make_tensor(shape, dtype):
    if dtype == boolean:
        return Tensor.bernoulli(shape)
    return Tensor.uniform(shape, dtype=dtype)


def _rand_shape(max_rank=4, max_size=6):
    rank = random.randint(1, max_rank)
    return tuple(random.randint(1, max_size) for _ in range(rank))


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_clone(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x = _make_tensor(shape, dtype)
        a = x
        b = a.clone()
        assert a.shape == b.shape
        assert a.strides == b.strides
        assert a.numel == b.numel
        assert a.rank == b.rank
        assert a.is_contiguous == b.is_contiguous
        assert a.tolist() == b.tolist()

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_gather(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x = _make_tensor(shape, dtype)
        torch_x = totorch(x)
        rank = len(shape)
        for dim in range(rank):
            index_torch = torch.randint(0, shape[dim], size=shape, dtype=torch.int64)
            index_own = Tensor.of(index_torch.tolist(), dtype=int64)
            out_own = x.gather(dim, index_own)
            out_torch = torch.gather(torch_x, dim, index_torch)
            torch.testing.assert_close(totorch(out_own), out_torch)

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_cat_random(dtype):
    base_shape = _rand_shape()
    rank = len(base_shape)
    dim = random.randint(0, rank - 1)
    k = random.randint(2, 5)
    others = list(base_shape)
    sizes = [random.randint(2, 5) for _ in range(k)]
    shapes = [tuple(others[:dim] + [s] + others[dim + 1 :]) for s in sizes]
    xs = [_make_tensor(s, dtype) for s in shapes]
    ts = [totorch(x) for x in xs]
    y = Tensor.cat(xs, dim=dim)
    yt = torch.cat(ts, dim=dim)
    torch.testing.assert_close(totorch(y), yt)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_cat_negative_dim(dtype):
    base_shape = _rand_shape()
    dim = -1
    k = random.randint(2, 4)
    others = list(base_shape)
    sizes = [random.randint(1, 4) for _ in range(k)]
    shapes = [tuple(others[:-1] + [s]) for s in sizes]
    xs = [_make_tensor(s, dtype) for s in shapes]
    ts = [totorch(x) for x in xs]

    y = Tensor.cat(xs, dim=dim)
    yt = torch.cat(ts, dim=dim)

    torch.testing.assert_close(totorch(y), yt)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_cat_single_tensor_noop(dtype):
    shape = _rand_shape()
    x = _make_tensor(shape, dtype)
    y = Tensor.cat([x], dim=0)
    torch.testing.assert_close(totorch(y), totorch(x))


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_cat_shape_mismatch_raises(dtype):
    rank = random.randint(2, 4)
    shape_a = [random.randint(2, 5) for _ in range(rank)]
    shape_b = shape_a[:]
    dim = random.randint(0, rank - 1)

    other = (dim + 1) % rank
    shape_b[other] = shape_a[other] + 1

    a = _make_tensor(tuple(shape_a), dtype)
    b = _make_tensor(tuple(shape_b), dtype)

    with pytest.raises(AssertionError):
        _ = Tensor.cat([a, b], dim=dim)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_cat_matches_torch_many_chunks(dtype):
    base_shape = _rand_shape()
    rank = len(base_shape)
    dim = random.randint(0, rank - 1)

    others = list(base_shape)
    k = 10
    sizes = [random.randint(2, 5) for _ in range(k)]
    shapes = [tuple(others[:dim] + [s] + others[dim + 1 :]) for s in sizes]

    xs = [_make_tensor(s, dtype) for s in shapes]
    ts = [totorch(x) for x in xs]

    y = Tensor.cat(xs, dim=dim)
    yt = torch.cat(ts, dim=dim)
    torch.testing.assert_close(totorch(y), yt)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_split_cat_roundtrip_random(dtype):
    for _ in range(1000):
        shape = _rand_shape()
        x = _make_tensor(shape, dtype)
        rank = len(shape)
        dim = random.randint(0, rank - 1)
        split_size = random.randint(1, shape[dim])
        parts = x.split(split_size, dim)
        assert len(parts) >= 1
        y = Tensor.cat(parts, dim=dim)
        assert x.tolist() == y.tolist()


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_split(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x = _make_tensor(shape, dtype)
        if len(shape) <= 1:
            return
        dim = random.randint(0, len(shape) - 1)
        split_size = random.randint(1, shape[dim])
        a = x.split(split_size, dim)
        b = totorch(x).split(split_size, dim)
        assert len(a) == len(b)
        for i in range(len(a)):
            torch.testing.assert_close(totorch(a[i]), b[i])

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_tolist(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x = _make_tensor(shape, dtype)
        assert x.tolist() == totorch(x).tolist()

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_transpose(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if len(shape) <= 1:
            return
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        sample = lambda: random.randint(-len(shape) + 1, len(shape) - 1)
        dim1: int = sample()
        dim2: int = dim1
        while dim2 == dim1:  # Reject equal transposition axes
            dim2 = sample()
        a = totorch(x.transpose(dim1, dim2))
        b = totorch(x).transpose(dim1, dim2)
        if not torch.allclose(a, b):
            print('M=' + str(a))
            print('T=' + str(b))
            print(f'axes: {dim1} {dim2}')
        torch.testing.assert_close(a, b)

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_view(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x = _make_tensor(shape, dtype)
        shape = list(shape)
        random.shuffle(shape)  # Shuffle view shape
        shape = tuple(shape)
        y = x.view(*shape)
        torch.testing.assert_close(totorch(y), totorch(x).view(shape))

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_view_infer_axis(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x = _make_tensor(shape, dtype)
        shape = list(shape)
        random.shuffle(shape)  # Shuffle view shape
        shape[random.randint(0, len(shape) - 1)] = -1  # Set inferred axis randomly
        shape = tuple(shape)
        y = x.view(*shape)
        torch.testing.assert_close(totorch(y), totorch(x).view(shape))

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_reshape(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x = _make_tensor(shape, dtype)
        shape = list(shape)
        random.shuffle(shape)  # Shuffle reshape shape
        shape = tuple(shape)
        y = x.T.reshape(*shape)
        torch.testing.assert_close(totorch(y), totorch(y).reshape(shape))

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_reshape_infer_axis(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x = _make_tensor(shape, dtype)
        shape = list(shape)
        random.shuffle(shape)  # Shuffle reshape shape
        shape[random.randint(0, len(shape) - 1)] = -1  # Set inferred axis randomly
        shape = tuple(shape)
        y = x.T.reshape(*shape)
        torch.testing.assert_close(totorch(y), totorch(y).reshape(shape))

    for_all_shapes(func)


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
