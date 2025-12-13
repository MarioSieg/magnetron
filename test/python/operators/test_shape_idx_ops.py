# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch.nn.functional

from magnetron import dtype
from ..common import *

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_view(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dtype)
        assert not x.is_view
        r = x.view(shape)
        assert r.is_view
        torch.testing.assert_close(totorch(r), totorch(x).view(shape))
        assert (r.view(shape) == x).all()

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_reshape(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        new_shape = list(shape)
        random.shuffle(new_shape)
        new_shape = tuple(new_shape)
        x = random_tensor(shape, dtype)
        r = x.reshape(new_shape)
        torch.testing.assert_close(totorch(r), totorch(x).reshape(new_shape))
        assert (r.reshape(shape) == x).all()

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_transpose(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) < 2: # transpose requires at least 2 dimensions
            return
        x = random_tensor(shape, dtype)
        dim0 = random.randint(0, len(shape)-1)
        dim1 = dim0
        while dim1 == dim0: # ensure different dimensions
            dim1 = random.randint(0, len(shape)-1)
        r = x.transpose(dim0, dim1)
        torch.testing.assert_close(totorch(r), totorch(x).transpose(dim0, dim1))
        assert (r.transpose(dim0, dim1) == x).all()

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_permute(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dtype)
        perm = random.sample(range(len(shape)), len(shape))
        r = x.permute(perm)
        torch.testing.assert_close(totorch(r), totorch(x).permute(perm))

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_contiguous(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) < 2:
            return
        x = random_tensor(shape, dtype)
        assert x.is_contiguous
        dim0 = 0
        dim1 = 1
        r = x.transpose(dim0, dim1)
        tx = totorch(x).transpose(dim0, dim1)
        assert r.is_contiguous == tx.is_contiguous()
        rc = r.contiguous()
        txc = tx.contiguous()
        assert rc.is_contiguous
        assert txc.is_contiguous()
        torch.testing.assert_close(totorch(rc), txc, equal_nan=True)

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_squeeze(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0: # squeeze on scalars is not supported
            return
        x = random_tensor(shape, dtype)
        dim = random.choice(list(range(-len(shape), len(shape))))
        r = x.squeeze(dim)
        torch.testing.assert_close(totorch(r), totorch(x).squeeze(dim))

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_unsqueeze(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0: # squeeze on scalars is not supported
            return
        x = random_tensor(shape, dtype)
        dim = random.choice(list(range(-len(shape), len(shape))))
        r = x.unsqueeze(dim)
        torch.testing.assert_close(totorch(r), totorch(x).unsqueeze(dim))
        assert (r.squeeze(dim) == x).all()

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_flatten(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0: # squeeze on scalars is not supported
            return
        x = random_tensor(shape, dtype)
        start_dim = random.choice(list(range(-len(shape), len(shape))))
        end_dim = start_dim
        while end_dim < start_dim:
            end_dim = random.choice(list(range(-len(shape), len(shape))))
        r = x.flatten(start_dim, end_dim)
        torch.testing.assert_close(totorch(r), totorch(x).flatten(start_dim, end_dim))

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_unflatten(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0:  # unflatten on scalars is weird, skip
            return
        x = random_tensor(shape, dtype)
        dim = random.choice(list(range(-len(shape), len(shape))))
        dim_norm = dim % len(shape)
        dim_size = shape[dim_norm]
        factor1 = random.randint(1, dim_size)
        while dim_size % factor1 != 0:
            factor1 = random.randint(1, dim_size)
        factor2 = dim_size // factor1
        sizes = [factor1, factor2]  # only factors for this dim
        r = x.unflatten(dim, sizes)
        tx = totorch(x).unflatten(dim, sizes)
        torch.testing.assert_close(totorch(r), tx, equal_nan=True)

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_narrow(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0:
            return
        x = random_tensor(shape, dtype)
        dim = random.choice(range(-len(shape), len(shape)))
        dim_norm = dim % len(shape)
        size = shape[dim_norm]
        if size == 0:
            return
        start = random.randint(0, size - 1)
        length = random.randint(0, size - start)
        if length == 0:
            return
        r = x.narrow(dim, start, length)
        t = totorch(x).narrow(dim, start, length)
        torch.testing.assert_close(totorch(r), t, equal_nan=True)

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_movedim(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) < 2:
            return
        x = random_tensor(shape, dtype)
        src = random.randint(-len(shape), len(shape) - 1)
        dst = random.randint(-len(shape), len(shape) - 1)
        r = x.movedim(src, dst)
        t = totorch(x).movedim(src, dst)
        torch.testing.assert_close(totorch(r), t, equal_nan=True)

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_select(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0:
            return
        x = random_tensor(shape, dtype)
        dim = random.choice(range(-len(shape), len(shape)))
        dim_norm = dim % len(shape)
        size = shape[dim_norm]
        if size == 0:
            return
        index = random.randint(0, size - 1)
        r = x.select(dim, index)
        t = totorch(x).select(dim, index)
        torch.testing.assert_close(totorch(r), t, equal_nan=True)
    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_split_and_cat_roundtrip(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0:
            return
        x = random_tensor(shape, dtype)
        rank = len(shape)
        dim = random.randint(-rank, rank - 1)
        dim_norm = dim if dim >= 0 else dim + rank
        dim_size = shape[dim_norm]
        if dim_size == 0:
            return
        split_size = random.randint(1, dim_size)
        parts_mag = x.split(split_size, dim=dim)
        y_mag = Tensor.cat(parts_mag, dim=dim)
        tx = totorch(x)
        parts_torch = tx.split(split_size, dim=dim)
        y_torch = torch.cat(parts_torch, dim=dim)
        assert len(parts_mag) == len(parts_torch)
        for pm, pt in zip(parts_mag, parts_torch):
            torch.testing.assert_close(totorch(pm), pt, equal_nan=True)

        torch.testing.assert_close(totorch(y_mag), y_torch, equal_nan=True)

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_cat(dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0:
            return
        rank = len(shape)
        dim = random.randint(-rank, rank - 1)
        dim_norm = dim if dim >= 0 else dim + rank
        base = list(shape)
        if base[dim_norm] == 0:
            return
        total = base[dim_norm]
        first = random.randint(1, total)
        second = total - first
        if second == 0:
            if total < 2:
                return
            first = total // 2
            second = total - first
        sizes = [first, second]
        xs_mag: list[Tensor] = []
        xs_torch: list[torch.Tensor] = []
        for s in sizes:
            new_shape = list(base)
            new_shape[dim_norm] = s
            x = random_tensor(tuple(new_shape), dtype)
            xs_mag.append(x)
            xs_torch.append(totorch(x))
        y_mag = Tensor.cat(xs_mag, dim=dim)
        y_torch = torch.cat(xs_torch, dim=dim)
        torch.testing.assert_close(totorch(y_mag), y_torch, equal_nan=True)

    for_all_shapes(test)

# TODO: Gather
