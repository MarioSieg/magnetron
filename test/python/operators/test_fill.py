# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

import random
import torch

from ..common import *


def _get_random_fill_val(dtype: DataType) -> int | float | bool:
    if dtype.is_integer:
        return random.randint(-0x80000000, 0x7FFFFFFF)
    elif dtype.is_floating_point:
        return random.uniform(-1000.0, 1000.0)
    elif dtype == boolean:
        return random.choice([True, False])
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_fill(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        fill_val = _get_random_fill_val(dtype)
        x = Tensor.full(*shape, fill_value=fill_val, dtype=dtype)
        y = torch.full(shape, fill_val, dtype=totorch_dtype(dtype))
        torch.testing.assert_close(totorch(x), y)

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_copy(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        fill_val = _get_random_fill_val(dtype)
        x = Tensor.empty(*shape, dtype=dtype)
        x.copy_(Tensor.full(*shape, fill_value=fill_val, dtype=dtype))
        y = torch.full(shape, fill_val, dtype=totorch_dtype(dtype))
        torch.testing.assert_close(totorch(x), y)

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, boolean, int32])
def test_tensor_masked_fill(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        fill_val = _get_random_fill_val(dtype)
        mask = Tensor.bernoulli(shape)
        x = Tensor.full(*shape, fill_value=-1, dtype=dtype).masked_fill(mask, fill_val)
        y = torch.full(shape, fill_value=-1, dtype=totorch_dtype(dtype)).masked_fill(totorch(mask), fill_val)
        torch.testing.assert_close(totorch(x), y)

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [float16, float32, int32])
def test_tensor_arange(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        start = _get_random_fill_val(dtype)
        step = _get_random_fill_val(dtype)
        x = Tensor.arange(start, start + shape[0] * step, step=step, dtype=dtype)
        y = torch.arange(start, start + shape[0] * step, step=step, dtype=totorch_dtype(dtype))
        torch.testing.assert_close(totorch(x), y)

    for_all_shapes(func)


@pytest.mark.parametrize('dtype', [uint8, int8, int16, int32, int64])
def test_tensor_rand_perm(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x = Tensor.rand_perm(sum(shape), dtype=dtype)
        y = torch.arange(sum(shape), dtype=totorch_dtype(dtype))
        torch.testing.assert_close(totorch(x), y)

    for_all_shapes(func)
