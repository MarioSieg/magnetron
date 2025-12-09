# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch.nn.functional

from ..common import *

@pytest.mark.parametrize('dtype', ALL_DTYPES)
def test_factory_full(dtype: DataType) -> None:
    # We only test full here because Tensor.full_like, Tensor.ones etc. are just wrappers around Tensor.full
    def test(shape: tuple[int, ...]) -> None:
        fill_value = random.randint(-100, 100) if dtype.is_integer else random.uniform(-100.0, 100.0)
        x = Tensor.full(shape, fill_value=fill_value, dtype=dtype)
        y = torch.full(shape, fill_value=fill_value, dtype=totorch_dtype(dtype))
        torch.testing.assert_close(totorch(x), totorch(y))

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', NUMERIC_DTYPES)
def test_factory_arange(dtype: DataType) -> None:
    # We test against numpy here because torch does not support arange for unsigned integers (uint8, uint16, uint32, uint64)
    def test() -> None:
        start = random.randint(-100, 0) if dtype.is_integer else random.uniform(-100.0, 0.0)
        if dtype.is_unsigned_integer:
            start = abs(start)

        end = random.randint(1, 100) if dtype.is_integer else random.uniform(1.0, 100.0)
        step = random.randint(1, 10) if dtype.is_integer else random.uniform(1.0, 10.0)

        x = Tensor.arange(start, end, step, dtype=dtype)
        y = np.arange(start, end, step, dtype=tonumpy_dtype(dtype))
        np.testing.assert_allclose(tonumpy(x), tonumpy(y))

    for _ in range(1000):
        test()
