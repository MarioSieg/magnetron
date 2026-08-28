# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch.nn.functional

from ..common import *


@pytest.mark.parametrize('dt', dtype.all)
def test_factory_full(dt: dtype.DType) -> None:
    # We only test full here because Tensor.full_like, Tensor.ones etc. are just wrappers around Tensor.full
    def test(shape: tuple[int, ...]) -> None:
        if dt == dtype.boolean:
            fill_value = random.randint(0, 1)
        elif dt.is_integer():
            fill_value = random.randint(-100, 100)
        else:
            fill_value = random.uniform(-100.0, 100.0)
        if dt.is_unsigned_integer():
            fill_value = abs(fill_value)
        x = Tensor.full(shape, fill_value=fill_value, dtype=dt)
        if dt == dtype.boolean:
            y = torch.full(shape, bool(fill_value), dtype=torch.bool)
        else:
            y = torch.full(shape, fill_value=fill_value, dtype=totorch_dtype(dt))
        torch.testing.assert_close(totorch(x), y)

    for_all_shapes(test)


@pytest.mark.parametrize('dt', tuple(d for d in dtype.numeric if d != dtype.float8_e4m3fn))
def test_factory_arange(dt: dtype.DType) -> None:
    # We test against numpy here because torch does not support arange for unsigned integers (uint8, uint16, uint32, uint64)
    rtol, atol = compare_tol(dt) if dt.is_floating_point() else (1e-7, 0)

    def test() -> None:
        if dt.is_integer():
            if dt.is_unsigned_integer():
                start = random.randint(0, 5)
                end = start + random.randint(1, 20)
            else:
                start = random.randint(-10, 0)
                end = random.randint(1, 20)
            step = random.randint(1, 5)
        else:
            start = random.uniform(-10.0, 0.0)
            end = random.uniform(1.0, 10.0)
            step = random.uniform(0.25, 2.0)
            if end <= start:
                end = start + abs(step) + 1.0

        x = Tensor.arange(start, end, step, dtype=dt)
        if dt.is_integer():
            expected = np.arange(start, end, step, dtype=tonumpy_dtype(dt))
            np.testing.assert_allclose(tonumpy(x), expected, rtol=rtol, atol=atol)
        else:
            expected = torch.arange(start, end, step, dtype=torch.float64).to(totorch_dtype(dt))
            torch.testing.assert_close(totorch(x), expected, rtol=0, atol=0)

    for _ in range(1000):
        test()
