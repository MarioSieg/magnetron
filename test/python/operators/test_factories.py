# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import math

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
            expected = round_f64_to_dtype(torch.arange(start, end, step, dtype=torch.float64), totorch_dtype(dt))
            torch.testing.assert_close(totorch(x), expected, rtol=0, atol=0)

    for _ in range(1000):
        test()


_STAT_N = (1000, 1000)
_SIGMAS = 6.0


def _stats(x: Tensor) -> torch.Tensor:
    return totorch(x).to(torch.float64)


def _assert_mean_std(t: torch.Tensor, mean: float, std: float) -> None:
    n = t.numel()
    assert abs(t.mean().item() - mean) <= _SIGMAS * std / math.sqrt(n)
    assert abs(t.std().item() - std) <= _SIGMAS * std / math.sqrt(2 * n)


def _assert_uniform(t: torch.Tensor, low: float, high: float) -> None:
    assert t.min().item() >= low
    assert t.max().item() <= high
    _assert_mean_std(t, (low + high) / 2.0, (high - low) / math.sqrt(12.0))


@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16, dtype.bfloat16), ids=['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize('low, high', [(-1.0, 1.0), (0.0, 1.0), (2.5, 7.5), (-10.0, -3.0)])
def test_uniform_distribution(dt: dtype.DType, low: float, high: float) -> None:
    _assert_uniform(_stats(Tensor.uniform(_STAT_N, low=low, high=high, dtype=dt)), low, high)
    x = Tensor.zeros(_STAT_N, dtype=dt)
    x.uniform_(low, high)
    _assert_uniform(_stats(x), low, high)


@pytest.mark.parametrize('mean, std', [(0.0, 1.0), (0.3, 0.6), (-2.0, 3.0)])
def test_normal_distribution(mean: float, std: float) -> None:
    _assert_mean_std(_stats(Tensor.normal(_STAT_N, mean=mean, std=std)), mean, std)
    x = Tensor.zeros(_STAT_N)
    x.normal_(mean, std)
    _assert_mean_std(_stats(x), mean, std)


@pytest.mark.parametrize('p', [0.1, 0.5, 0.9])
def test_bernoulli_distribution(p: float) -> None:
    n = _STAT_N[0] * _STAT_N[1]
    se = math.sqrt(p * (1.0 - p) / n)
    x = Tensor.bernoulli(_STAT_N, p=p)
    assert x.dtype == dtype.boolean
    assert abs(_stats(x).mean().item() - p) <= _SIGMAS * se
    y = Tensor.zeros(_STAT_N, dtype=dtype.boolean)
    y.bernoulli_(p=p)
    assert abs(_stats(y).mean().item() - p) <= _SIGMAS * se


def test_rand_perm_is_uniform_over_positions() -> None:
    n, trials = 8, 4000
    counts = torch.zeros(n, n, dtype=torch.float64)
    for _ in range(trials):
        perm = Tensor.rand_perm(n).tolist()
        assert sorted(perm) == list(range(n))
        for pos, v in enumerate(perm):
            counts[pos, v] += 1
    expected = trials / n
    se = math.sqrt(trials * (1.0 / n) * (1.0 - 1.0 / n))
    assert (counts - expected).abs().max().item() <= _SIGMAS * se
