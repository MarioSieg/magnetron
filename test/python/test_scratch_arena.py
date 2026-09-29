# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

import threading

import pytest
import torch

from magnetron import dtype, Tensor
from .common import assert_close_mag_torch, totorch


def _distinct(n: int) -> list[float]:
    return (torch.randperm(n).to(torch.float32) / n * 2 - 1).tolist()


def _run_on_fresh_thread(fn) -> None:
    result: list = []

    def runner() -> None:
        try:
            fn()
            result.append(None)
        except BaseException as e:
            result.append(e)

    th = threading.Thread(target=runner)
    th.start()
    th.join()
    assert result, 'thread did not finish'
    if result[0] is not None:
        raise result[0]


@pytest.mark.parametrize('n', [1024, 1025, 2048, 4096, 8192, 100000])
def test_sort_grows_scratch_between_allocations(n: int) -> None:
    def body() -> None:
        x = Tensor(_distinct(n), dtype=dtype.float32)
        t = totorch(x)
        values, indices = x.sort(dim=0)
        ev, ei = torch.sort(t, dim=0, stable=True)
        assert_close_mag_torch(values, ev, dtype.float32)
        assert indices.tolist() == ei.tolist()
        y = Tensor.empty(64, dtype=dtype.float32)
        y.fill_(1.0)
        assert y.tolist() == [1.0] * 64

    _run_on_fresh_thread(body)


@pytest.mark.parametrize('n,k', [(1024, 1024), (4096, 4096), (100000, 1000)])
def test_topk_grows_scratch_between_allocations(n: int, k: int) -> None:
    def body() -> None:
        x = Tensor(_distinct(n), dtype=dtype.float32)
        t = totorch(x)
        values, indices = x.topk(k, dim=0)
        ev, ei = torch.topk(t, k, dim=0)
        assert_close_mag_torch(values, ev, dtype.float32)
        assert indices.tolist() == ei.tolist()

    _run_on_fresh_thread(body)
