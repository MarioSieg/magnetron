"""Fusion must not change any answer, only when the work happens."""

from __future__ import annotations

import numpy as np
import pytest

import magnetron as mag
from magnetron import Tensor, no_grad

# Below magnetron's minimum chain size nothing is captured, so a test on a small tensor
# would quietly be testing the eager path instead.
N = 1 << 15


def _rand(n: int = N) -> Tensor:
    return Tensor.uniform((n,), low=0.25, high=1.75)


CHAINS = {
    'mul_add': lambda x, w, b: x * w + b,
    'deep_mul_add': lambda x, w, b: ((x * w + b) * w + b) * w + b,
    'shared_operand': lambda x, w, b: (x * x) + (x * w),
    'sub_neg': lambda x, w, b: -(x - w) + b,
    'relu_chain': lambda x, w, b: ((x - b).relu() * w).relu(),
    'abs_sqrt': lambda x, w, b: (x * w).abs().sqrt() + b,
    'min_max': lambda x, w, b: x.max(w).min(b) * x,
    'reuses_result_twice': lambda x, w, b: (x * w) * (x * w) + b,
}


@pytest.mark.parametrize('name', list(CHAINS))
def test_fusing_does_not_change_the_value(name: str) -> None:
    fn = CHAINS[name]
    x, w, b = _rand(), _rand(), _rand()
    with no_grad():
        expect = fn(x, w, b).numpy().copy()
    with no_grad(), mag.fuse():
        got = fn(x, w, b)
    # Bit-identical, not merely close. A chain that rounds differently from the operators it
    # replaces is one nobody can safely turn on.
    assert np.array_equal(got.numpy(), expect), name


@pytest.mark.parametrize('name', list(CHAINS))
def test_fusing_does_not_change_the_gradient(name: str) -> None:
    fn = CHAINS[name]

    def run(fused: bool) -> tuple[np.ndarray, np.ndarray]:
        x, w, b = _rand(), _rand(), _rand()
        x.requires_grad = True
        w.requires_grad = True
        if fused:
            with mag.fuse():
                out = fn(x, w, b)
        else:
            out = fn(x, w, b)
        out.sum().backward()
        return x.grad.numpy().copy(), w.grad.numpy().copy()

    # The same inputs are drawn by both arms only because uniform() is seeded per tensor, so
    # compare shapes and values through a single seeded context instead.
    mag.context.manual_seed(1234)
    eager = run(False)
    mag.context.manual_seed(1234)
    fused = run(True)
    assert np.array_equal(fused[0], eager[0]), f'{name}: grad wrt x'
    assert np.array_equal(fused[1], eager[1]), f'{name}: grad wrt w'


def test_reading_a_value_inside_a_region_runs_the_chain_first() -> None:
    x, w, b = _rand(), _rand(), _rand()
    with no_grad():
        expect = (x * w + b).numpy().copy()
    with no_grad(), mag.fuse():
        mid = x * w
        # Asking for the numbers mid-region has to settle what the chain still owes.
        seen = mid.numpy().copy()
        out = mid + b
    assert np.array_equal(seen, (x * w).numpy())
    assert np.array_equal(out.numpy(), expect)


def test_an_operator_that_cannot_join_splits_the_chain() -> None:
    x, w, b = _rand(), _rand(), _rand()
    with no_grad():
        expect = ((x * w).tanh() + b).numpy().copy()
    before = mag.fusion_stats()['chains']
    with no_grad(), mag.fuse():
        out = (x * w).tanh() + b
    # tanh is not exactly defined, so it ends one chain and the add starts another.
    assert mag.fusion_stats()['chains'] - before == 2
    assert np.array_equal(out.numpy(), expect)


def test_a_chain_nobody_keeps_runs_no_kernel() -> None:
    x, w, b = _rand(), _rand(), _rand()
    before = mag.fusion_stats()['chains']
    with no_grad(), mag.fuse():
        x * w + b  # noqa: B018 - the result is deliberately discarded
    # Nothing outside the chain can observe any of it, so there is nothing worth computing.
    assert mag.fusion_stats()['chains'] == before


def test_nesting_only_runs_the_chain_on_the_outermost_exit() -> None:
    x, w, b = _rand(), _rand(), _rand()
    with no_grad():
        expect = ((x * w) + b).numpy().copy()
    before = mag.fusion_stats()['chains']
    with no_grad(), mag.fuse():
        with mag.fuse():
            mid = x * w
        out = mid + b
    assert mag.fusion_stats()['chains'] - before == 1
    assert np.array_equal(out.numpy(), expect)


def test_a_broadcast_scalar_operand_still_fuses() -> None:
    x, w = _rand(), _rand()
    with no_grad():
        expect = (x * w + 2.5).numpy().copy()
    with no_grad(), mag.fuse():
        out = x * w + 2.5
    assert np.array_equal(out.numpy(), expect)


def test_a_read_only_operand_can_be_read_by_a_chain() -> None:
    """A chain must ask for write access only to what it writes.

    A tensor can borrow memory it is not allowed to write - Tensor(arr, copy=False,
    is_writeable=False) makes one - and asking for a mutable pointer to it aborts rather than
    returning an error. Reading one as a chain operand is perfectly legitimate.
    """
    rng = np.random.default_rng(0)
    a = rng.uniform(0.25, 1.75, N).astype(np.float32)
    b = rng.uniform(0.25, 1.75, N).astype(np.float32)
    read_only = Tensor(a, copy=False, is_writeable=False)
    writable = Tensor(b, copy=False)
    with no_grad():
        expect = (read_only * writable + writable).numpy().copy()
    with no_grad(), mag.fuse():
        out = read_only * writable + writable
    assert np.array_equal(out.numpy(), expect)
