"""Automatic capture of pointwise chains inside a mag.fuse() region.

Fusing must not change a value, and the chain must break where it has to: at an operator that
cannot join, at a read of a value the chain has not produced yet, and at an operator whose CPU
kernel only approximates IEEE arithmetic. The counters are what pin the second half - a test that
only compared values would pass just as happily if nothing were ever captured.
"""

from contextlib import contextmanager
from typing import Callable, Iterator

import pytest

from .common import *

import magnetron as mag
from magnetron import context, no_grad


def _chain(a: Tensor, b: Tensor) -> Tensor:
    return (a * b + 0.5) * a - b


def _long(a: Tensor, b: Tensor) -> Tensor:
    t = ((a * b + a) - b) * (a * b + a)
    return t.sqrt() / a


def _tanh_midway(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).tanh() * a


def _reduction_midway(a: Tensor, b: Tensor) -> Tensor:
    t = a * b + 0.5
    return t * t.sum()


@contextmanager
def _counted() -> Iterator[list[int]]:
    """Yields [chains, ops] fused while the block runs, filled in on exit."""
    before = context.jit_stats()
    out = [0, 0]
    yield out
    after = context.jit_stats()
    out[0] = after['chains_fused'] - before['chains_fused']
    out[1] = after['ops_fused'] - before['ops_fused']


@pytest.fixture(scope='module')
def jit() -> None:
    """Skip the capture-shape tests where no kernel can be built: MAG_JIT=off, or no C compiler."""
    probe = Tensor.uniform((64,), low=0.5, high=1.5)
    with _counted() as counted:
        with mag.fuse():
            _ = probe * probe + probe
    if counted[0] == 0:
        pytest.skip('no host compiler, or MAG_JIT=off: nothing is fused here')


def _pair(n: int = 4096) -> tuple[Tensor, Tensor]:
    return Tensor.uniform((n,), low=0.5, high=1.5), Tensor.uniform((n,), low=0.5, high=1.5)


BUILDS = [
    ('chain', _chain),
    ('long_chain', _long),
    ('scalars', lambda a, b: (a * 2.5 + 0.25) - b * 0.5),
    ('approximated_op', _tanh_midway),
    ('reduction_midway', _reduction_midway),
]


@pytest.mark.parametrize('name,build', BUILDS, ids=[b[0] for b in BUILDS])
@pytest.mark.parametrize('grad', [True, False], ids=['grad', 'no_grad'])
def test_fusing_does_not_change_the_value(name: str, build: Callable, grad: bool) -> None:
    a, b = _pair()
    if grad:
        eager = build(a, b).numpy().copy()
        with mag.fuse():
            fused = build(a, b)
    else:
        with no_grad():
            eager = build(a, b).numpy().copy()
            with mag.fuse():
                fused = build(a, b)
    assert np.array_equal(eager, fused.numpy())


# name, build, expected (chains, ops). tanh is approximated in the vector kernels, so generated
# libm code could not reproduce it and it has to split the chain; a reduction reads the whole
# tensor, so it cannot join one that is still pending.
SHAPES = [
    ('one_kernel_for_the_whole_chain', _chain, (1, 4)),
    ('approximated_op_splits', _tanh_midway, (2, 2)),
    ('reduction_splits', _reduction_midway, (2, 3)),
]


@pytest.mark.parametrize('name,build,expected', SHAPES, ids=[s[0] for s in SHAPES])
def test_where_the_chain_breaks(jit: None, name: str, build: Callable, expected: tuple[int, int]) -> None:
    a, b = _pair(512)
    with _counted() as counted:
        with mag.fuse():
            out = build(a, b)
        out.numpy()
    assert tuple(counted) == expected


def test_reading_a_value_mid_region_flushes() -> None:
    """Touching a pending value inside the region must materialize it, not read stale memory."""
    a, b = _pair(1024)
    eager_mid = float((a * b).numpy()[0])
    eager_end = ((a * b) + a).numpy().copy()
    with mag.fuse():
        t = a * b
        mid = float(t.numpy()[0])  # read before the region closes
        out = t + a
    assert mid == eager_mid
    assert np.array_equal(eager_end, out.numpy())


def test_only_the_outermost_exit_flushes(jit: None) -> None:
    """Leaving an inner region must not flush, or nesting would break chains that could continue."""
    a, b = _pair(1024)
    eager = (a * b + a).numpy().copy()
    with _counted() as counted:
        with mag.fuse():
            with mag.fuse():
                t = a * b
            out = t + a  # still inside the outer region, so the chain continues
        out.numpy()
    assert tuple(counted) == (1, 2)
    assert np.array_equal(eager, out.numpy())


def test_dead_chain_runs_no_kernel(jit: None) -> None:
    """With no graph holding them, intermediates nothing refers to never have to reach memory."""
    a, _ = _pair()
    with _counted() as counted:
        with no_grad(), mag.fuse():
            a * a + a  # not bound to anything, so the whole chain is dead  # noqa: B018
    assert counted[0] == 0
