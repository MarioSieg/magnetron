"""Fusion must not change any answer, only when the work happens."""

from __future__ import annotations

import os

import numpy as np
import pytest

import magnetron as mag
from magnetron import Tensor, dtype, no_grad

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


def test_compiled_and_interpreted_chains_agree() -> None:
    """The CPU backend has two lowerings, and they must not disagree by a single bit.

    Compiling needs a host toolchain and can always fail - no compiler, an unwritable cache
    directory - in which case the chain is interpreted instead. That fallback is only safe if the
    two produce identical numbers, so this runs the same chain both ways in separate processes and
    compares the raw bytes.
    """
    import subprocess
    import sys

    prog = """
import numpy as np, magnetron as mag
from magnetron import Tensor, dtype, no_grad
mag.context.manual_seed(20260918)
n = 1 << 16
x, w, b = (Tensor.uniform((n,), low=0.25, high=1.75) for _ in range(3))
with no_grad(), mag.fuse():
    y = x
    for _ in range(6):
        y = ((y * w + b).abs().sqrt() * w - b).relu()
import sys
sys.stdout.buffer.write(y.numpy().tobytes())
"""
    def run(compile_enabled: bool) -> bytes:
        env = {**os.environ}
        if not compile_enabled:
            env['MAG_FUSE_COMPILE'] = 'off'
        out = subprocess.run([sys.executable, '-c', prog], capture_output=True, env=env, check=True)
        return out.stdout

    assert run(True) == run(False)


def test_a_chain_runs_the_same_with_compilation_disabled() -> None:
    """Turning compilation off must change speed and nothing else."""
    import subprocess
    import sys

    prog = """
import numpy as np, magnetron as mag
from magnetron import Tensor, dtype, no_grad
mag.context.manual_seed(99)
n = 1 << 16
x, w, b = (Tensor.uniform((n,)) for _ in range(3))
with no_grad():
    eager = (x * w + b).numpy().copy()
with no_grad(), mag.fuse():
    fused = x * w + b
assert np.array_equal(fused.numpy(), eager)
print('ok')
"""
    env = {**os.environ, 'MAG_FUSE_COMPILE': 'off'}
    out = subprocess.run([sys.executable, '-c', prog], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == 'ok'

# Chains are fused per dtype: the arithmetic runs in float either way, and narrow storage only
# changes where loads widen and results round. bfloat16 is absent on purpose - see below.
FUSED_DTYPES = [dtype.float32, dtype.float16]


@pytest.mark.parametrize('dt', FUSED_DTYPES, ids=lambda d: str(d).rsplit('.', 1)[-1])
@pytest.mark.parametrize('name', list(CHAINS))
def test_narrow_storage_rounds_where_eager_rounds(name: str, dt) -> None:
    """A chain must round at exactly the points the operators it replaces round at.

    Eager execution materializes a tensor after every operator, so each intermediate is narrowed on
    its way out and widened on its way back in. A chain that carried full float precision straight
    through would be *more* accurate and would disagree, which is the one thing it may not do.
    """
    fn = CHAINS[name]
    x, w, b = (_rand().cast(dt) for _ in range(3))
    with no_grad():
        expect = fn(x, w, b).cast(dtype.float32).numpy().copy()
    with no_grad(), mag.fuse():
        got = fn(x, w, b)
    assert np.array_equal(got.cast(dtype.float32).numpy(), expect), f'{name} {dt}'


def test_bfloat16_is_left_to_the_eager_kernels() -> None:
    """bfloat16 chains are declined, and the reason is a bug in the eager path rather than a gap here.

    This backend's eager bfloat16 store truncates on the vector path and rounds to nearest on the
    scalar tail, so the same multiply gives two different answers depending on how long the tensor
    is. Until that is one answer there is nothing definite for a chain to reproduce, and choosing
    either of the two would bake the inconsistency in.
    """
    short = Tensor([0.9] * 3, dtype=dtype.float32).cast(dtype.bfloat16)
    short_b = Tensor([0.012] * 3, dtype=dtype.float32).cast(dtype.bfloat16)
    long = Tensor([0.9] * 64, dtype=dtype.float32).cast(dtype.bfloat16)
    long_b = Tensor([0.012] * 64, dtype=dtype.float32).cast(dtype.bfloat16)
    with no_grad():
        a = float((short * short_b).cast(dtype.float32).numpy()[0])
        c = float((long * long_b).cast(dtype.float32).numpy()[0])
    # If this ever starts passing, the eager store has been fixed and bfloat16 chains can be enabled.
    assert a != c, 'eager bfloat16 is now length-independent; enable bfloat16 fusion'

    # Meanwhile a bfloat16 chain still produces the eager answer, by falling back to it.
    x, w, b = (_rand().cast(dtype.bfloat16) for _ in range(3))
    before = mag.fusion_stats()['chains']
    with no_grad():
        expect = (x * w + b).cast(dtype.float32).numpy().copy()
    with no_grad(), mag.fuse():
        out = x * w + b
    assert np.array_equal(out.cast(dtype.float32).numpy(), expect)
    assert mag.fusion_stats()['chains'] == before, 'bfloat16 should not have been lowered'



# Chains whose backwards never read their operands' values, so their intermediates can be dropped
# even while gradients are recording. add, sub and neg are the only operators that qualify.
ELIDING_CHAINS = {
    'adds': lambda x, w, b: (((x + w) + b) + w) + b,
    'subs': lambda x, w, b: (((x - w) - b) - w) - b,
    'negs': lambda x, w, b: -(-(x + w) - b),
}


@pytest.mark.parametrize('name', list(ELIDING_CHAINS))
def test_intermediates_are_dropped_while_gradients_record(name: str) -> None:
    """A chain of value-ignoring operators should write one result, not one per link.

    Recording gradients gives every consumer's autodiff state a reference to its operands, so
    without knowing which backwards actually read those operands nothing could ever be dropped.
    """
    fn = ELIDING_CHAINS[name]
    x, w, b = _rand(), _rand(), _rand()
    x.requires_grad = True
    before = mag.fusion_stats()
    with mag.fuse():
        out = fn(x, w, b)
    after = mag.fusion_stats()
    ops = after['ops_fused'] - before['ops_fused']
    elided = after['elided'] - before['elided']
    assert ops >= 4, f'{name}: expected a chain, got {ops} operators'
    # Everything but the result the caller kept.
    assert elided == ops - 1, f'{name}: dropped {elided} of {ops} results'
    out.sum().backward()
    assert np.all(np.isfinite(x.grad.numpy()))


def test_a_backward_that_reads_its_operand_keeps_the_value() -> None:
    """mul needs both factors to differentiate, so none of its intermediates may be dropped."""
    x, w, b = _rand(), _rand(), _rand()
    x.requires_grad = True
    before = mag.fusion_stats()
    with mag.fuse():
        out = ((x * w) * b) * w
    after = mag.fusion_stats()
    assert after['ops_fused'] - before['ops_fused'] == 3
    assert after['elided'] - before['elided'] == 0
    out.sum().backward()
    assert np.all(np.isfinite(x.grad.numpy()))


@pytest.mark.parametrize('name', list(ELIDING_CHAINS))
def test_dropping_intermediates_does_not_change_the_gradient(name: str) -> None:
    """The gradients a chain produces must match the ones the operators would have produced.

    This is the check that catches a wrong entry in the ignores-value table. Verified by corrupting
    that table on purpose - declaring that mul ignores its operands - and confirming this comparison
    goes red.
    """
    fn = ELIDING_CHAINS[name]

    def grads(fused: bool) -> tuple[np.ndarray, np.ndarray]:
        mag.context.manual_seed(4242)
        x, w = _rand(), _rand()
        b = _rand()
        x.requires_grad = True
        w.requires_grad = True
        if fused:
            with mag.fuse():
                out = fn(x, w, b)
        else:
            out = fn(x, w, b)
        out.sum().backward()
        return x.grad.numpy().copy(), w.grad.numpy().copy()

    eager, fused = grads(False), grads(True)
    assert np.array_equal(fused[0], eager[0]), f'{name}: grad wrt x'
    assert np.array_equal(fused[1], eager[1]), f'{name}: grad wrt w'


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
