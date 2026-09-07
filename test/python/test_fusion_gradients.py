"""Fusing a chain must not change the gradients it produces.

Fusion elides values that nothing will read again. Deciding which those are means knowing, per
operator, whether its backward reads an operand's data or only its shape - a wrong answer there
produces gradients that are slightly off. Nothing crashes and nothing reports it; the model just
trains a little worse, which is the worst failure mode available. These tests are the release-build net for that: for every fusible operator, the
gradient with fusion on must equal the gradient with it off, bit for bit.

Debug builds carry the other half: an elided value's storage is poisoned, so a backward that reads
one it was told it could skip yields NaN immediately.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from .common import *

import magnetron as mag

# Every operator in the fusion table, as an expression that puts it inside a chain rather than
# alone, so the surrounding values are candidates for elision too.
FUSIBLE = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'neg': lambda x, y: (-x) * y,
    'abs': lambda x, y: x.abs() * y,
    'sgn': lambda x, y: x.sgn() * y,
    'sqr': lambda x, y: x.sqr() + y,
    'sqrt': lambda x, y: x.sqrt() * y,
    'min': lambda x, y: x.min(y) * y,
    'max': lambda x, y: x.max(y) * y,
    'relu': lambda x, y: x.relu() * y,
    'step': lambda x, y: x.step() * y,
    'floor': lambda x, y: x.floor() * y,
    'ceil': lambda x, y: x.ceil() * y,
    'round': lambda x, y: x.round() * y,
    'trunc': lambda x, y: x.trunc() * y,
    'clamp': lambda x, y: x.clamp(-0.5, 0.5) * y,
    # chains, where the interesting eliding happens
    'chain_mul_add': lambda x, y: (x * y + x) * y,
    'chain_long': lambda x, y: ((x * y) + x - y) * (x + y),
    'chain_sqrt': lambda x, y: (x * x + y).sqrt() * y,
}

_DATA_X = np.linspace(0.35, 2.4, 512, dtype=np.float32)
_DATA_Y = np.linspace(0.6, 1.9, 512, dtype=np.float32)


def _grad_of(t: Tensor) -> np.ndarray | None:
    """None when no gradient reaches the input, which is correct for ops with a zero derivative."""
    return None if t.grad is None else t.grad.numpy().copy()


def _same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(a, b)


def _run(build, fused: bool) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    x = Tensor(_DATA_X.copy(), requires_grad=True)
    y = Tensor(_DATA_Y.copy(), requires_grad=True)
    if fused:
        with mag.fuse():
            out = build(x, y)
    else:
        out = build(x, y)
    value = out.numpy().copy()
    out.sum().backward()
    return value, _grad_of(x), _grad_of(y)


def _grads(build, fused: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
    return _run(build, fused)[1:]


@pytest.mark.parametrize('name', sorted(FUSIBLE))
def test_fusion_changes_neither_the_value_nor_the_gradient(name: str) -> None:
    """The forward has to agree too, or the gradient comparison proves nothing."""
    build = FUSIBLE[name]
    v_eager, gx_eager, gy_eager = _run(build, fused=False)
    v_fused, gx_fused, gy_fused = _run(build, fused=True)
    assert np.array_equal(v_eager, v_fused), f'{name}: forward changed under fusion'
    assert _same(gx_eager, gx_fused), f'{name}: d/dx changed under fusion'
    assert _same(gy_eager, gy_fused), f'{name}: d/dy changed under fusion'


def test_gradients_survive_a_chain_read_midway() -> None:
    """Flushing early must not change what backward later computes."""
    def build(x, y):
        t = x * y + x
        _ = float(t.numpy()[0])  # forces the chain mid-region
        return t * y

    gx_eager, gy_eager = _grads(build, fused=False)
    gx_fused, gy_fused = _grads(build, fused=True)
    assert _same(gx_eager, gx_fused)
    assert _same(gy_eager, gy_fused)


def test_gradients_flow_through_a_second_backward_use() -> None:
    """A value consumed by two operators is needed by both backwards; eliding it would show here."""
    def build(x, y):
        t = x * y
        return t * t + t

    gx_eager, gy_eager = _grads(build, fused=False)
    gx_fused, gy_fused = _grads(build, fused=True)
    assert _same(gx_eager, gx_fused)
    assert _same(gy_eager, gy_fused)


_POISON_CHECK = """
import numpy as np, magnetron as mag
from magnetron import Tensor
X = np.linspace(0.35, 2.4, 256, dtype=np.float32)
Y = np.linspace(0.6, 1.9, 256, dtype=np.float32)
bad = []
for name, build in [
    ('add',   lambda x, y: ((x + y) + x) + y),
    ('sub',   lambda x, y: ((x - y) - x) - y),
    ('neg',   lambda x, y: (-(x + y)) + y),
    ('mixed', lambda x, y: ((x + y) * x) - y),
]:
    x = Tensor(X.copy(), requires_grad=True)
    y = Tensor(Y.copy(), requires_grad=True)
    with mag.fuse():
        out = build(x, y)
    out.sum().backward()
    for label, t in (('x', x), ('y', y)):
        if t.grad is None:
            continue
        g = t.grad.numpy()
        if not np.isfinite(g).all():
            bad.append(name + '.' + label)
print('BAD=' + ','.join(bad))
"""


def test_declared_dead_values_are_never_read_by_a_backward() -> None:
    """Run with elided values poisoned; a wrong entry in the backward value table turns into NaN.

    mag_op_backward_ignores_value claims certain operands are never read, which is what lets a fused
    chain skip writing them. Filling those buffers with NaN makes a wrong claim fail loudly here
    instead of producing gradients that are merely a bit off.
    """
    env = dict(os.environ, MAG_JIT_POISON='on', MAG_LOG_LEVEL='off')
    proc = subprocess.run([sys.executable, '-c', _POISON_CHECK], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith('BAD=')][-1]
    assert line == 'BAD=', f'a backward read a value declared unneeded: {line[4:]}'
