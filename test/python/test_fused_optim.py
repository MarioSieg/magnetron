"""The fusion JIT must not change what the optimizer computes.

Adam's update is a chain of pointwise ops; the JIT collapses it into one generated kernel. These
tests pin that the fused path agrees with the eager path it replaces, and with torch, so a wrong
kernel cannot silently degrade training.
"""

import os
import subprocess
import sys

from .common import *

import pytest
import torch

from magnetron import nn, optim
from magnetron._magnetron_bindings import _fused_adam_supported


_TRAIN = """
import numpy as np, sys
from magnetron import Tensor, nn, optim, context, dtype

context.set_default_dtype(getattr(dtype, sys.argv[1]))
context.manual_seed(1234)
x = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(4, 64)

layer = nn.Linear(64, 8, bias=False)   # seeded, so both subprocesses start identically
opt = optim.Adam(layer.parameters(), lr=1e-2)
inp = Tensor(x).cast(context.get_default_dtype())
for _ in range(12):
    loss = (layer(inp) ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
w = layer.weight.cast(dtype.float32).numpy().ravel()[:32]  # widening is exact
print(' '.join(f'{v:.9e}' for v in w))
print('jit', context.jit_stats()['kernels_compiled'])
"""


def _run_training(jit: str, dt: str = 'float32') -> tuple[list[float], int]:
    env = dict(os.environ, MAG_JIT=jit, MAG_LOG_LEVEL='off')
    proc = subprocess.run([sys.executable, '-c', _TRAIN, dt], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln]
    weights = [float(v) for v in lines[-2].split()]
    compiled = int(lines[-1].split()[1])
    return weights, compiled


def test_fused_adam_matches_eager_bitwise() -> None:
    """Same arithmetic in a different order would still drift over 12 steps, so require equality."""
    fused, fused_compiles = _run_training('on')
    eager, eager_compiles = _run_training('off')
    assert fused_compiles == 1, 'JIT did not compile a kernel, so this compared eager against eager'
    assert eager_compiles == 0, 'MAG_JIT=off still compiled a kernel'
    assert fused == eager, 'fused optimizer diverged from the eager one'


def test_fused_adam_matches_eager_bitwise_float16() -> None:
    """float16 rounds every intermediate back to storage in the eager path, and converts scalar
    operands to the tensor dtype before use. The generated kernel has to do both at the same points
    or the weights drift."""
    fused, fused_compiles = _run_training('on', 'float16')
    eager, eager_compiles = _run_training('off', 'float16')
    assert fused_compiles == 1, 'JIT declined float16, so this compared eager against eager'
    assert eager_compiles == 0
    assert fused == eager, 'fused float16 optimizer diverged from the eager one'


def test_bfloat16_is_left_to_the_eager_kernels() -> None:
    """bfloat16 conversion is soft-fp, and the generated kernel would run it once per operator per
    element while the eager kernels convert a vector at a time. Measured slower fused than eager,
    so the JIT declines it."""
    _, compiles = _run_training('on', 'bfloat16')
    assert compiles == 0, 'bfloat16 should not reach the JIT'


def test_fused_adam_matches_torch() -> None:
    context.manual_seed(99)
    w = np.linspace(-0.3, 0.3, 256, dtype=np.float32).reshape(8, 32)
    x = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(4, 32)

    layer = nn.Linear(32, 8, bias=False)
    layer.weight.data = Tensor(w)
    layer.weight.requires_grad = True  # assigning .data replaces the tensor and clears the flag
    opt = optim.Adam(layer.parameters(), lr=1e-2)
    inp = Tensor(x)

    tw = torch.nn.Linear(32, 8, bias=False)
    with torch.no_grad():
        tw.weight.copy_(torch.from_numpy(w))
    topt = torch.optim.Adam(tw.parameters(), lr=1e-2)
    tin = torch.from_numpy(x)

    for _ in range(10):
        loss = (layer(inp) ** 2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        tloss = (tw(tin) ** 2).sum()
        topt.zero_grad()
        tloss.backward()
        topt.step()

    got = torch.from_numpy(layer.weight.numpy())
    torch.testing.assert_close(got, tw.weight.detach(), rtol=1e-4, atol=1e-5)


def test_fused_adam_declines_non_contiguous() -> None:
    """A transposed parameter is not contiguous, so the JIT must refuse and the eager path run."""
    a = Tensor.uniform((8, 16), low=0.0, high=1.0)
    b = Tensor.uniform((8, 16), low=0.0, high=1.0)
    c = Tensor.zeros((8, 16))
    d = Tensor.zeros((8, 16))
    assert _fused_adam_supported(a, b, c, d) is True
    assert _fused_adam_supported(a.T, b, c, d) is False
