# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch
import torch.nn.functional as F

import magnetron as mag
import magnetron.nn as nn
from magnetron.nn.init import inplace_init, UniformInitStrategy

from .common import *

_GN_CASES = [
    # (shape, num_groups)
    ((2, 6), 2),
    ((2, 6, 5), 2),
    ((3, 8, 4, 4), 4),
    ((1, 4, 3, 5), 1),
    ((2, 4, 2, 3, 3), 4),
]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('case', _GN_CASES)
def test_group_norm(device: str, affine: bool, case) -> None:
    shape, groups = case
    channels = shape[1]
    with mag.device(device):
        layer = nn.GroupNorm(groups, channels, eps=1e-5, affine=affine)
        if affine:
            inplace_init(layer.weight, UniformInitStrategy(0.5, 1.5))
            inplace_init(layer.bias, UniformInitStrategy(-0.5, 0.5))
        x = random_tensor(shape, dtype.float32, device)
        tx = totorch(x).clone()
        x.requires_grad = True
        tx.requires_grad = True
        y = layer(x)
        tw = totorch(layer.weight).clone().requires_grad_(True) if affine else None
        tb = totorch(layer.bias).clone().requires_grad_(True) if affine else None
        ty = F.group_norm(tx, groups, tw, tb, eps=1e-5)
        assert y.shape == tuple(ty.shape)
        assert_close_mag_torch(y, ty, dtype.float32)
        w = random_tensor(y.shape, dtype.float32, device)
        (y * w).sum().backward()
        (ty * totorch(w)).sum().backward()
        assert_close_mag_torch(x.grad, tx.grad, dtype.float32)
        if affine:
            assert set(dict(layer.named_parameters()).keys()) == {'weight', 'bias'}
            assert_close_mag_torch(layer.weight.grad, tw.grad, dtype.float32, rtol=1e-4, atol=1e-3)
            assert_close_mag_torch(layer.bias.grad, tb.grad, dtype.float32, rtol=1e-4, atol=1e-3)
        else:
            assert layer.weight is None and layer.bias is None
            assert dict(layer.named_parameters()) == {}


def test_group_norm_invalid_args() -> None:
    with pytest.raises(ValueError):
        nn.GroupNorm(4, 6)
    with pytest.raises(ValueError):
        nn.GroupNorm(0, 6)
    with pytest.raises(ValueError):
        nn.GroupNorm(2, 6)(Tensor.uniform(2, 4, 3))
