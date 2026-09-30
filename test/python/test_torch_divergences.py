# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import *

_STRICT = dict(strict=True)


@pytest.mark.xfail(reason='in-place ops on a view of a grad-requiring tensor raise; torch rebases the base through CopySlices', **_STRICT)
def test_inplace_on_view_rebases_like_torch() -> None:
    x = uniform_tensor((3, 4), -2.0, 2.0)
    tx = totorch(x).clone().requires_grad_(True)
    x.requires_grad = True
    y = x * 2.0
    ty = tx * 2.0
    y[0].mul_(3.0)
    ty[0].mul_(3.0)
    y.sum().backward()
    ty.sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)


@pytest.mark.xfail(reason='floor division has a zero gradient here; torch has no derivative for floor_divide at all', **_STRICT)
def test_floordiv_backward_like_torch() -> None:
    x = uniform_tensor((3, 4), -5.0, 5.0)
    tx = totorch(x).clone().requires_grad_(True)
    x.requires_grad = True
    (tx // 2.0).sum().backward()
    (x // 2.0).sum().backward()
