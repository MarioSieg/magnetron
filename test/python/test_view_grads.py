"""Gradients through views, checked against torch.

mag_op_backward_strided_view has a fast path for views that are a bijection onto the whole base and
a general scatter for everything else. A wrong fast path corrupts the weight gradient of every
nn.Linear, so both paths are exercised here.
"""

import pytest
import torch

from .common import *

# name, view, and which path it is meant to take
BIJECTIVE = [
    ('transpose_2d', lambda t: t.T, (16, 12)),  # the nn.Linear case: x @ weight.T
    ('transpose_square', lambda t: t.T, (8, 8)),
    ('permute_3d', lambda t: t.permute(2, 0, 1), (3, 4, 5)),
    ('permute_4d', lambda t: t.permute(1, 3, 0, 2), (2, 3, 4, 5)),
    ('reshape', lambda t: t.reshape(6, 4), (3, 8)),
    ('flatten', lambda t: t.flatten(0, -1), (3, 4, 2)),
    ('unsqueeze', lambda t: t.reshape(1, 4, 6), (4, 6)),
    # Size-1 dims carry a meaningless stride, so dim matching must key on extent for them.
    ('size_one_dims', lambda t: t.permute(1, 0, 2), (1, 5, 1)),
]

GENERAL = [
    ('slice', lambda t: t[2:6], (10, 3)),  # does not cover the base: untouched elements stay zero
    ('double_transpose', lambda t: t.T.T, (7, 5)),
    ('transpose_of_slice', lambda t: t[1:5].T, (8, 6)),  # non-contiguous base: fast path must decline
    ('two_views_of_one_base', lambda t: t.T.reshape(-1) + t.reshape(-1), (6, 6)),  # must accumulate
]


def _check(view, shape: tuple[int, ...]) -> None:
    data = torch.rand(shape, dtype=torch.float32)

    tx = data.clone().requires_grad_(True)
    tout = view(tx)
    (tout * tout).sum().backward()

    mx = Tensor(data.numpy(), requires_grad=True)
    mout = view(mx)
    (mout * mout).sum().backward()

    got = torch.from_numpy(mx.grad.numpy())
    assert got.shape == tx.grad.shape, f'grad shape {tuple(got.shape)} != {tuple(tx.grad.shape)}'
    torch.testing.assert_close(got, tx.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('name,view,shape', BIJECTIVE, ids=[c[0] for c in BIJECTIVE])
def test_bijective_views(name: str, view, shape: tuple[int, ...]) -> None:
    _check(view, shape)


@pytest.mark.parametrize('name,view,shape', GENERAL, ids=[c[0] for c in GENERAL])
def test_general_views(name: str, view, shape: tuple[int, ...]) -> None:
    _check(view, shape)
