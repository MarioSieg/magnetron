# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from ..common import *

def _maybe_negative_axes(axes, nd):
    out = []
    for ax in axes:
        out.append(ax - nd if random.random() < 0.5 else ax)
    return out

@pytest.mark.parametrize('dtype', [float16, float32])
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_mean(dtype: DataType, keepdim: bool):
    nd = random.randint(1, 4)
    shape = tuple(random.randint(1, 5) for _ in range(nd))
    k = random.randint(0, nd)
    axes = sorted(random.sample(range(nd), k))
    axes = _maybe_negative_axes(axes, nd)
    dim_arg = axes if len(axes) > 0 else None
    unary_op(dtype, lambda x: x.mean(dim=dim_arg, keepdim=keepdim), lambda x: x.mean(dim=dim_arg, keepdim=keepdim))

@pytest.mark.parametrize('dtype', [float16, float32])
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_min(dtype: DataType, keepdim: bool):
    nd = random.randint(1, 4)
    shape = tuple(random.randint(1, 5) for _ in range(nd))
    k = random.randint(0, nd)
    axes = sorted(random.sample(range(nd), k))
    axes = _maybe_negative_axes(axes, nd)
    dim_arg = axes if len(axes) > 0 else None
    unary_op(dtype, lambda x: x.min(dim=dim_arg, keepdim=keepdim), lambda x: x.min(dim=dim_arg, keepdim=keepdim))

@pytest.mark.parametrize('dtype', [float16, float32])
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_max(dtype: DataType, keepdim: bool):
    nd = random.randint(1, 4)
    shape = tuple(random.randint(1, 5) for _ in range(nd))
    k = random.randint(0, nd)
    axes = sorted(random.sample(range(nd), k))
    axes = _maybe_negative_axes(axes, nd)
    dim_arg = axes if len(axes) > 0 else None
    unary_op(dtype, lambda x: x.max(dim=dim_arg, keepdim=keepdim), lambda x: x.max(dim=dim_arg, keepdim=keepdim))

@pytest.mark.parametrize('dtype', [float16, float32])
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_sum(dtype: DataType, keepdim: bool):
    nd = random.randint(1, 4)
    shape = tuple(random.randint(1, 5) for _ in range(nd))
    k = random.randint(0, nd)
    axes = sorted(random.sample(range(nd), k))
    axes = _maybe_negative_axes(axes, nd)
    dim_arg = axes if len(axes) > 0 else None
    unary_op(dtype, lambda x: x.sum(dim=dim_arg, keepdim=keepdim), lambda x: x.sum(dim=dim_arg, keepdim=keepdim))
