# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from ..common import *

def _maybe_negative_axes(axes, nd):
    out = []
    for ax in axes:
        out.append(ax - nd if random.random() < 0.5 else ax)
    return out

def reduce_op(
        dtype: DataType,
        mag_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor],
        torch_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor]
) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = Tensor.bernoulli(shape) if dtype == boolean else Tensor.uniform(shape, dtype=dtype)
        r = mag_callback(x.clone())
        torch.testing.assert_close(totorch(r), torch_callback(totorch(x)))

    for_all_shapes(test)

@pytest.mark.parametrize('dtype', dtype.FLOATING_POINT_DTYPES)
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_mean(dtype: DataType, keepdim: bool):
    nd = random.randint(1, 4)
    shape = tuple(random.randint(1, 5) for _ in range(nd))
    k = random.randint(0, nd)
    axes = sorted(random.sample(range(nd), k))
    axes = _maybe_negative_axes(axes, nd)
    dim_arg = axes if len(axes) > 0 else None
    reduce_op(dtype, lambda x: x.mean(dim=dim_arg, keepdim=keepdim), lambda x: x.mean(dim=dim_arg, keepdim=keepdim))

@pytest.mark.parametrize('dtype', dtype.NUMERIC_DTYPES)
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_min(dtype: DataType, keepdim: bool):
    nd = random.randint(1, 4)
    shape = tuple(random.randint(1, 5) for _ in range(nd))
    k = random.randint(0, nd)
    axes = sorted(random.sample(range(nd), k))
    axes = _maybe_negative_axes(axes, nd)
    dim_arg = tuple(set(axes)) if len(axes) > 0 else None
    reduce_op(dtype, lambda x: x.min(dim=dim_arg, keepdim=keepdim), lambda x: x.amin(dim_arg, keepdim))

@pytest.mark.parametrize('dtype', dtype.NUMERIC_DTYPES)
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_max(dtype: DataType, keepdim: bool):
    nd = random.randint(1, 4)
    shape = tuple(random.randint(1, 5) for _ in range(nd))
    k = random.randint(0, nd)
    axes = sorted(random.sample(range(nd), k))
    axes = _maybe_negative_axes(axes, nd)
    dim_arg = tuple(set(axes)) if len(axes) > 0 else None
    reduce_op(dtype, lambda x: x.max(dim=dim_arg, keepdim=keepdim), lambda x: x.amax(dim=dim_arg, keepdim=keepdim))

@pytest.mark.parametrize('dtype', dtype.NUMERIC_DTYPES)
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_sum(dtype: DataType, keepdim: bool):
    nd = random.randint(1, 4)
    shape = tuple(random.randint(1, 5) for _ in range(nd))
    k = random.randint(0, nd)
    axes = sorted(random.sample(range(nd), k))
    axes = _maybe_negative_axes(axes, nd)
    dim_arg = axes if len(axes) > 0 else None
    reduce_op(dtype, lambda x: x.sum(dim=dim_arg, keepdim=keepdim), lambda x: x.sum(dim=dim_arg, keepdim=keepdim))
