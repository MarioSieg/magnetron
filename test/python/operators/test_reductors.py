# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from ..common import *


def _maybe_negative_axes(axes, nd):
    out = []
    for ax in axes:
        out.append(ax - nd if random.random() < 0.5 else ax)
    return out


def reduce_op_with_random_axes(
        dtype: DataType,
        make_mag: Callable[[list[int] | None, bool], Callable[[Tensor], Tensor]],
        make_torch: Callable[[list[int] | None, bool], Callable[[torch.Tensor], torch.Tensor]],
        keepdim: bool,
) -> None:
    def test(shape: tuple[int, ...]) -> None:
        nd = len(shape)
        k = random.randint(0, nd)
        axes = sorted(random.sample(range(nd), k))
        axes = _maybe_negative_axes(axes, nd)
        dim_arg = axes if axes else None

        x = Tensor.bernoulli(shape) if dtype == boolean else Tensor.uniform(shape, dtype=dtype)
        mag_cb = make_mag(dim_arg, keepdim)
        torch_cb = make_torch(dim_arg, keepdim)

        r = mag_cb(x.clone())
        r_t = totorch(r)
        t = torch_cb(totorch(x))

        # fix scalar vs (1,) mismatch (see next section)
        if r_t.ndim == 1 and r_t.shape[0] == 1 and t.ndim == 0:
            t = t.reshape(1)

        torch.testing.assert_close(r_t, t)

    for_all_shapes(test)


@pytest.mark.parametrize('dtype', dtype.FLOATING_POINT_DTYPES)
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_mean(dtype: DataType, keepdim: bool):
    reduce_op_with_random_axes(
        dtype,
        make_mag   = lambda dim, keep: (lambda x: x.mean(dim=dim, keepdim=keep)),
        make_torch = lambda dim, keep: (lambda x: x.mean(dim=dim, keepdim=keep)),
        keepdim=keepdim,
    )

@pytest.mark.parametrize('dtype', dtype.NUMERIC_DTYPES)
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_sum(dtype: DataType, keepdim: bool):
    reduce_op_with_random_axes(
        dtype,
        make_mag   = lambda dim, keep: (lambda x: x.sum(dim=dim, keepdim=keep)),
        make_torch = lambda dim, keep: (lambda x: x.sum(dim=dim, keepdim=keep)),
        keepdim=keepdim,
    )

@pytest.mark.parametrize('dtype', dtype.NUMERIC_DTYPES)
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_prod(dtype: DataType, keepdim: bool):
    reduce_op_with_random_axes(
        dtype,
        make_mag   = lambda dim, keep: (lambda x: x.prod(dim=dim, keepdim=keep)),
        make_torch = lambda dim, keep: (lambda x: x.prod(dim=dim, keepdim=keep)),
        keepdim=keepdim,
    )

@pytest.mark.parametrize('dtype', dtype.NUMERIC_DTYPES)
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_min(dtype: DataType, keepdim: bool):
    reduce_op_with_random_axes(
        dtype,
        make_mag   = lambda dim, keep: (lambda x: x.min(dim=dim, keepdim=keep)),
        make_torch = lambda dim, keep: (lambda x: x.amin(dim=dim, keepdim=keep)),
        keepdim=keepdim,
    )

@pytest.mark.parametrize('dtype', dtype.NUMERIC_DTYPES)
@pytest.mark.parametrize('keepdim', [False, True])
def test_reduction_max(dtype: DataType, keepdim: bool):
    reduce_op_with_random_axes(
        dtype,
        make_mag   = lambda dim, keep: (lambda x: x.max(dim=dim, keepdim=keep)),
        make_torch = lambda dim, keep: (lambda x: x.amax(dim=dim, keepdim=keep)),
        keepdim=keepdim,
    )
