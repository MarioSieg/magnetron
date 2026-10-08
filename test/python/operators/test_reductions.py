# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch.nn.functional

from ..common import *

_ALL_DTYPE_REDUCES = (
    'sum',
    'prod',
    'min',
    'max',
    'argmin',
    'argmax',
    'all',
    'any',
)


@pytest.mark.parametrize('dtype', FLOATING_NO_FLOAT8)
@pytest.mark.parametrize('op', _ALL_DTYPE_REDUCES)
@pytest.mark.parametrize('keepdim', [True, False])
def test_reduce_op(dtype: dtype.DType, op: str, keepdim: bool) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=dtype)
        dim = random_dim(shape)
        tx = totorch(x)
        if dim is None:
            r = call_reduction(x, op, None, keepdim)
            t = getattr(tx, op)()
        else:
            r = call_reduction(x, op, dim, keepdim)
            t = getattr(tx, op)(dim=dim, keepdim=keepdim)

        if not isinstance(t, torch.Tensor):
            t = t[0]  # min, max, argmin, argmax return (values, indices)

        assert_close_mag_torch(r, t, dtype, equal_nan=True)

    for_all_shapes(test)


@pytest.mark.parametrize('dtype', FLOATING_NO_FLOAT8)
@pytest.mark.parametrize('keepdim', [True, False])
def test_reduce_op_mean(dtype: dtype.DType, keepdim: bool) -> None:  # Mean is only for floating point
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=dtype)
        dim = random_dim(shape)
        if dim is None:
            r = x.mean()
            t = totorch(x).mean()
        else:
            r = x.mean(dim=dim, keepdim=keepdim)
            t = totorch(x).mean(dim=dim, keepdim=keepdim)

        assert_close_mag_torch(r, t, dtype, equal_nan=True)

    for_all_shapes(test)


@pytest.mark.parametrize('dtype', FLOATING_NO_FLOAT8)
@pytest.mark.parametrize('largest', [True, False])
def test_reduce_op_topk(dtype: dtype.DType, largest: bool) -> None:
    def test(shape: tuple[int, ...]) -> None:
        if len(shape) == 0:  # topk not defined for 0-dim tensors
            return
        x = random_tensor(shape, dt=dtype)
        k = random.randint(1, max(1, min(shape)))
        dim = random_dim(shape)
        tx = totorch(x)
        if dim is None:
            rv, ri = x.topk(k, largest=largest)
            tv, ti = tx.topk(k, largest=largest)
        else:
            rv, ri = x.topk(k, dim=dim, largest=largest)
            tv, ti = tx.topk(k, dim=dim, largest=largest)

        assert_close_mag_torch(rv, tv, dtype, equal_nan=True)

        axis = (len(shape) - 1) if dim is None else (dim % len(shape))
        mi = totorch(ri)
        gathered = totorch_for_reference(x, dtype).gather(axis, mi)
        torch.testing.assert_close(gathered, totorch_for_reference(rv, dtype), rtol=0, atol=0, equal_nan=True)
        if k > 1:
            srt = mi.sort(dim=axis).values
            assert (srt.narrow(axis, 1, k - 1) != srt.narrow(axis, 0, k - 1)).all()

    for_all_shapes(test)


@pytest.mark.parametrize('dtype', FLOATING_NO_FLOAT8 | dtype.integer)
@pytest.mark.parametrize('descending', [True, False])
def test_sort(dtype: dtype.DType, descending: bool) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=dtype)
        dim = random_dim(shape)
        tx = totorch(x)
        if dim is None:
            rv, ri = x.sort(descending=descending)
            tv, ti = tx.sort(descending=descending, stable=True)
        else:
            rv, ri = x.sort(dim=dim, descending=descending)
            tv, ti = tx.sort(dim=dim, descending=descending, stable=True)
        assert_close_mag_torch(rv, tv, dtype, equal_nan=True)
        assert ri.tolist() == ti.tolist()

    for_all_shapes(test)


@pytest.mark.parametrize('dtype', FLOATING_NO_FLOAT8 | dtype.integer)
@pytest.mark.parametrize('descending', [True, False])
def test_argsort(dtype: dtype.DType, descending: bool) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=dtype)
        dim = random_dim(shape)
        tx = totorch(x)
        if dim is None:
            ri = x.argsort(descending=descending)
            ti = tx.argsort(descending=descending, stable=True)
        else:
            ri = x.argsort(dim=dim, descending=descending)
            ti = tx.argsort(dim=dim, descending=descending, stable=True)
        assert ri.tolist() == ti.tolist()

    for_all_shapes(test)


_INT_REDUCES = ('sum', 'prod', 'min', 'max', 'argmin', 'argmax')
_INT_DTYPES = tuple(sorted(dtype.integer, key=lambda d: d.name))


def _int_reference(tx64: torch.Tensor, op: str, dim: int | None, keepdim: bool) -> torch.Tensor:
    t = getattr(tx64, op)() if dim is None else getattr(tx64, op)(dim=dim, keepdim=keepdim)
    return t if isinstance(t, torch.Tensor) else t[0]


def _int_result_dtype(dt: dtype.DType, op: str) -> dtype.DType:
    if op in ('argmin', 'argmax'):
        return dtype.int64
    if op in ('sum', 'prod'):
        return dtype.int64 if dt.is_signed_integer() else dtype.uint64
    return dt


@pytest.mark.parametrize('dt', _INT_DTYPES, ids=[d.name for d in _INT_DTYPES])
@pytest.mark.parametrize('op', _INT_REDUCES)
@pytest.mark.parametrize('keepdim', [True, False])
def test_reduce_op_integer(dt: dtype.DType, op: str, keepdim: bool) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = uniform_tensor(shape, low=1 if op == 'prod' else 0, high=4, dtype=dt)
        dim = random_dim(shape)
        r = call_reduction(x, op, dim, keepdim)
        t = _int_reference(totorch(x).to(torch.int64), op, dim, keepdim)
        assert r.dtype == _int_result_dtype(dt, op)
        assert r.shape == tuple(t.shape)
        assert r.tolist() == t.to(totorch_dtype(r.dtype)).tolist()

    for_all_shapes(test)


@pytest.mark.parametrize('dt', _INT_DTYPES, ids=[d.name for d in _INT_DTYPES])
@pytest.mark.parametrize('op', ['argmin', 'argmax'])
def test_arg_reduction_returns_first_occurrence(dt: dtype.DType, op: str) -> None:
    for shape in ((8,), (4, 6), (2, 3, 5), (64,), (3, 129)):
        x = uniform_tensor(shape, low=0, high=2, dtype=dt)
        tx = totorch(x).to(torch.int64)
        assert getattr(x, op)().tolist() == getattr(tx, op)().tolist()
        for dim in range(-len(shape), len(shape)):
            for keepdim in (False, True):
                r = getattr(x, op)(dim, keepdim=keepdim)
                assert r.tolist() == getattr(tx, op)(dim=dim, keepdim=keepdim).tolist()


@pytest.mark.parametrize('dt', tuple(sorted(FLOATING_NO_FLOAT8 | dtype.integer | {dtype.boolean}, key=lambda d: d.name)), ids=lambda d: d.name)
@pytest.mark.parametrize('op', ['all', 'any'])
@pytest.mark.parametrize('keepdim', [True, False])
def test_all_any_matches_torch(dt: dtype.DType, op: str, keepdim: bool) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = uniform_tensor(shape, low=0, high=3, dtype=dtype.int32).cast(dt)
        tx = totorch(x)
        dim = random_dim(shape)
        r = call_reduction(x, op, dim, keepdim)
        t = getattr(tx, op)() if dim is None else getattr(tx, op)(dim=dim, keepdim=keepdim)
        assert r.dtype == dtype.boolean
        assert r.shape == tuple(t.shape)
        assert r.tolist() == t.tolist()

    for_all_shapes(test)


@pytest.mark.parametrize('dt', tuple(sorted(FLOATING_NO_FLOAT8, key=lambda d: d.name)), ids=lambda d: d.name)
@pytest.mark.parametrize('op', ['sum', 'mean', 'prod', 'min', 'max', 'argmin', 'argmax'])
def test_reduce_multi_dim_matches_torch(dt: dtype.DType, op: str) -> None:
    shape = (2, 3, 4, 5)
    x = uniform_tensor(shape, low=0.8, high=1.25, dtype=dt)
    tx = totorch(x)
    for dims in ((0, 1), (1, 3), (0, 2, 3), (-1, -2), (0, 1, 2, 3)):
        for keepdim in (False, True):
            if op in ('min', 'max', 'argmin', 'argmax'):
                continue
            r = getattr(x, op)(dim=dims, keepdim=keepdim)
            if op == 'prod':
                t = tx.to(torch.float64)
                for d in sorted(d % tx.dim() for d in dims):
                    t = t.prod(dim=d, keepdim=True)
                if not keepdim:
                    t = t.squeeze()
                t = t.to(totorch_dtype(dt))
            else:
                t = getattr(tx, op)(dim=dims, keepdim=keepdim)
            assert_close_mag_torch(r, t, dt)
