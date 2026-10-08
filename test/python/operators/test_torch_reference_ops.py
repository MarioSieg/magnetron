# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..common import *

_F32_I32 = (dtype.float32, dtype.int32)
_F32_I32_BOOL = (dtype.float32, dtype.int32, dtype.boolean)


def _rand(shape: tuple[int, ...], dt: dtype.DType, device: str) -> Tensor:
    return random_tensor(shape, dt, device)


def _same(r: Tensor, ref: torch.Tensor, dt: dtype.DType) -> None:
    assert r.shape == tuple(ref.shape)
    if ref.dtype == torch.bool:
        assert r.dtype == dtype.boolean
        assert r.tolist() == ref.tolist()
    elif dt.is_integer():
        assert r.tolist() == ref.tolist()
    else:
        assert_close_mag_torch(r, ref, dt)


_EXPAND_CASES = (((3, 1), (3, 4)), ((1, 4), (3, 4)), ((3, 1, 5), (3, 4, 5)), ((3,), (2, 3)), ((3, 1), (2, 3, 4)), ((1,), (2, 3)), ((), (2, 3)), ((2, 1, 1), (2, 3, 4)))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32_BOOL, ids=[d.name for d in _F32_I32_BOOL])
@pytest.mark.parametrize('method', ['expand', 'broadcast'])
def test_expand_matches_torch(device: str, dt: dtype.DType, method: str) -> None:
    for src, dst in _EXPAND_CASES:
        x = _rand(src, dt, device)
        tx = totorch(x)
        r = getattr(x, method)(*dst)
        _same(r, tx.expand(*dst), dt)
        if method == 'expand' and len(src) == len(dst):
            keep = tuple(-1 if s == d else d for s, d in zip(src, dst))
            _same(getattr(x, method)(*keep), tx.expand(*keep), dt)


_STRIDED_CASES = (
    ((24,), (3, 4), (4, 1), 0),
    ((24,), (4, 3), (1, 4), 0),
    ((24,), (3, 3), (2, 1), 1),
    ((4, 6), (2, 3), (12, 2), 1),
    ((4, 6), (6, 4), (1, 6), 0),
    ((24,), (5,), (5,), 2),
    ((24,), (2, 2, 2), (8, 4, 1), 3),
    ((4, 6), (3, 4), (1, 1), 2),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.int64, dtype.float32), ids=['int64', 'float32'])
def test_strided_view_matches_as_strided(device: str, dt: dtype.DType) -> None:
    for base, shape, strides, offset in _STRIDED_CASES:
        x = Tensor.arange(24, device=device).reshape(*base).cast(dt)
        tx = totorch(x)
        _same(x.strided_view(shape, strides, offset=offset), torch.as_strided(tx, shape, strides, offset), dt)
        if offset == 0:
            _same(Tensor.strided_view(x, shape, strides), torch.as_strided(tx, shape, strides), dt)


_VIEW_SLICE_CASES = (((10,), 0, 1, 4, 2), ((4, 6), 1, 0, 3, 2), ((4, 6), -1, 1, 2, 3), ((3, 4, 5), 1, 1, 3, 1), ((3, 4, 5), 0, 0, 2, 2), ((3, 4, 5), 2, 4, 1, 1), ((3, 4, 5), -2, 3, 1, 5))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_view_slice_matches_torch_slicing(device: str, dt: dtype.DType) -> None:
    for shape, dim, start, length, step in _VIEW_SLICE_CASES:
        x = _rand(shape, dt, device)
        tx = totorch(x)
        idx = [slice(None)] * len(shape)
        idx[dim] = slice(start, start + (length - 1) * step + 1, step)
        r = x.view_slice(dim, start, length, step)
        assert r.is_view
        _same(r, tx[tuple(idx)], dt)


_REPEAT_CASES = (((3,), (2,)), ((3,), (2, 2)), ((2, 3), (2, 1)), ((2, 3), (1, 2)), ((2, 3), (3, 2, 2)), ((2, 1, 3), (1, 4, 1)), ((), (2, 3)), ((1, 1), (2, 3)), ((2, 3, 4), (2, 1, 3)))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32_BOOL, ids=[d.name for d in _F32_I32_BOOL])
def test_repeat_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape, reps in _REPEAT_CASES:
        x = _rand(shape, dt, device)
        _same(x.repeat(*reps), totorch(x).repeat(*reps), dt)


_RI_CASES = (((3,), 2, None), ((2, 3), 2, None), ((2, 3), 3, 1), ((2, 3), 2, 0), ((2, 3), 2, -1), ((2, 3, 4), 3, 1), ((3,), [1, 2, 3], 0), ((2, 3), [2, 1], 0), ((2, 3), [1, 0, 2], 1), ((3,), [0, 2, 1], None), ((2, 3, 4), [2, 0, 1, 3], 2))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.int64), ids=['float32', 'int64'])
def test_repeat_interleave_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape, reps, dim in _RI_CASES:
        x = _rand(shape, dt, device)
        tx = totorch(x)
        mreps = Tensor(reps, device=device) if isinstance(reps, list) else reps
        treps = torch.tensor(reps) if isinstance(reps, list) else reps
        if dim is None:
            _same(x.repeat_interleave(mreps), tx.repeat_interleave(treps), dt)
        else:
            _same(x.repeat_interleave(mreps, dim=dim), tx.repeat_interleave(treps, dim=dim), dt)


_INDEX_ADD_CASES = (((5, 3), 0, [0, 4, 2], (3, 3), 1.0), ((5, 3), 1, [2, 0], (5, 2), 1.0), ((5, 3), 0, [1, 1, 3], (3, 3), -1.0), ((2, 4, 3), 1, [3, 0, 3], (2, 3, 3), 2.0), ((4,), 0, [3, 1], (2,), 1.0), ((2, 4, 3), -1, [0, 2], (2, 4, 2), 1.0), ((6,), 0, [5, 5, 5, 0], (4,), 3.0))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_index_add_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape, dim, index, src_shape, alpha in _INDEX_ADD_CASES:
        x = _rand(shape, dt, device)
        src = _rand(src_shape, dt, device)
        tx, tsrc = totorch(x), totorch(src)
        a = int(alpha) if dt.is_integer() else alpha
        x.index_add_(dim, Tensor(index, device=device), src, alpha=a)
        _same(x, tx.index_add_(dim, torch.tensor(index), tsrc, alpha=a), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16, dtype.int32), ids=['float32', 'float16', 'int32'])
def test_outer_matches_torch(device: str, dt: dtype.DType) -> None:
    for n, m in ((1, 1), (3, 4), (7, 5), (16, 33), (1, 9)):
        a = _rand((n,), dt, device)
        b = _rand((m,), dt, device)
        _same(a.outer(b), torch.outer(totorch(a), totorch(b)), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16), ids=['float32', 'float16'])
@pytest.mark.parametrize('shape', [(4,), (3, 4), (2, 3, 5)])
def test_lerp_matches_torch(device: str, dt: dtype.DType, shape) -> None:
    a = _rand(shape, dt, device)
    b = _rand(shape, dt, device)
    ta, tb = totorch(a), totorch(b)
    for w in (0.0, 0.3, 1.0, 1.7, -0.5):
        _same(a.lerp(b, w), torch.lerp(ta, tb, w), dt)
        c = a.clone()
        c.lerp_(b, w)
        _same(c, torch.lerp(ta, tb, w), dt)
    w = uniform_tensor(shape, low=-0.5, high=1.5, dtype=dt, device=device)
    _same(a.lerp(b, w), torch.lerp(ta, tb, totorch(w)), dt)
    c = a.clone()
    c.lerp_(b, w)
    _same(c, torch.lerp(ta, tb, totorch(w)), dt)
    row = uniform_tensor((shape[-1],), low=-0.5, high=1.5, dtype=dt, device=device)
    _same(a.lerp(b, row), torch.lerp(ta, tb, totorch(row)), dt)


_GATHER_CASES = (((3, 4), 0, (2, 4)), ((3, 4), 1, (3, 2)), ((3, 4), 0, (5, 4)), ((3, 4), 1, (2, 1)), ((2, 3, 4), 2, (2, 3, 6)), ((2, 3, 4), 0, (1, 3, 4)), ((2, 3, 4), 2, (2, 2, 2)), ((2, 3, 4), -1, (2, 2, 2)), ((5,), 0, (8,)), ((2, 3, 4), 1, (2, 5, 4)), ((2, 3, 4), 1, (2, 5, 3)), ((2, 3, 4), 0, (3, 1, 1)), ((2, 3, 4), -2, (2, 1, 1)))


def _gather_index(shape: tuple[int, ...], dim: int, idx_shape: tuple[int, ...], device: str) -> tuple[Tensor, torch.Tensor]:
    tidx = torch.randint(0, shape[dim], idx_shape)
    return Tensor(tidx.tolist(), device=device).reshape(*idx_shape) if tidx.numel() else Tensor(tidx.tolist(), device=device), tidx


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_gather_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape, dim, idx_shape in _GATHER_CASES:
        x = _rand(shape, dt, device)
        idx, tidx = _gather_index(shape, dim, idx_shape, device)
        _same(x.gather(dim, idx), totorch(x).gather(dim, tidx), dt)


_SCATTER_CASES = (((3, 5), 0, (2, 5)), ((3, 5), 1, (3, 3)), ((2, 3, 4), 2, (2, 3, 2)), ((2, 3, 4), 0, (1, 3, 4)), ((4,), 0, (3,)), ((2, 3, 4), -1, (2, 3, 4)), ((5, 4), 1, (3, 2)))


def _unique_index(shape: tuple[int, ...], dim: int, idx_shape: tuple[int, ...]) -> torch.Tensor:
    d = dim % len(shape)
    full = list(idx_shape)
    full[d] = shape[d]
    perm = torch.rand(full).argsort(dim=d)
    return perm.narrow(d, 0, idx_shape[d])


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_scatter_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape, dim, idx_shape in _SCATTER_CASES:
        base = _rand(shape, dt, device)
        src = _rand(idx_shape, dt, device)
        tidx = _unique_index(shape, dim, idx_shape)
        idx = Tensor(tidx.tolist(), device=device)
        tbase, tsrc = totorch(base), totorch(src)
        expected = tbase.scatter(dim, tidx, tsrc)
        _same(base.scatter(dim, idx, src), expected, dt)
        _same(base, tbase, dt)
        b = base.clone()
        b.scatter_(dim, idx, src)
        _same(b, expected, dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_scatter_add_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape, dim, idx_shape in _SCATTER_CASES:
        base = _rand(shape, dt, device)
        src = _rand(idx_shape, dt, device)
        tidx = torch.randint(0, shape[dim], idx_shape)
        idx = Tensor(tidx.tolist(), device=device)
        tbase, tsrc = totorch(base), totorch(src)
        expected = tbase.scatter_add(dim, tidx, tsrc)
        _same(base.scatter_add(dim, idx, src), expected, dt)
        _same(base, tbase, dt)
        b = base.clone()
        b.scatter_add_(dim, idx, src)
        _same(b, expected, dt)


_FLIP_SHAPES = ((5,), (3, 4), (2, 3, 4), (2, 1, 3, 2))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32_BOOL, ids=[d.name for d in _F32_I32_BOOL])
def test_flip_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape in _FLIP_SHAPES:
        x = _rand(shape, dt, device)
        tx = totorch(x)
        rank = len(shape)
        for r in range(1, rank + 1):
            for dims in itertools.combinations(range(-rank, rank), r):
                if len({d % rank for d in dims}) != len(dims):
                    continue
                _same(x.flip(*dims), tx.flip(*dims), dt)


_CUM_SHAPES = ((7,), (3, 5), (2, 3, 4), (2, 1, 5, 3))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.int32, dtype.int64), ids=['float32', 'int32', 'int64'])
def test_cumsum_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape in _CUM_SHAPES:
        x = _rand(shape, dt, device)
        tx = totorch(x)
        for dim in range(-len(shape), len(shape)):
            r = x.cusum(dim)
            assert r.dtype == dt
            _same(r, tx.cumsum(dim).to(totorch_dtype(dt)), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.int32, dtype.int64), ids=['float32', 'int32', 'int64'])
def test_cumprod_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape in _CUM_SHAPES:
        x = uniform_tensor(shape, low=1, high=4, dtype=dt, device=device) if dt.is_integer() else uniform_tensor(shape, low=0.8, high=1.25, dtype=dt, device=device)
        tx = totorch(x)
        for dim in range(-len(shape), len(shape)):
            r = x.cuprod(dim)
            assert r.dtype == dt
            _same(r, tx.cumprod(dim).to(totorch_dtype(dt)), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('op', ['cumax', 'cumin'])
@pytest.mark.parametrize('dt', (dtype.float32, dtype.int32), ids=['float32', 'int32'])
def test_cummax_cummin_matches_torch(device: str, op: str, dt: dtype.DType) -> None:
    tname = 'cummax' if op == 'cumax' else 'cummin'
    for shape in _CUM_SHAPES:
        if dt.is_integer():
            tx = torch.randperm(int(np.prod(shape))).reshape(shape).to(torch.int32)
            x = Tensor(tx.tolist(), dtype=dt, device=device)
        else:
            x = _rand(shape, dt, device)
            tx = totorch(x)
        for dim in range(-len(shape), len(shape)):
            values, indices = getattr(x, op)(dim)
            t = getattr(tx, tname)(dim)
            assert indices.dtype == dtype.int64
            _same(values, t.values, dt)
            assert indices.tolist() == t.indices.tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_one_hot_matches_torch(device: str) -> None:
    for shape in ((5,), (2, 3), (2, 2, 2), (1,)):
        tidx = torch.randint(0, 4, shape)
        idx = Tensor(tidx.tolist(), device=device)
        r = idx.one_hot()
        assert r.dtype == dtype.int64
        _same(r, F.one_hot(tidx), dtype.int64)
        _same(idx.one_hot(num_classes=7), F.one_hot(tidx, num_classes=7), dtype.int64)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.int64), ids=['float32', 'int64'])
def test_meshgrid_ij_matches_torch(device: str, dt: dtype.DType) -> None:
    for lengths in ((3, 2), (2, 3, 4), (1, 5), (4,)):
        xs = [_rand((n,), dt, device) for n in lengths]
        txs = [totorch(x) for x in xs]
        got = Tensor.meshgrid(*xs, indexing='ij')
        expected = torch.meshgrid(*txs, indexing='ij')
        assert len(got) == len(expected)
        for g, e in zip(got, expected):
            _same(g, e, dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_linspace_matches_torch(device: str) -> None:
    for start, end, steps in ((0.0, 1.0, 5), (-10.0, 10.0, 5), (3.0, 10.0, 1), (0.0, 1.0, 2), (1.0, 0.0, 7), (0.5, 0.75, 33), (-1000.0, 1000.0, 129), (2.0, 2.0, 4)):
        r = Tensor.linspace(start, end, steps=steps, device=device)
        assert_close_mag_torch(r, torch.linspace(start, end, steps=steps), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
@pytest.mark.parametrize('shape', [(3, 4), (2, 3, 4), (6,)])
def test_masked_fill_matches_torch(device: str, dt: dtype.DType, shape) -> None:
    mask_shapes = [shape, (shape[-1],), (1,) * len(shape)]
    if len(shape) >= 2:
        mask_shapes.append(shape[:-1] + (1,))
    for ms in mask_shapes:
        for value in (0.5, -3.0, 7.0):
            v = int(value) if dt.is_integer() else value
            x = _rand(shape, dt, device)
            mask = Tensor.bernoulli(ms, p=0.5, device=device)
            tx, tm = totorch(x), totorch(mask)
            _same(x.masked_fill(mask, v), tx.masked_fill(tm, v), dt)
            _same(x, tx, dt)
            y = x.clone()
            y.masked_fill_(mask, v)
            _same(y, tx.masked_fill(tm, v), dt)


_PAD_CONSTANT = (((3, 4), (1, 2), 0.0), ((3, 4), (1, 2, 0, 1), -1.5), ((2, 3, 4), (2, 1, 1, 1, 0, 2), 0.0), ((2, 3, 4, 5), (1, 1, 2, 2), 3.0), ((5,), (2, 3), 0.0), ((2, 3, 4), (0, 0), 1.0), ((2, 3, 4), (3, 0, 0, 2), 2.5), ((3, 6), (-1, 2), 0.0), ((2, 3, 6), (-2, -1, 1, -1), 4.0), ((2, 3, 4, 5), (-1, 1, 0, -2, 1, 0), -1.0), ((5,), (-2, -2), 0.0))
_PAD_REFLECT_REPLICATE = (((2, 3, 5), (2, 1)), ((2, 3, 4, 5), (1, 2, 2, 1)), ((1, 2, 3, 4, 5), (1, 1, 1, 1, 1, 1)), ((2, 3, 6), (0, 3)), ((2, 3, 4, 5), (3, 3, 0, 0)))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_pad_constant_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape, pad, value in _PAD_CONSTANT:
        v = int(value) if dt.is_integer() else value
        x = _rand(shape, dt, device)
        _same(x.pad(list(pad), value=v), F.pad(totorch(x), list(pad), value=v), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('mode', ['reflect', 'replicate'])
def test_pad_reflect_replicate_matches_torch(device: str, mode: str) -> None:
    for shape, pad in _PAD_REFLECT_REPLICATE:
        x = _rand(shape, dtype.float32, device)
        _same(x.pad(list(pad), mode=mode), F.pad(totorch(x), list(pad), mode=mode), dtype.float32)


_WHERE_CASES = (((2, 3), (2, 3), (2, 3)), ((2, 3), (3,), (2, 1)), ((3, 1), (1, 4), (3, 4)), ((2, 1, 4), (3, 1), (1,)), ((4,), (2, 4), (2, 4)), ((), (2, 3), (2, 3)))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_where_matches_torch(device: str, dt: dtype.DType) -> None:
    for cs, xs, ys in _WHERE_CASES:
        cond = Tensor.bernoulli(cs, p=0.5, device=device)
        x = _rand(xs, dt, device)
        y = _rand(ys, dt, device)
        tc, tx, ty = totorch(cond), totorch(x), totorch(y)
        _same(Tensor.where(cond, x, y), torch.where(tc, tx, ty), dt)
        scalar = 3 if dt.is_integer() else 0.75
        _same(Tensor.where(cond, x, scalar), torch.where(tc, tx, torch.tensor(scalar, dtype=tx.dtype)), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16, dtype.int32, dtype.int8), ids=['float32', 'float16', 'int32', 'int8'])
@pytest.mark.parametrize('shape', [(7,), (3, 4), (2, 3, 4)])
def test_clamp_matches_torch(device: str, dt: dtype.DType, shape) -> None:
    x = _rand(shape, dt, device)
    tx = totorch(x)
    bounds = ((-3, 7), (0, 50), (-50, 0), (5, 5), (10, -10)) if dt.is_integer() else ((-0.5, 0.5), (0.0, 1.0), (-0.25, 0.0), (0.3, 0.3), (0.5, -0.5))
    for lo, hi in bounds:
        _same(x.clamp(lo, hi), torch.clamp(tx, lo, hi), dt)
        _same(x.clamp_min(lo), torch.clamp_min(tx, lo), dt)
        _same(x.clamp_max(hi), torch.clamp_max(tx, hi), dt)
    lo = uniform_tensor(shape, low=-1, high=0, dtype=dt, device=device)
    hi = uniform_tensor(shape, low=0, high=1, dtype=dt, device=device)
    _same(x.clamp(lo, hi), torch.clamp(tx, totorch(lo), totorch(hi)), dt)
    _same(x.clamp_min(lo), torch.clamp_min(tx, totorch(lo)), dt)
    _same(x.clamp_max(hi), torch.clamp_max(tx, totorch(hi)), dt)
    lo_row = uniform_tensor((shape[-1],), low=-1, high=0, dtype=dt, device=device)
    hi_row = uniform_tensor((shape[-1],), low=0, high=1, dtype=dt, device=device)
    _same(x.clamp(lo_row, hi_row), torch.clamp(tx, totorch(lo_row), totorch(hi_row)), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16, dtype.int32, dtype.uint8), ids=['float32', 'float16', 'int32', 'uint8'])
def test_binary_minmax_matches_torch(device: str, dt: dtype.DType) -> None:
    for xs, ys in (((7,), (7,)), ((3, 4), (3, 4)), ((2, 3, 4), (2, 3, 4)), ((3, 1), (1, 4)), ((2, 3, 4), (4,)), ((5,), (2, 5)), ((), (3,))):
        x = _rand(xs, dt, device)
        y = _rand(ys, dt, device)
        tx, ty = totorch(x), totorch(y)
        _same(x.min(y), torch.minimum(tx, ty), dt)
        _same(x.max(y), torch.maximum(tx, ty), dt)
        _same(y.min(x), torch.minimum(ty, tx), dt)
        _same(y.max(x), torch.maximum(ty, tx), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16, dtype.bfloat16), ids=['float32', 'float16', 'bfloat16'])
def test_embedding_matches_torch(device: str, dt: dtype.DType) -> None:
    for vocab, dim in ((6, 3), (10, 4), (1, 5)):
        w = _rand((vocab, dim), dt, device)
        tw = totorch(w)
        for idx_shape in ((4,), (2, 3), (2, 2, 2), (1,)):
            tidx = torch.randint(0, vocab, idx_shape)
            idx = Tensor(tidx.tolist(), device=device)
            r = w.embedding(idx)
            assert r.dtype == dt
            _same(r, F.embedding(tidx, tw), dt)
            tidx32 = tidx.to(torch.int32)
            _same(w.embedding(Tensor(tidx32.tolist(), dtype=dtype.int32, device=device)), F.embedding(tidx32, tw), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32_BOOL, ids=[d.name for d in _F32_I32_BOOL])
@pytest.mark.parametrize('name', ['tril', 'triu'])
def test_tril_triu_matches_torch(device: str, dt: dtype.DType, name: str) -> None:
    for shape in ((3, 5), (5, 3), (4, 4), (2, 3, 4), (2, 1, 3, 3), (1, 6)):
        x = _rand(shape, dt, device)
        tx = totorch(x)
        for diagonal in range(-4, 5):
            _same(getattr(x, name)(diagonal), getattr(torch, name)(tx, diagonal), dt)
            y = x.clone()
            getattr(y, f'{name}_')(diagonal)
            _same(y, getattr(torch, name)(tx, diagonal), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_stack_family_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape in ((3,), (2, 3), (2, 3, 4), ()):
        xs = [_rand(shape, dt, device) for _ in range(3)]
        txs = [totorch(x) for x in xs]
        for dim in range(-(len(shape) + 1), len(shape) + 1):
            _same(Tensor.stack(xs, dim), torch.stack(txs, dim), dt)
        _same(Tensor.stack(xs), torch.stack(txs), dt)
        if shape:
            _same(Tensor.hstack(xs), torch.hstack(txs), dt)
            _same(Tensor.vstack(xs), torch.vstack(txs), dt)
            _same(Tensor.dstack(xs), torch.dstack(txs), dt)
    a = _rand((2, 3), dt, device)
    b = _rand((2, 5), dt, device)
    _same(Tensor.hstack([a, b]), torch.hstack([totorch(a), totorch(b)]), dt)
    c = _rand((4, 3), dt, device)
    _same(Tensor.vstack([a, c]), torch.vstack([totorch(a), totorch(c)]), dt)
    d = _rand((2, 3, 2), dt, device)
    e = _rand((2, 3, 5), dt, device)
    _same(Tensor.dstack([d, e]), torch.dstack([totorch(d), totorch(e)]), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_cat_matches_torch(device: str, dt: dtype.DType) -> None:
    for base, dim in (((3, 4), 0), ((3, 4), 1), ((3, 4), -1), ((2, 3, 4), 1), ((2, 3, 4), -3), ((5,), 0)):
        xs = []
        for n in (1, 2, 3):
            s = list(base)
            s[dim] = n
            xs.append(_rand(tuple(s), dt, device))
        _same(Tensor.cat(xs, dim), torch.cat([totorch(x) for x in xs], dim), dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape', [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
def test_softmax_any_dim_matches_torch(device: str, shape) -> None:
    x = uniform_tensor(shape, -6.0, 6.0, dtype.float32, device)
    tx = totorch(x)
    for dim in range(-len(shape), len(shape)):
        assert_close_mag_torch(x.softmax(dim), torch.softmax(tx, dim), dtype.float32)
        y = x.clone()
        y.softmax_(dim)
        assert_close_mag_torch(y, torch.softmax(tx, dim), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name', ['max', 'min'])
def test_indexed_minmax_matches_torch(device: str, name: str) -> None:
    for shape in ((7,), (3, 4), (2, 3, 4)):
        x = uniform_tensor(shape, -3.0, 3.0, dtype.float32, device)
        tx = totorch(x)
        for dim in range(-len(shape), len(shape)):
            for keepdim in (False, True):
                values, indices = getattr(x, name)(dim, keepdim=keepdim)
                ref = getattr(tx, name)(dim, keepdim=keepdim)
                assert_close_mag_torch(values, ref.values, dtype.float32)
                assert indices.dtype == dtype.int64
                assert indices.tolist() == ref.indices.tolist()
    ties = torch.tensor([[1.0, 3.0, 3.0, 0.0], [2.0, 2.0, 2.0, 2.0], [-1.0, 5.0, -1.0, 5.0]])
    x = Tensor(ties.tolist(), device=device)
    for dim in (0, 1):
        values, indices = getattr(x, name)(dim)
        ref = getattr(ties, name)(dim)
        assert values.tolist() == ref.values.tolist()
        assert indices.tolist() == ref.indices.tolist()
    with pytest.raises(TypeError):
        x.max((0, 1))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_index_add_out_of_place_matches_torch(device: str, dt: dtype.DType) -> None:
    for shape, dim, index, src_shape, alpha in _INDEX_ADD_CASES:
        x = _rand(shape, dt, device)
        src = _rand(src_shape, dt, device)
        tx, tsrc = totorch(x), totorch(src)
        a = int(alpha) if dt.is_integer() else alpha
        r = x.index_add(dim, Tensor(index, device=device), src, alpha=a)
        _same(r, tx.index_add(dim, torch.tensor(index), tsrc, alpha=a), dt)
        _same(x, tx, dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_softmax_last_dim_matches_torch(device: str) -> None:
    for shape in ((5,), (3, 4), (2, 3, 4), (1, 1, 7), (4, 257)):
        x = uniform_tensor(shape, low=-6.0, high=6.0, device=device)
        tx = totorch(x)
        assert_close_mag_torch(x.softmax(), torch.softmax(tx, -1), dtype.float32)
        assert_close_mag_torch(x.softmax(-1), torch.softmax(tx, -1), dtype.float32)
        assert_close_mag_torch(x.softmax(x.rank - 1), torch.softmax(tx, -1), dtype.float32)
        y = x.clone()
        y.softmax_()
        assert_close_mag_torch(y, torch.softmax(tx, -1), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _F32_I32, ids=[d.name for d in _F32_I32])
def test_reshape_of_permuted_tensor_with_unit_dims_matches_torch(device: str, dt: dtype.DType) -> None:
    cases = (
        ((2, 1, 3, 4), (0, 1, 3, 2), (8, 3)),
        ((2, 1, 3, 4), (0, 1, 3, 2), (2, 4, 3)),
        ((2, 1, 3, 4), (0, 1, 3, 2), (2, 12)),
        ((3, 1, 1, 4, 5), (0, 1, 2, 4, 3), (15, 4)),
        ((3, 1, 4, 1, 5), (0, 3, 1, 4, 2), (3, 20)),
        ((2, 3, 1, 4), (1, 0, 2, 3), (3, 8)),
        ((1, 2, 3, 4), (0, 3, 1, 2), (4, 6)),
        ((2, 1, 3, 4), (2, 0, 1, 3), (6, 4)),
    )
    for shape, perm, new_shape in cases:
        x = _rand(shape, dt, device)
        tx = totorch(x)
        p = x.permute(perm)
        tp = tx.permute(*perm)
        _same(p.reshape(new_shape), tp.reshape(new_shape), dt)
        _same(p.contiguous().reshape(new_shape), tp.reshape(new_shape), dt)
        try:
            v = p.view(*new_shape)
        except RuntimeError:
            v = None
        if v is not None:
            _same(v, tp.reshape(new_shape), dt)
        w = Tensor.uniform((new_shape[-1], 3), dtype=dtype.float32, device=device)
        if dt == dtype.float32:
            assert_close_mag_torch(p.reshape(new_shape) @ w, tp.reshape(new_shape) @ totorch(w), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('op', ['cumax', 'cumin'])
def test_cummax_cummin_ties_match_torch(device: str, op: str) -> None:
    tname = 'cummax' if op == 'cumax' else 'cummin'
    for values in ([1, 3, 3, 2, 3], [3, 1, 1, 2, 1], [2, 2, 2, 2], [5], [1, 1, 5, 5, 0, 0, 5]):
        for dt in (dtype.int32, dtype.float32):
            x = Tensor(values, dtype=dt, device=device)
            tx = torch.tensor(values, dtype=totorch_dtype(dt))
            v, i = getattr(x, op)(0)
            ref = getattr(tx, tname)(0)
            assert v.tolist() == ref.values.tolist()
            assert i.tolist() == ref.indices.tolist()
    x = uniform_tensor((3, 4, 5), 0, 3, dtype.int32, device)
    tx = totorch(x)
    for dim in range(-3, 3):
        v, i = getattr(x, op)(dim)
        ref = getattr(tx, tname)(dim)
        assert v.tolist() == ref.values.tolist()
        assert i.tolist() == ref.indices.tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16, dtype.bfloat16), ids=['float32', 'float16', 'bfloat16'])
def test_binary_minmax_propagate_nan_like_torch(device: str, dt: dtype.DType) -> None:
    nan = float('nan')
    a = Tensor([nan, 1.0, 2.0, nan, -1.0], dtype=dt, device=device)
    b = Tensor([0.0, nan, 3.0, nan, -2.0], dtype=dt, device=device)
    ta, tb = totorch(a), totorch(b)
    torch.testing.assert_close(totorch(a.min(b)), torch.minimum(ta, tb), equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(totorch(a.max(b)), torch.maximum(ta, tb), equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(totorch(a.clamp_min(b)), torch.clamp_min(ta, tb), equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_meshgrid_xy_matches_torch(device: str) -> None:
    for lengths in ((3, 2), (2, 3, 4), (4, 1), (5,)):
        xs = [uniform_tensor((n,), -3.0, 3.0, dtype.float32, device) for n in lengths]
        txs = [totorch(x) for x in xs]
        got = Tensor.meshgrid(*xs, indexing='xy')
        expected = torch.meshgrid(*txs, indexing='xy')
        assert len(got) == len(expected)
        for g, e in zip(got, expected):
            assert g.shape == tuple(e.shape)
            assert g.tolist() == e.tolist()
    with pytest.raises(ValueError):
        Tensor.meshgrid(xs[0], indexing='zz')


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('op', ['sum', 'prod'])
def test_bool_sum_prod_matches_torch(device: str, op: str) -> None:
    for shape in ((7,), (3, 4), (2, 3, 4)):
        x = Tensor((torch.rand(shape) < 0.6).tolist(), dtype=dtype.boolean, device=device)
        tx = totorch(x)
        r = getattr(x, op)()
        t = getattr(tx, op)()
        assert r.dtype == dtype.int64
        assert r.tolist() == t.tolist()
        for dim in range(-len(shape), len(shape)):
            for keepdim in (False, True):
                r = getattr(x, op)(dim=dim, keepdim=keepdim)
                t = getattr(tx, op)(dim=dim, keepdim=keepdim)
                assert r.shape == tuple(t.shape)
                assert r.tolist() == t.tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_gather_negative_dim_matches_torch(device: str) -> None:
    x = uniform_tensor((2, 3, 4), -3.0, 3.0, dtype.float32, device)
    tx = totorch(x)
    for dim in (-1, -2, -3):
        tidx = torch.randint(0, tx.shape[dim], (2, 3, 4))
        _same(x.gather(dim, Tensor(tidx.tolist(), device=device)), tx.gather(dim, tidx), dtype.float32)


def test_numpy_scalar_constructor_matches_torch() -> None:
    for value, expected in ((np.float32(1.5), dtype.float32), (np.float16(-2.0), dtype.float16), (np.float64(0.25), dtype.float32), (np.bool_(True), dtype.boolean), (np.int64(7), dtype.int64), (np.int32(-3), dtype.int32), (np.uint8(200), dtype.uint8)):
        t = Tensor(value)
        assert t.shape == ()
        assert t.dtype == expected
        assert t.item() == value.item()
    t = Tensor(np.float32(1.5), dtype=dtype.int32)
    assert t.dtype == dtype.int32 and t.item() == 1


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', (dtype.float32, dtype.float16), ids=['float32', 'float16'])
def test_inplace_float_floordiv_matches_torch(device: str, dt: dtype.DType) -> None:
    for shapes in (((3, 4), (3, 4)), ((3, 4), (4,)), ((2, 3, 4), (3, 1))):
        x = uniform_tensor(shapes[0], -5.0, 5.0, dt, device)
        y = uniform_tensor(shapes[1], 0.5, 2.0, dt, device)
        tx, ty = totorch(x), totorch(y)
        x //= y
        expected = torch_floordiv(tx, ty)
        assert_close_mag_torch(x, expected, dt)
