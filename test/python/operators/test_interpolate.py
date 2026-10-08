# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch
import torch.nn.functional as F

import magnetron as mag
import magnetron.nn as nn

from ..common import *

_NEAREST_CASES = [
    # (in_shape, size, scale_factor)
    ((2, 3, 5), (10,), None),
    ((1, 2, 7), None, 3),
    ((2, 2, 8), (5,), None),
    ((2, 3, 4, 5), (8, 10), None),
    ((1, 4, 6, 6), None, 2),
    ((2, 1, 5, 3), None, (1.5, 3)),
    ((1, 2, 9, 7), (4, 5), None),
    ((1, 2, 3, 4, 5), (6, 8, 10), None),
    ((2, 2, 2, 3, 2), None, 2),
    ((1, 1, 4, 3, 5), (2, 6, 5), None),
    ((1, 3, 5, 5), None, 1.1),
    ((1, 1, 7, 7), None, 0.6),
]

_LINEAR_CASES = [
    # (mode, in_shape, size, scale_factor)
    ('linear', (2, 3, 5), (11,), None),
    ('linear', (1, 2, 8), None, 2.5),
    ('linear', (2, 2, 9), (4,), None),
    ('linear', (1, 1, 6), (6,), None),
    ('bilinear', (2, 3, 4, 5), (9, 11), None),
    ('bilinear', (1, 2, 6, 6), None, 2),
    ('bilinear', (1, 2, 9, 7), (4, 3), None),
    ('bilinear', (2, 1, 5, 5), None, (0.5, 1.5)),
    ('bicubic', (2, 3, 4, 5), (9, 11), None),
    ('bicubic', (1, 2, 6, 6), None, 2),
    ('bicubic', (1, 2, 9, 7), (4, 3), None),
    ('bicubic', (2, 1, 5, 5), None, (0.5, 1.5)),
    ('trilinear', (1, 2, 3, 4, 5), (6, 8, 10), None),
    ('trilinear', (2, 2, 4, 3, 5), None, 1.5),
    ('trilinear', (1, 1, 6, 5, 4), (3, 2, 3), None),
]

_AREA_CASES = [
    ((2, 3, 5), (10,), None),
    ((1, 2, 12), (5,), None),
    ((2, 2, 7), None, 0.5),
    ((2, 3, 8, 6), (4, 3), None),
    ((1, 2, 5, 7), (9, 4), None),
    ((1, 1, 6, 5, 4), (3, 2, 3), None),
    ((1, 2, 3, 4, 5), None, 2),
]

_AA_CASES = [
    # (mode, in_shape, size)
    ('bilinear', (2, 3, 12, 10), (5, 4)),
    ('bilinear', (1, 2, 7, 9), (3, 3)),
    ('bilinear', (1, 2, 4, 5), (9, 11)),
    ('bicubic', (2, 3, 12, 10), (5, 4)),
    ('bicubic', (1, 2, 7, 9), (3, 3)),
    ('bicubic', (1, 2, 4, 5), (9, 11)),
]


def _tol(dt: dtype.DType) -> tuple[float, float]:
    rtol, atol = compare_tol(dt)
    return rtol * 4, atol * 4


def _check_backward(device: str, shape, mode, size, scale, align_corners=False, antialias=False, mag_kwargs=None) -> None:
    x = random_tensor(shape, dtype.float32, device)
    tx = totorch(x).clone()
    x.requires_grad = True
    tx.requires_grad = True
    kwargs = dict(size=size, scale_factor=scale, mode=mode)
    tkwargs = dict(kwargs)
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        kwargs['align_corners'] = align_corners
        tkwargs['align_corners'] = align_corners
    if antialias:
        kwargs['antialias'] = True
        tkwargs['antialias'] = True
    y = x.interpolate(**kwargs)
    w = random_tensor(y.shape, dtype.float32, device)
    (y * w).sum().backward()
    ty = F.interpolate(tx, **tkwargs)
    (ty * totorch(w)).sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('mode', ['nearest', 'nearest-exact'])
@pytest.mark.parametrize('dt', [dtype.float32, dtype.float16, dtype.bfloat16, dtype.int32, dtype.uint8, dtype.int64])
@pytest.mark.parametrize('case', _NEAREST_CASES)
def test_interpolate_nearest_forward(device: str, mode: str, dt: dtype.DType, case) -> None:
    shape, size, scale = case
    x = random_tensor(shape, dt, device)
    r = x.interpolate(size=size, scale_factor=scale, mode=mode)
    tx = totorch(x)
    ref_in = tx.float() if dt.is_integer() else tx
    t = F.interpolate(ref_in, size=size, scale_factor=scale, mode=mode)
    assert r.shape == tuple(t.shape)
    assert r.dtype == dt
    assert totorch(r).float().tolist() == t.float().tolist()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('mode', ['nearest', 'nearest-exact'])
@pytest.mark.parametrize('case', _NEAREST_CASES)
def test_interpolate_nearest_backward(device: str, mode: str, case) -> None:
    shape, size, scale = case
    _check_backward(device, shape, mode, size, scale)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', [dtype.float32, dtype.float16, dtype.bfloat16])
@pytest.mark.parametrize('align_corners', [False, True])
@pytest.mark.parametrize('case', _LINEAR_CASES)
def test_interpolate_linear_forward(device: str, dt: dtype.DType, align_corners: bool, case) -> None:
    mode, shape, size, scale = case
    x = random_tensor(shape, dt, device)
    r = x.interpolate(size=size, scale_factor=scale, mode=mode, align_corners=align_corners)
    t = F.interpolate(totorch(x).float(), size=size, scale_factor=scale, mode=mode, align_corners=align_corners)
    assert r.shape == tuple(t.shape)
    assert r.dtype == dt
    rtol, atol = _tol(dt)
    assert_close_mag_torch(totorch(r).float(), t, dtype.float32, rtol=rtol, atol=atol)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('align_corners', [False, True])
@pytest.mark.parametrize('case', _LINEAR_CASES)
def test_interpolate_linear_backward(device: str, align_corners: bool, case) -> None:
    mode, shape, size, scale = case
    _check_backward(device, shape, mode, size, scale, align_corners=align_corners)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', [dtype.float32, dtype.float16, dtype.bfloat16])
@pytest.mark.parametrize('case', _AREA_CASES)
def test_interpolate_area_forward(device: str, dt: dtype.DType, case) -> None:
    shape, size, scale = case
    x = random_tensor(shape, dt, device)
    r = x.interpolate(size=size, scale_factor=scale, mode='area')
    t = F.interpolate(totorch(x).float(), size=size, scale_factor=scale, mode='area')
    assert r.shape == tuple(t.shape)
    rtol, atol = _tol(dt)
    assert_close_mag_torch(totorch(r).float(), t, dtype.float32, rtol=rtol, atol=atol)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('case', _AREA_CASES)
def test_interpolate_area_backward(device: str, case) -> None:
    shape, size, scale = case
    _check_backward(device, shape, 'area', size, scale)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('align_corners', [False, True])
@pytest.mark.parametrize('case', _AA_CASES)
def test_interpolate_antialias(device: str, align_corners: bool, case) -> None:
    mode, shape, size = case
    x = random_tensor(shape, dtype.float32, device)
    r = x.interpolate(size=size, mode=mode, align_corners=align_corners, antialias=True)
    t = F.interpolate(totorch(x), size=size, mode=mode, align_corners=align_corners, antialias=True)
    assert r.shape == tuple(t.shape)
    assert_close_mag_torch(r, t, dtype.float32, rtol=1e-4, atol=1e-4)
    _check_backward(device, shape, mode, size, None, align_corners=align_corners, antialias=True)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_interpolate_non_contiguous_input(device: str) -> None:
    x = random_tensor((2, 5, 4, 3), dtype.float32, device).transpose(2, 3)
    r = x.interpolate(scale_factor=2)
    t = F.interpolate(totorch(x), scale_factor=2, mode='nearest')
    assert r.shape == tuple(t.shape)
    assert totorch(r).tolist() == t.tolist()
    r = x.interpolate(scale_factor=2, mode='bilinear')
    t = F.interpolate(totorch(x), scale_factor=2, mode='bilinear')
    assert_close_mag_torch(r, t, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_interpolate_invalid_args(device: str) -> None:
    x = random_tensor((1, 2, 4, 4), dtype.float32, device)
    with pytest.raises(ValueError):
        x.interpolate()
    with pytest.raises(ValueError):
        x.interpolate(size=(8, 8), scale_factor=2)
    with pytest.raises(ValueError):
        x.interpolate(size=(8, 8, 8))
    with pytest.raises(ValueError):
        x.interpolate(scale_factor=0)
    with pytest.raises(ValueError):
        random_tensor((4, 4), dtype.float32, device).interpolate(scale_factor=2)
    with pytest.raises(RuntimeError):
        x.interpolate(size=(0, 8))
    with pytest.raises(ValueError):
        x.interpolate(size=(8, 8), mode='cubic')
    with pytest.raises(RuntimeError):
        x.interpolate(size=(8, 8), mode='linear')
    with pytest.raises(RuntimeError):
        x.interpolate(size=(8, 8), mode='trilinear')
    with pytest.raises(RuntimeError):
        x.interpolate(size=(8, 8), mode='nearest', align_corners=True)
    with pytest.raises(RuntimeError):
        x.interpolate(size=(8, 8), mode='area', align_corners=True)
    with pytest.raises(RuntimeError):
        x.interpolate(size=(8, 8), mode='nearest', antialias=True)
    with pytest.raises(RuntimeError):
        random_tensor((1, 2, 4), dtype.float32, device).interpolate(size=(8,), mode='linear', antialias=True)
    with pytest.raises(RuntimeError):
        random_tensor((1, 2, 4, 4), dtype.int32, device).interpolate(size=(8, 8), mode='bilinear')


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_upsample_module(device: str) -> None:
    with mag.device(device):
        x = random_tensor((2, 3, 5, 6), dtype.float32, device)
        assert totorch(nn.Upsample(scale_factor=2)(x)).tolist() == F.interpolate(totorch(x), scale_factor=2, mode='nearest').tolist()
        assert totorch(nn.Upsample(size=(7, 9))(x)).tolist() == F.interpolate(totorch(x), size=(7, 9), mode='nearest').tolist()
        y = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
        assert_close_mag_torch(y, F.interpolate(totorch(x), scale_factor=2, mode='bilinear', align_corners=True), dtype.float32)
    with pytest.raises(ValueError):
        nn.Upsample()
    with pytest.raises(ValueError):
        nn.Upsample(size=4, scale_factor=2)
