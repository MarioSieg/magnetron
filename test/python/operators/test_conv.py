# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch
import torch.nn.functional as F

import magnetron as mag
import magnetron.nn as nn

from ..common import *

_CONV_CASES = [
    # (spatial, N, cin, cout, in_size, k, stride, padding, dilation, groups)
    (1, 2, 3, 4, (9,), (3,), (1,), (0,), (1,), 1),
    (1, 1, 4, 6, (11,), (2,), (2,), (1,), (1,), 2),
    (1, 2, 2, 2, (8,), (3,), (1,), (2,), (2,), 1),
    (2, 2, 3, 4, (7, 6), (3, 3), (1, 1), (1, 1), (1, 1), 1),
    (2, 1, 4, 4, (8, 9), (3, 2), (2, 1), (0, 1), (1, 2), 4),
    (2, 2, 2, 6, (5, 5), (2, 2), (2, 2), (1, 0), (1, 1), 2),
    (3, 1, 2, 3, (4, 5, 6), (2, 3, 2), (1, 1, 1), (1, 0, 1), (1, 1, 1), 1),
    (3, 2, 4, 2, (5, 4, 5), (2, 2, 2), (2, 1, 2), (0, 1, 0), (1, 2, 1), 2),
]

_CONVT_CASES = [
    # (spatial, N, cin, cout, in_size, k, stride, padding, output_padding, dilation, groups)
    (1, 2, 3, 4, (7,), (3,), (1,), (0,), (0,), (1,), 1),
    (1, 1, 4, 6, (6,), (3,), (2,), (1,), (1,), (1,), 2),
    (1, 2, 2, 2, (5,), (2,), (3,), (1,), (2,), (2,), 1),
    (2, 2, 3, 4, (5, 4), (3, 3), (1, 1), (1, 1), (0, 0), (1, 1), 1),
    (2, 1, 4, 4, (4, 5), (3, 2), (2, 1), (0, 1), (1, 0), (1, 2), 4),
    (2, 2, 2, 6, (3, 3), (2, 2), (2, 2), (0, 0), (1, 1), (1, 1), 2),
    (3, 1, 2, 3, (3, 4, 3), (2, 3, 2), (1, 1, 1), (1, 0, 1), (0, 0, 0), (1, 1, 1), 1),
    (3, 2, 4, 2, (3, 3, 4), (2, 2, 2), (2, 1, 2), (0, 1, 0), (1, 0, 1), (1, 2, 1), 2),
]


def _torch_conv(spatial: int, T: bool):
    return {
        (1, False): F.conv1d,
        (2, False): F.conv2d,
        (3, False): F.conv3d,
        (1, True): F.conv_transpose1d,
        (2, True): F.conv_transpose2d,
        (3, True): F.conv_transpose3d,
    }[(spatial, T)]


def _mag_conv(spatial: int, T: bool):
    def call(x, *args, **kwargs):
        return getattr(x, f'convT{spatial}D' if T else f'conv{spatial}D')(*args, **kwargs)

    return call


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', [dtype.float32, dtype.float16, dtype.bfloat16])
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('case', _CONV_CASES)
def test_conv_forward(device: str, dt: dtype.DType, use_bias: bool, case) -> None:
    spatial, n, cin, cout, in_size, k, stride, padding, dilation, groups = case
    x = random_tensor((n, cin, *in_size), dt, device)
    w = random_tensor((cout, cin // groups, *k), dt, device)
    b = random_tensor((cout,), dt, device) if use_bias else None
    r = _mag_conv(spatial, False)(x, w, b, stride, padding, dilation, groups)
    tx, tw = totorch(x).float(), totorch(w).float()
    tb = totorch(b).float() if use_bias else None
    t = _torch_conv(spatial, False)(tx, tw, tb, stride, padding, dilation, groups)
    assert r.shape == tuple(t.shape)
    tol = compare_tol(dt)
    assert_close_mag_torch(totorch(r).float(), t, dtype.float32, rtol=tol[0] * 4, atol=tol[1] * 4)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', [dtype.float32, dtype.float16, dtype.bfloat16])
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('case', _CONVT_CASES)
def test_conv_transpose_forward(device: str, dt: dtype.DType, use_bias: bool, case) -> None:
    spatial, n, cin, cout, in_size, k, stride, padding, output_padding, dilation, groups = case
    x = random_tensor((n, cin, *in_size), dt, device)
    w = random_tensor((cin, cout // groups, *k), dt, device)
    b = random_tensor((cout,), dt, device) if use_bias else None
    r = _mag_conv(spatial, True)(x, w, b, stride, padding, output_padding, groups, dilation)
    tx, tw = totorch(x).float(), totorch(w).float()
    tb = totorch(b).float() if use_bias else None
    t = _torch_conv(spatial, True)(tx, tw, tb, stride, padding, output_padding, groups, dilation)
    assert r.shape == tuple(t.shape)
    tol = compare_tol(dt)
    assert_close_mag_torch(totorch(r).float(), t, dtype.float32, rtol=tol[0] * 4, atol=tol[1] * 4)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('case', _CONV_CASES)
def test_conv_backward(device: str, case) -> None:
    spatial, n, cin, cout, in_size, k, stride, padding, dilation, groups = case
    x = random_tensor((n, cin, *in_size), dtype.float32, device)
    w = random_tensor((cout, cin // groups, *k), dtype.float32, device)
    b = random_tensor((cout,), dtype.float32, device)
    tx, tw, tb = totorch(x).clone(), totorch(w).clone(), totorch(b).clone()
    x.requires_grad = True
    w.requires_grad = True
    b.requires_grad = True
    tx.requires_grad = True
    tw.requires_grad = True
    tb.requires_grad = True
    y = _mag_conv(spatial, False)(x, w, b, stride, padding, dilation, groups)
    scale = random_tensor(y.shape, dtype.float32, device)
    (y * scale).sum().backward()
    ty = _torch_conv(spatial, False)(tx, tw, tb, stride, padding, dilation, groups)
    (ty * totorch(scale)).sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)
    assert_close_mag_torch(w.grad, tw.grad, dtype.float32)
    assert_close_mag_torch(b.grad, tb.grad, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('case', _CONVT_CASES)
def test_conv_transpose_backward(device: str, case) -> None:
    spatial, n, cin, cout, in_size, k, stride, padding, output_padding, dilation, groups = case
    x = random_tensor((n, cin, *in_size), dtype.float32, device)
    w = random_tensor((cin, cout // groups, *k), dtype.float32, device)
    b = random_tensor((cout,), dtype.float32, device)
    tx, tw, tb = totorch(x).clone(), totorch(w).clone(), totorch(b).clone()
    x.requires_grad = True
    w.requires_grad = True
    b.requires_grad = True
    tx.requires_grad = True
    tw.requires_grad = True
    tb.requires_grad = True
    y = _mag_conv(spatial, True)(x, w, b, stride, padding, output_padding, groups, dilation)
    scale = random_tensor(y.shape, dtype.float32, device)
    (y * scale).sum().backward()
    ty = _torch_conv(spatial, True)(tx, tw, tb, stride, padding, output_padding, groups, dilation)
    (ty * totorch(scale)).sum().backward()
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)
    assert_close_mag_torch(w.grad, tw.grad, dtype.float32)
    assert_close_mag_torch(b.grad, tb.grad, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_conv_non_contiguous_inputs(device: str) -> None:
    x = random_tensor((2, 6, 5, 4), dtype.float32, device).transpose(2, 3)
    w = random_tensor((3, 3, 6, 4), dtype.float32, device).transpose(1, 2)
    r = x.conv2D(w, padding=1)
    t = F.conv2d(totorch(x), totorch(w), padding=1)
    assert_close_mag_torch(r, t, dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_conv_invalid_args(device: str) -> None:
    x = random_tensor((1, 4, 8), dtype.float32, device)
    with pytest.raises(RuntimeError):
        x.conv1D(random_tensor((2, 3, 3), dtype.float32, device))
    with pytest.raises(RuntimeError):
        x.conv1D(random_tensor((2, 4, 3), dtype.float32, device), stride=0)
    with pytest.raises(RuntimeError):
        x.conv1D(random_tensor((2, 4, 9), dtype.float32, device))
    with pytest.raises(RuntimeError):
        x.conv1D(random_tensor((2, 2, 3), dtype.float32, device), groups=3)
    with pytest.raises(RuntimeError):
        x.convT1D(random_tensor((4, 2, 3), dtype.float32, device), stride=2, output_padding=2)
    with pytest.raises(RuntimeError):
        x.conv1D(random_tensor((2, 4, 3), dtype.float32, device), bias=random_tensor((3,), dtype.float32, device))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('spatial', [1, 2, 3])
@pytest.mark.parametrize('transposed', [False, True])
def test_conv_module(device: str, spatial: int, transposed: bool) -> None:
    cls = getattr(nn, f'ConvT{spatial}D' if transposed else f'Conv{spatial}D')
    with mag.device(device):
        layer = cls(4, 6, kernel_size=3, stride=2, padding=1, groups=2)
        x = random_tensor((2, 4, *([7] * spatial)), dtype.float32, device)
        y = layer(x)
    params = dict(layer.named_parameters())
    assert set(params.keys()) == {'weight', 'bias'}
    if transposed:
        assert layer.weight.shape == (4, 3, *([3] * spatial))
        ref = _torch_conv(spatial, True)(totorch(x), totorch(layer.weight), totorch(layer.bias), 2, 1, 0, 2, 1)
    else:
        assert layer.weight.shape == (6, 2, *([3] * spatial))
        ref = _torch_conv(spatial, False)(totorch(x), totorch(layer.weight), totorch(layer.bias), 2, 1, 1, 2)
    assert_close_mag_torch(y, ref, dtype.float32)
    layer_no_bias = cls(4, 6, kernel_size=3, bias=False)
    assert layer_no_bias.bias is None
