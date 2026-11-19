# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch

import magnetron as mag
from magnetron.utils import dsp

from .common import *

_WINDOW_SIZES = [1, 2, 3, 8, 15, 64, 400]
_FFT_SIZES = [1, 2, 3, 4, 7, 8, 16, 33, 64]

# Matmul-DFT accumulates O(N) float32 rounding, so loosen tolerances vs elementwise ops
_FFT_RTOL = 1e-4
_FFT_ATOL = 1e-3


def _window_tol() -> dict[str, float]:
    return dict(rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('n', _WINDOW_SIZES)
@pytest.mark.parametrize(
    'mag_fn,torch_fn',
    [
        (dsp.hann_window, torch.hann_window),
        (dsp.hamming_window, torch.hamming_window),
        (dsp.blackman_window, torch.blackman_window),
    ],
    ids=['hann', 'hamming', 'blackman'],
)
def test_cosine_windows(device: str, periodic: bool, n: int, mag_fn, torch_fn) -> None:
    with mag.device(device):
        w = mag_fn(n, periodic, dtype=dtype.float32)
    tw = torch_fn(n, periodic=periodic, dtype=torch.float64).float()
    assert w.shape == tuple(tw.shape)
    assert_close_mag_torch(w, tw, dtype.float32, **_window_tol())


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('beta', [0.0, 5.0, 8.6, 12.0])
@pytest.mark.parametrize('n', [0, *_WINDOW_SIZES])
def test_kaiser_window(device: str, periodic: bool, beta: float, n: int) -> None:
    with mag.device(device):
        w = dsp.kaiser_window(n, beta=beta, periodic=periodic, dtype=dtype.float32)
    tw = torch.kaiser_window(n, periodic=periodic, beta=beta, dtype=torch.float64).float()
    assert w.shape == tuple(tw.shape)
    if n > 0:
        assert_close_mag_torch(w, tw, dtype.float32, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_i0(device: str) -> None:
    with mag.device(device):
        x = uniform_tensor((256,), 0.0, 12.0, dtype.float32, device)
        y = dsp.i0(x)
    ty = torch.special.i0(totorch(x).double()).float()
    assert_close_mag_torch(y, ty, dtype.float32, rtol=1e-4, atol=1e-5)


def _rand(shape: tuple[int, ...], device: str) -> Tensor:
    return random_tensor(shape, dtype.float32, device)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('n', _FFT_SIZES)
@pytest.mark.parametrize('complex_input', [False, True])
def test_fft_1d(device: str, n: int, complex_input: bool) -> None:
    with mag.device(device):
        xr = _rand((n,), device)
        xi = _rand((n,), device) if complex_input else None
        yr, yi = dsp.fft(xr, xi)
    tx = torch.complex(totorch(xr).double(), totorch(xi).double() if xi is not None else torch.zeros(n, dtype=torch.float64))
    ty = torch.fft.fft(tx)
    assert_close_mag_torch(yr, ty.real.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(yi, ty.imag.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('n', _FFT_SIZES)
def test_ifft_1d(device: str, n: int) -> None:
    with mag.device(device):
        xr = _rand((n,), device)
        xi = _rand((n,), device)
        yr, yi = dsp.ifft(xr, xi)
    ty = torch.fft.ifft(torch.complex(totorch(xr).double(), totorch(xi).double()))
    assert_close_mag_torch(yr, ty.real.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(yi, ty.imag.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape,dim', [((3, 8), -1), ((3, 8), 0), ((2, 5, 6), 1), ((2, 5, 6), -1), ((4, 3, 2, 7), 0)])
def test_fft_batched_dim(device: str, shape: tuple[int, ...], dim: int) -> None:
    with mag.device(device):
        xr = _rand(shape, device)
        xi = _rand(shape, device)
        yr, yi = dsp.fft(xr, xi, dim=dim)
        zr, zi = dsp.ifft(xr, xi, dim=dim)
    tx = torch.complex(totorch(xr).double(), totorch(xi).double())
    ty = torch.fft.fft(tx, dim=dim)
    tz = torch.fft.ifft(tx, dim=dim)
    assert yr.shape == tuple(ty.shape) and yi.shape == tuple(ty.shape)
    assert_close_mag_torch(yr, ty.real.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(yi, ty.imag.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(zr, tz.real.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(zi, tz.imag.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('n', _FFT_SIZES)
def test_fft_ifft_roundtrip(device: str, n: int) -> None:
    with mag.device(device):
        xr = _rand((2, n), device)
        xi = _rand((2, n), device)
        yr, yi = dsp.ifft(*dsp.fft(xr, xi))
    assert_close_mag_torch(yr, totorch(xr), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(yi, totorch(xi), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


def test_ifft_requires_imag() -> None:
    with pytest.raises(RuntimeError):
        dsp.ifft(Tensor.zeros(4))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('n', _FFT_SIZES)
def test_rfft_1d(device: str, n: int) -> None:
    with mag.device(device):
        x = _rand((n,), device)
        yr, yi = dsp.rfft(x)
    ty = torch.fft.rfft(totorch(x).double())
    assert yr.shape == tuple(ty.shape) and yi.shape == tuple(ty.shape)
    assert_close_mag_torch(yr, ty.real.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(yi, ty.imag.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('n', [n for n in _FFT_SIZES if n > 1])
def test_irfft_1d(device: str, n: int) -> None:
    k = n // 2 + 1
    tx = torch.fft.rfft(torch.randn(n, dtype=torch.float64))  # valid Hermitian half-spectrum
    with mag.device(device):
        xr = Tensor(tx.real.float().tolist(), dtype=dtype.float32, device=device)
        xi = Tensor(tx.imag.float().tolist(), dtype=dtype.float32, device=device)
        assert xr.shape == (k,)
        y = dsp.irfft(xr, xi, n)
    ty = torch.fft.irfft(torch.complex(totorch(xr).double(), totorch(xi).double()), n=n)
    assert y.shape == tuple(ty.shape)
    assert_close_mag_torch(y, ty.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape,dim', [((3, 8), -1), ((8, 3), 0), ((2, 9, 4), 1), ((2, 4, 10), -1)])
def test_rfft_irfft_batched_dim(device: str, shape: tuple[int, ...], dim: int) -> None:
    n = shape[dim]
    with mag.device(device):
        x = _rand(shape, device)
        yr, yi = dsp.rfft(x, dim=dim)
        z = dsp.irfft(yr, yi, n, dim=dim)
    tx = totorch(x).double()
    ty = torch.fft.rfft(tx, dim=dim)
    assert yr.shape == tuple(ty.shape)
    assert_close_mag_torch(yr, ty.real.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(yi, ty.imag.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert z.shape == tuple(shape)
    assert_close_mag_torch(z, tx.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
