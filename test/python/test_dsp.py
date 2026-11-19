# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import math

import torch

import magnetron as mag

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
        (Tensor.hann_window, torch.hann_window),
        (Tensor.hamming_window, torch.hamming_window),
        (Tensor.blackman_window, torch.blackman_window),
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
        w = Tensor.kaiser_window(n, beta=beta, periodic=periodic, dtype=dtype.float32)
    tw = torch.kaiser_window(n, periodic=periodic, beta=beta, dtype=torch.float64).float()
    assert w.shape == tuple(tw.shape)
    if n > 0:
        assert_close_mag_torch(w, tw, dtype.float32, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_i0(device: str) -> None:
    with mag.device(device):
        x = uniform_tensor((256,), 0.0, 12.0, dtype.float32, device)
        y = x.i0()
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
        yr, yi = xr.fft(xi)
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
        yr, yi = xr.ifft(xi)
    ty = torch.fft.ifft(torch.complex(totorch(xr).double(), totorch(xi).double()))
    assert_close_mag_torch(yr, ty.real.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(yi, ty.imag.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape,dim', [((3, 8), -1), ((3, 8), 0), ((2, 5, 6), 1), ((2, 5, 6), -1), ((4, 3, 2, 7), 0)])
def test_fft_batched_dim(device: str, shape: tuple[int, ...], dim: int) -> None:
    with mag.device(device):
        xr = _rand(shape, device)
        xi = _rand(shape, device)
        yr, yi = xr.fft(xi, dim=dim)
        zr, zi = xr.ifft(xi, dim=dim)
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
        fr, fi = xr.fft(xi)
        yr, yi = fr.ifft(fi)
    assert_close_mag_torch(yr, totorch(xr), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(yi, totorch(xi), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


def test_ifft_requires_imag() -> None:
    with pytest.raises(RuntimeError):
        Tensor.zeros(4).ifft()


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('n', _FFT_SIZES)
def test_rfft_1d(device: str, n: int) -> None:
    with mag.device(device):
        x = _rand((n,), device)
        yr, yi = x.rfft()
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
        y = xr.irfft(xi, n)
    ty = torch.fft.irfft(torch.complex(totorch(xr).double(), totorch(xi).double()), n=n)
    assert y.shape == tuple(ty.shape)
    assert_close_mag_torch(y, ty.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape,dim', [((3, 8), -1), ((8, 3), 0), ((2, 9, 4), 1), ((2, 4, 10), -1)])
def test_rfft_irfft_batched_dim(device: str, shape: tuple[int, ...], dim: int) -> None:
    n = shape[dim]
    with mag.device(device):
        x = _rand(shape, device)
        yr, yi = x.rfft(dim=dim)
        z = yr.irfft(yi, n, dim=dim)
    tx = totorch(x).double()
    ty = torch.fft.rfft(tx, dim=dim)
    assert yr.shape == tuple(ty.shape)
    assert_close_mag_torch(yr, ty.real.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert_close_mag_torch(yi, ty.imag.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)
    assert z.shape == tuple(shape)
    assert_close_mag_torch(z, tx.float(), dtype.float32, rtol=_FFT_RTOL, atol=_FFT_ATOL)


def _torch_resample_ref(x: torch.Tensor, orig_freq: int, new_freq: int, lowpass_filter_width: int, rolloff: float, beta: float | None) -> torch.Tensor:
    """float64 port of torchaudio.functional.resample (sinc_interp_hann / sinc_interp_kaiser)."""
    g = math.gcd(orig_freq, new_freq)
    orig, new = orig_freq // g, new_freq // g
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    idx = torch.arange(-width, width + orig, dtype=torch.float64)[None, None] / orig
    t = torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + idx
    t = (t * base).clamp(-lowpass_filter_width, lowpass_filter_width)
    if beta is None:
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        bt = torch.tensor(float(beta), dtype=torch.float64)
        window = torch.i0(bt * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(bt)
    t = t * math.pi
    kernel = torch.where(t == 0, torch.ones_like(t), t.sin() / t) * window * (base / orig)
    shape = x.shape
    xb = x.reshape(-1, shape[-1])
    length = xb.shape[-1]
    y = torch.nn.functional.conv1d(torch.nn.functional.pad(xb, (width, width + orig))[:, None], kernel, stride=orig)
    y = y.transpose(1, 2).reshape(xb.shape[0], -1)[..., : math.ceil(new * length / orig)]
    return y.reshape(*shape[:-1], y.shape[-1])


_RESAMPLE_RATES = [(48000, 16000), (44100, 16000), (22050, 16000), (8000, 16000), (16000, 44100), (32000, 16000)]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('rates', _RESAMPLE_RATES)
@pytest.mark.parametrize('length', [1, 100, 4410, 9601])
def test_resample_vs_torchaudio_kaiser_best(device: str, rates: tuple[int, int], length: int) -> None:
    orig, new = rates
    with mag.device(device):
        x = _rand((length,), device)
        y = x.resample(orig, new)
    ty = _torch_resample_ref(totorch(x).double(), orig, new, 64, 0.9475937167399596, 14.769656459379492)
    assert y.shape == tuple(ty.shape)
    assert_close_mag_torch(y, ty.float(), dtype.float32, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('rates', [(48000, 16000), (44100, 16000), (16000, 22050)])
def test_resample_vs_torchaudio_hann_default(device: str, rates: tuple[int, int]) -> None:
    orig, new = rates
    with mag.device(device):
        x = _rand((3, 2000), device)
        y = x.resample(orig, new, lowpass_filter_width=6, rolloff=0.99, beta=None)
    ty = _torch_resample_ref(totorch(x).double(), orig, new, 6, 0.99, None)
    assert y.shape == tuple(ty.shape)
    assert_close_mag_torch(y, ty.float(), dtype.float32, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_resample_identity_and_invalid(device: str) -> None:
    with mag.device(device):
        x = _rand((2, 50), device)
        assert x.resample(16000, 16000) is x
        with pytest.raises(ValueError):
            x.resample(0, 16000)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('orig', [44100, 48000])
def test_resample_tone_passband_and_alias_rejection(device: str, orig: int) -> None:
    new, n = 16000, orig  # 1 second
    t = torch.arange(n, dtype=torch.float64) / orig
    with mag.device(device):
        tone = Tensor((torch.sin(2 * math.pi * 1000 * t)).tolist(), dtype=dtype.float32).resample(orig, new)
        alias = Tensor((torch.sin(2 * math.pi * 12000 * t)).tolist(), dtype=dtype.float32).resample(orig, new)
    core = slice(1000, new - 1000)  # Skip filter edge transients
    expect = torch.sin(2 * math.pi * 1000 * torch.arange(new, dtype=torch.float64) / new).float()
    assert_close_mag_torch(totorch(tone)[core], expect[core], dtype.float32, rtol=0, atol=2e-3)
    assert totorch(alias)[core].abs().max().item() < 1e-3  # 12 kHz is above the 8 kHz Nyquist and must be rejected (> 60 dB)
