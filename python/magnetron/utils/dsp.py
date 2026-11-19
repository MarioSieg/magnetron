# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import math
from functools import lru_cache
from magnetron import Tensor

# Smol but cute DSP lib - we emulate FFTs with matmuls for now until real kernels are added one day
# For our usage in Whisper etc. FFTs are not the bottleneck rn


def i0(x: Tensor, terms: int = 32) -> Tensor:  # Impl of modified Bessel function using Tailor series
    y = x * x * 0.25
    t = Tensor.ones_like(x)
    r = t
    for k in range(1, terms):
        t = t * (y / k**2)
        r += t
    return r


def hann_window(n: int, periodic: bool, **kwargs) -> Tensor:
    if n == 1:
        return Tensor.ones(1, **kwargs)
    N = n if periodic else n - 1
    t = Tensor.arange(n, **kwargs)
    return 0.5 - 0.5 * (math.pi * 2 * t / N).cos()


def hamming_window(n: int, periodic: bool, **kwargs) -> Tensor:
    if n == 1:
        return Tensor.ones(1, **kwargs)
    N = n if periodic else n - 1
    t = Tensor.arange(n, **kwargs)
    return 0.54 - 0.46 * (math.pi * 2 * t / N).cos()


def blackman_window(n: int, periodic: bool, **kwargs) -> Tensor:
    if n == 1:
        return Tensor.ones(1, **kwargs)
    N = n if periodic else n - 1
    t = Tensor.arange(n, **kwargs)
    a = math.pi * 2 * t / N
    return 0.42 - 0.5 * a.cos() + 0.08 * (2 * a).cos()


def kaiser_window(n: int, beta: float = 12.0, periodic: bool = True, **kwargs) -> Tensor:
    if n == 0:
        return Tensor.ones(0, **kwargs)
    if n == 1:
        return Tensor.ones(1, **kwargs)
    N = n + 1 if periodic else n
    t = Tensor.arange(n, **kwargs)
    a = 2 * t / (N - 1) - 1
    x = beta * (1 - a**2).clamp_min(0).sqrt()
    return i0(x) / i0(beta * Tensor.ones_like(x))


def _cfft(inv: bool, xr: Tensor, xi: Tensor | None = None, dim: int = -1) -> tuple[Tensor, Tensor]:
    if xi is None:
        if inv:
            raise RuntimeError('Required')
        xi = Tensor.zeros_like(xr)
    N: int = xr.shape[dim]
    n = Tensor.arange(N, dtype=xr.dtype, device=xr.device)
    theta = (math.pi * 2 / N) * ((n[:, None] * n[None, :]) % N)
    C, S = theta.cos(), theta.sin()
    if inv:
        C /= N
        S /= N
    M = Tensor.cat([Tensor.cat([C, -S if inv else S], dim=1), Tensor.cat([S if inv else -S, C], dim=1)], dim=0)
    x = Tensor.cat([xr.movedim(dim, -1), xi.movedim(dim, -1)], dim=-1)
    y = x @ M.T
    yr, yi = y.split(N, dim=-1)
    return yr.movedim(-1, dim), yi.movedim(-1, dim)


def fft(xr: Tensor, xi: Tensor | None = None, dim: int = -1) -> tuple[Tensor, Tensor]:
    return _cfft(False, xr, xi, dim)


def ifft(xr: Tensor, xi: Tensor | None = None, dim: int = -1) -> tuple[Tensor, Tensor]:
    return _cfft(True, xr, xi, dim)


def _rfft(inv: bool, xr: Tensor, xi: Tensor | None = None, n: int | None = None, dim: int = -1) -> Tensor | tuple[Tensor, Tensor]:
    if inv:
        if xi is None or n is None:
            raise RuntimeError('Required')
        N = n
        K = N // 2 + 1
        assert xr.shape[dim] == K
    else:
        N = xr.shape[dim]
        K = N // 2 + 1

    t = Tensor.arange(N, dtype=xr.dtype, device=xr.device)
    k = Tensor.arange(K, dtype=xr.dtype, device=xr.device)
    theta = (math.pi * 2 / N) * ((k[:, None] * t[None, :]) % N)
    C, S = theta.cos(), theta.sin()
    if inv:
        w = Tensor.ones(K, dtype=xr.dtype, device=xr.device) * 2
        w[0] = 1
        if (N & 1) == 0:
            w[-1] = 1
        C = C * w[:, None] / N
        S = -S * w[:, None] / N
        M = Tensor.cat([C, S], dim=0)
        x = Tensor.cat([xr.movedim(dim, -1), xi.movedim(dim, -1)], dim=-1)
        return (x @ M).movedim(-1, dim)
    M = Tensor.cat([C, -S], dim=0)
    y = xr.movedim(dim, -1) @ M.T
    yr, yi = y.split(K, dim=-1)
    return yr.movedim(-1, dim), yi.movedim(-1, dim)


def rfft(x: Tensor, dim: int = -1) -> tuple[Tensor, Tensor]:
    return _rfft(False, x, dim=dim)


def irfft(xr: Tensor, xi: Tensor, n: int, dim: int = -1) -> Tensor:
    return _rfft(True, xr, xi, n=n, dim=dim)


# Windowed-sinc polyphase resampler defaults
_KAISER_BEST_WIDTH: int = 64
_KAISER_BEST_ROLLOFF: float = 0.9475937167399596
_KAISER_BEST_BETA: float = 14.769656459379492


@lru_cache(maxsize=16)
def _sinc_resample_kernel(orig: int, new: int, lowpass_filter_width: int, rolloff: float, beta: float | None, **kwargs) -> tuple[Tensor, int]:
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    idx = Tensor.arange(-width, width + orig, **kwargs) / orig
    phase = Tensor.arange(new, **kwargs) / -new
    t = ((phase[:, None] + idx[None, :]) * base).clamp(-lowpass_filter_width, lowpass_filter_width)
    if beta is None:
        window = (t * (math.pi / lowpass_filter_width / 2)).cos() ** 2
    else:
        r = t / lowpass_filter_width
        window = i0(beta * (1 - r * r).clamp_min(0).sqrt()) / sum((beta * beta / 4) ** k / math.factorial(k) ** 2 for k in range(64))
    t = t * math.pi
    sinc = Tensor.where(t == 0, Tensor.ones_like(t), t.sin() / Tensor.where(t == 0, Tensor.ones_like(t), t))
    return (sinc * window * (base / orig)).contiguous(), width


def resample(
    x: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = _KAISER_BEST_WIDTH,
    rolloff: float = _KAISER_BEST_ROLLOFF,
    beta: float | None = _KAISER_BEST_BETA,
) -> Tensor:
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError(f'Sample rates must be positive, got {orig_freq} -> {new_freq}')
    if orig_freq == new_freq:
        return x
    g = math.gcd(int(orig_freq), int(new_freq))
    orig, new = int(orig_freq) // g, int(new_freq) // g
    kernel, width = _sinc_resample_kernel(orig, new, lowpass_filter_width, rolloff, beta, dtype=x.dtype, device=x.device)
    K = kernel.shape[1]
    shape = x.shape
    length = shape[-1]
    xb = x.reshape(-1, length)
    B = xb.shape[0]
    xp = xb.pad((width, width + orig)).contiguous()
    blocks = length // orig + 1
    frames = xp.strided_view((B, blocks, K), (xp.shape[1], orig, 1))
    y = (frames @ kernel.T).reshape(B, blocks * new)
    target = math.ceil(new * length / orig)
    return y[:, :target].contiguous().reshape(*shape[:-1], target)
