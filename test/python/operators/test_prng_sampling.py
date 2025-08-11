# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from ..common import *

import torch
import pytest
from magnetron import Tensor, float16, float32

def _make_x(B: int, K: int, dtype):
    return Tensor.uniform((B, K), dtype=dtype)

def _expected_probs_from_x(x_torch: torch.Tensor) -> torch.Tensor:
    w = x_torch.to(torch.float64)
    w = torch.where(torch.isfinite(w) & (w > 0), w, torch.zeros_like(w))
    s = w.sum(dim=-1, keepdim=True)
    s = torch.clamp_min(s, 1e-24)
    return w / s

def _hist_from_samples(samples_torch: torch.Tensor, K: int) -> torch.Tensor:
    idx = samples_torch.to(torch.int64).reshape(-1)
    return torch.bincount(idx, minlength=K)

@pytest.mark.parametrize('dtype', [float16, float32])
def test_multinomial_replacement_frequency(dtype):
    B, K, S = 2048, 8, 8
    x = _make_x(B, K, dtype)
    p_row = _expected_probs_from_x(totorch(x))
    p_exp = p_row.mean(dim=0)
    out = x.multinomial(S, replacement=True)
    hist = _hist_from_samples(totorch(out), K).to(torch.float64)
    n = float(B * S)
    p_hat = hist / n

    var = torch.clamp(p_exp * (1 - p_exp) / n, min=1e-12)
    z = (p_hat - p_exp) / torch.sqrt(var)
    assert torch.all(z.abs() < 5.0), f'z too large: {z}'

@pytest.mark.parametrize('dtype', [float16, float32])
def test_multinomial_no_replacement_uniqueness(dtype):
    B, K, S = 1024, 12, 6
    x = _make_x(B, K, dtype)
    out = totorch(x.multinomial(S, replacement=False)).to(torch.int64)
    offsets = (torch.arange(B, dtype=torch.int64).unsqueeze(1) * K)
    keys = out + offsets
    keys = keys[out >= 0]
    maxcount = torch.bincount(keys).max()
    assert int(maxcount) <= 1, 'duplicate index within a row for replacement=False'

@pytest.mark.parametrize('dtype', [float16, float32])
def test_multinomial_pl_pairwise_order(dtype):
    B, K = 2000, 6
    x = _make_x(B, K, dtype)
    w = totorch(x).to(torch.float64)
    w = torch.where(torch.isfinite(w) & (w > 0), w, torch.zeros_like(w))
    perm = totorch(x.multinomial(K, replacement=False)).to(torch.int64)

    pos = torch.empty_like(perm)
    ar = torch.arange(K, dtype=torch.int64).expand(B, K)
    pos.scatter_(1, perm, ar)

    z_max = 5.0
    for i in range(K):
        wi = w[:, i]
        for j in range(i + 1, K):
            wj = w[:, j]
            pexp_row = wi / torch.clamp_min(wi + wj, 1e-24)
            pexp = pexp_row.mean().item()
            wins = (pos[:, i] < pos[:, j]).float().mean().item()
            var = max(pexp * (1 - pexp) / B, 1e-12)
            z = (wins - pexp) / (var ** 0.5)
            assert abs(z) < z_max, f'pair ({i},{j}) z={z:.2f} wins={wins:.3f} exp={pexp:.3f}'
            