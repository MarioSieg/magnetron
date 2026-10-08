# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import math

import torch

import magnetron as mag
from magnetron import nn, optim
from magnetron.nn.init import (
    Activation,
    ConstantInitStrategy,
    FanMode,
    KaimingNormalInitStrategy,
    KaimingUniformInitStrategy,
    NormalInitStrategy,
    OnesInitStrategy,
    UniformInitStrategy,
    XavierNormalInitStrategy,
    XavierUniformInitStrategy,
    ZerosInitStrategy,
    compute_fan_inout,
    inplace_init,
)

from .common import *

_SHAPES = ((7,), (3, 5), (2, 3, 4))
_STEPS = 8


def _params(device: str) -> tuple[list[nn.Parameter], list[torch.nn.Parameter], list[Tensor], list[torch.Tensor]]:
    ps, tps, targets, ttargets = [], [], [], []
    for shape in _SHAPES:
        p = nn.Parameter(uniform_tensor(shape, low=-1.0, high=1.0, device=device))
        t = uniform_tensor(shape, low=-1.0, high=1.0, device=device)
        ps.append(p)
        tps.append(torch.nn.Parameter(totorch(p).clone()))
        targets.append(t)
        ttargets.append(totorch(t))
    return ps, tps, targets, ttargets


def _loss(ps, targets):
    total = None
    for p, t in zip(ps, targets):
        d = p - t
        term = (d * d + p * p * p * 0.1).sum()
        total = term if total is None else total + term
    return total


def _run(device: str, make_mag: Callable, make_torch: Callable) -> None:
    ps, tps, targets, ttargets = _params(device)
    opt = make_mag(ps)
    topt = make_torch(tps)
    for _ in range(_STEPS):
        opt.zero_grad()
        topt.zero_grad()
        loss = _loss(ps, targets)
        tloss = _loss(tps, ttargets)
        assert_close_mag_torch(loss, tloss.detach(), dtype.float32)
        loss.backward()
        tloss.backward()
        for p, tp in zip(ps, tps):
            assert_close_mag_torch(p.grad, tp.grad, dtype.float32)
        opt.step()
        topt.step()
        for p, tp in zip(ps, tps):
            assert_close_mag_torch(p, tp.detach(), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('lr', [0.1, 0.01])
def test_sgd_matches_torch(device: str, lr: float) -> None:
    _run(device, lambda ps: optim.SGD(ps, lr=lr), lambda ps: torch.optim.SGD(ps, lr=lr))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('lr, betas, eps', [(0.1, (0.9, 0.999), 1e-8), (0.01, (0.8, 0.99), 1e-6)])
def test_adam_matches_torch(device: str, lr: float, betas, eps: float) -> None:
    _run(device, lambda ps: optim.Adam(ps, lr=lr, betas=betas, eps=eps), lambda ps: torch.optim.Adam(ps, lr=lr, betas=betas, eps=eps))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('lr, betas, eps, weight_decay', [(0.1, (0.9, 0.999), 1e-8, 0.1), (0.01, (0.8, 0.99), 1e-6, 0.5), (0.05, (0.9, 0.999), 1e-8, 0.0)])
def test_adamw_matches_torch(device: str, lr: float, betas, eps: float, weight_decay: float) -> None:
    _run(
        device,
        lambda ps: optim.AdamW(ps, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        lambda ps: torch.optim.AdamW(ps, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
    )


def test_polynomial_decay_matches_torch_polynomial_lr() -> None:
    initial_lr, max_iter = 0.25, 40
    sched = optim.PolynomialDecayLRScheduler(initial_lr, max_iter)
    p = torch.nn.Parameter(torch.zeros(1))
    topt = torch.optim.SGD([p], lr=initial_lr)
    tsched = torch.optim.lr_scheduler.PolynomialLR(topt, total_iters=max_iter, power=2)
    for it in range(max_iter):
        assert math.isclose(sched.step(it), tsched.get_last_lr()[0], rel_tol=1e-9, abs_tol=1e-12)
        topt.step()
        tsched.step()


_FAN_SHAPES = ((6, 4), (8, 3, 5), (16, 4, 3, 3), (4, 2, 2, 3, 3), (1, 1), (5, 1, 7))


@pytest.mark.parametrize('shape', _FAN_SHAPES)
def test_fan_in_out_matches_torch(shape) -> None:
    fan_in, fan_out = compute_fan_inout(Tensor.empty(*shape))
    assert (fan_in, fan_out) == torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*shape))


def test_fan_rejects_rank_below_two() -> None:
    with pytest.raises(ValueError):
        compute_fan_inout(Tensor.empty(5))
    with pytest.raises(ValueError):
        torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(5))


@pytest.mark.parametrize('activation, param, torch_name', [(Activation.SIGMOID, None, 'sigmoid'), (Activation.TANH, None, 'tanh'), (Activation.RELU, None, 'relu'), (Activation.LEAKY_RELU, None, 'leaky_relu'), (Activation.LEAKY_RELU, 0.2, 'leaky_relu'), (Activation.LEAKY_RELU, math.sqrt(5.0), 'leaky_relu')])
def test_gain_matches_torch(activation: Activation, param, torch_name: str) -> None:
    assert math.isclose(activation.compute_gain(param), torch.nn.init.calculate_gain(torch_name, param), rel_tol=1e-12)


_INIT_SHAPE = (256, 512)
_SIGMAS = 6.0


def _sample(strategy) -> torch.Tensor:
    w = Tensor.empty(*_INIT_SHAPE)
    inplace_init(w, strategy)
    return totorch(w).to(torch.float64)


def _assert_mean_std(t: torch.Tensor, mean: float, std: float) -> None:
    n = t.numel()
    assert abs(t.mean().item() - mean) <= _SIGMAS * std / math.sqrt(n)
    assert abs(t.std().item() - std) <= _SIGMAS * std / math.sqrt(2 * n)


def _assert_uniform(t: torch.Tensor, low: float, high: float) -> None:
    assert t.min().item() >= low
    assert t.max().item() <= high
    _assert_mean_std(t, (low + high) / 2.0, (high - low) / math.sqrt(12.0))


def test_constant_inits() -> None:
    assert torch.equal(_sample(ZerosInitStrategy()), torch.zeros(_INIT_SHAPE, dtype=torch.float64))
    assert torch.equal(_sample(OnesInitStrategy()), torch.ones(_INIT_SHAPE, dtype=torch.float64))
    assert torch.equal(_sample(ConstantInitStrategy(-2.5)), torch.full(_INIT_SHAPE, -2.5, dtype=torch.float64))


@pytest.mark.parametrize('low, high', [(-1.0, 1.0), (0.0, 1.0), (2.5, 7.5)])
def test_uniform_init(low: float, high: float) -> None:
    _assert_uniform(_sample(UniformInitStrategy(low, high)), low, high)


@pytest.mark.parametrize('mean, std', [(0.0, 1.0), (0.3, 0.6), (-2.0, 3.0)])
def test_normal_init(mean: float, std: float) -> None:
    _assert_mean_std(_sample(NormalInitStrategy(mean, std)), mean, std)


@pytest.mark.parametrize('gain', [1.0, 0.5, math.sqrt(2.0)])
def test_xavier_uniform_init(gain: float) -> None:
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*_INIT_SHAPE))
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    _assert_uniform(_sample(XavierUniformInitStrategy(gain)), -bound, bound)


@pytest.mark.parametrize('gain', [1.0, 0.5, math.sqrt(2.0)])
def test_xavier_normal_init(gain: float) -> None:
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*_INIT_SHAPE))
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    _assert_mean_std(_sample(XavierNormalInitStrategy(gain)), 0.0, std)


@pytest.mark.parametrize('a, mode, activation', [(0.0, FanMode.FAN_IN, Activation.RELU), (math.sqrt(5.0), FanMode.FAN_IN, Activation.LEAKY_RELU), (0.2, FanMode.FAN_OUT, Activation.LEAKY_RELU), (0.0, FanMode.FAN_OUT, Activation.TANH)])
def test_kaiming_inits(a: float, mode: FanMode, activation: Activation) -> None:
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(*_INIT_SHAPE))
    fan = fan_in if mode == FanMode.FAN_IN else fan_out
    gain = torch.nn.init.calculate_gain(activation.value, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    _assert_uniform(_sample(KaimingUniformInitStrategy(a, mode, activation)), -bound, bound)
    _assert_mean_std(_sample(KaimingNormalInitStrategy(a, mode, activation)), 0.0, std)


def test_linear_default_init_matches_torch_bounds() -> None:
    layer = nn.Linear(512, 256)
    w = totorch(layer.weight).to(torch.float64)
    b = totorch(layer.bias).to(torch.float64)
    ref = torch.nn.Linear(512, 256)
    fan_in = 512
    w_bound = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5.0)) / math.sqrt(fan_in) * math.sqrt(3.0)
    b_bound = 1.0 / math.sqrt(fan_in)
    assert ref.weight.abs().max().item() <= w_bound
    _assert_uniform(w, -w_bound, w_bound)
    _assert_uniform(b, -b_bound, b_bound)
