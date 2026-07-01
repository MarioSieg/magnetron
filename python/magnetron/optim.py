# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Callable
from typing import Any

from . import Tensor, no_grad
from .nn import Parameter


class PolynomialDecayLRScheduler:
    """Polynomial Decay Learning Rate Scheduler"""

    def __init__(self, initial_lr: float, max_iter: float) -> None:
        self.initial_lr = initial_lr
        self.max_iter = max_iter

    def step(self, iter: float) -> float:
        y: float = iter / self.max_iter
        return max(self.initial_lr * (1 - y) ** 2, 1.0e-7)


class Optimizer(ABC):
    """Base class of all optimizers."""

    def __init__(self, params: Iterable[Parameter | dict[str, Any]], defaults: dict[str, Any]) -> None:
        self.defaults = defaults
        self.state: defaultdict[Parameter, dict[str, Any]] = defaultdict(dict)
        self.param_groups: list[dict[str, Any]] = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError('Optimizer got an empty parameter list')
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for group in param_groups:
            self.add_param_group(group)

    @abstractmethod
    def step(self, closure: Callable[[], None] | None = None):
        raise NotImplementedError()

    @no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # if set_to_none:
                    #    p.grad = None
                    # else:
                    #    p.grad.zero_()
                    p.grad.zero_()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if not isinstance(param_group, dict):
            raise TypeError(f'param_group must be a dict, got {type(param_group)}')
        if 'params' not in param_group:
            raise ValueError("param group must contain a 'params' key")
        params = param_group['params']
        if isinstance(params, Parameter):
            params = [params]
        elif isinstance(params, Tensor):
            raise TypeError(f'optimizer can only optimize Parameter, got raw Tensor')
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need deterministic ordering; use list/tuple, not set')
        else:
            params = list(params)
        if len(params) == 0:
            raise ValueError('optimizer got an empty parameter group')
        for p in params:
            if not isinstance(p, Parameter):
                raise TypeError(f'optimizer can only optimize Parameter, got {type(p)}')
        param_group = dict(param_group)
        param_group['params'] = params
        for k, v in self.defaults.items():
            param_group.setdefault(k, v)
        existing = {p for g in self.param_groups for p in g['params']}
        overlap = existing.intersection(params)
        if overlap:
            raise ValueError('some parameters appear in more than one parameter group')
        self.param_groups.append(param_group)


class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def __init__(self, params: Iterable[Parameter | dict[str, Any]], lr: float) -> None:
        super().__init__(params, defaults=dict(lr=float(lr)))

    @no_grad()
    def step(self, closure: Callable[[], None] | None = None) -> None:
        if closure is not None:
            closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p -= p.grad * lr


class Adam(Optimizer):
    """Adaptive Moment Estimation"""

    def __init__(
        self,
        params: Iterable[Parameter | dict[str, Any]],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(lr=float(lr), betas=betas, eps=float(eps))
        super().__init__(params, defaults)

    @no_grad()
    def step(self, closure: Callable[[], None] | None = None) -> None:
        if closure is not None:
            closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = Tensor.zeros(p.shape)
                    state['v'] = Tensor.zeros(p.shape)
                state['step'] += 1
                t = state['step']
                m = state['m']
                v = state['v']
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad.sqr()
                state['m'] = m
                state['v'] = v
                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)
                p -= lr * m_hat / (v_hat.sqrt() + eps)


class AdamW(Optimizer):
    """Adam with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Parameter | dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=float(lr),
            betas=betas,
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)

    @no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = Tensor.zeros(p.shape)
                    state['v'] = Tensor.zeros(p.shape)
                state['step'] += 1
                t = state['step']
                if weight_decay != 0.0:
                    p -= lr * weight_decay * p
                m = state['m']
                v = state['v']
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad.sqr()
                state['m'] = m
                state['v'] = v
                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)
                p -= lr * m_hat / (v_hat.sqrt() + eps)
