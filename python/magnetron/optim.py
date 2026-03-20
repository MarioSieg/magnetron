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
from collections.abc import Iterable

from . import Tensor, no_grad, context
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
    """Base class for all optimizers."""

    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        seen: set[int] = set()
        self.params: list[Parameter] = []
        for p in params:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                self.params.append(p)
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @no_grad()
    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()


class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        super().__init__(params, lr)
        self.lr = float(lr)

    @no_grad()
    def step(self) -> None:
        lazy_on = context.is_lazy_execution_enabled()
        for param in self.params:
            grad = param.grad
            if grad is None:
                continue
            if lazy_on:
                grad = grad.eval()
            upd = grad * self.lr
            if lazy_on:
                upd = upd.eval()
            param -= upd
            if lazy_on:
                param.eval()


class Adam(Optimizer):
    """Adaptive Moment Estimation"""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [Tensor.zeros(p.shape) for p in self.params]
        self.v = [Tensor.zeros(p.shape) for p in self.params]

    @no_grad()
    def step(self) -> None:
        lazy_on = context.is_lazy_execution_enabled()
        self.t += 1
        for i, p in enumerate(self.params):
            grad = p.grad
            if grad is None:
                continue
            if lazy_on:
                grad = grad.eval()
            m = self.betas[0] * self.m[i] + (1.0 - self.betas[0]) * grad
            v = self.betas[1] * self.v[i] + (1.0 - self.betas[1]) * grad.sqr()
            if lazy_on:
                m = m.eval()
                v = v.eval()
            self.m[i] = m
            self.v[i] = v
            m_hat: Tensor = self.m[i] / (1.0 - self.betas[0] ** self.t)
            v_hat: Tensor = self.v[i] / (1.0 - self.betas[1] ** self.t)
            upd = self.lr * m_hat / (v_hat.sqrt_() + self.eps)
            if lazy_on:
                upd = upd.eval()
            p -= upd
            if lazy_on:
                p.eval()
