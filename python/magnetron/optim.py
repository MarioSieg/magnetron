# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

from abc import ABC, abstractmethod

from ._tensor import Tensor
from ._context import no_grad
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

    def __init__(self, params: list[Parameter], lr: float) -> None:
        self.params = params
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @no_grad()
    def zero_grad(self) -> None:
        for param in self.params:
            param.x.zero_grad()


class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def __init__(self, params: list[Parameter], lr: float) -> None:
        super().__init__(params, lr)

    @no_grad()
    def step(self) -> None:
        for param in self.params:
            param.x = param.x - param.x.grad * self.lr
            param.x.requires_grad = True


class Adam(Optimizer):
    """Adaptive Moment Estimation"""

    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [Tensor.zeros(p.x.shape) for p in self.params]
        self.v = [Tensor.zeros(p.x.shape) for p in self.params]

    @no_grad()
    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            grad = p.x.grad
            if grad is None:
                continue
            self.m[i] = self.betas[0] * self.m[i] + (1.0 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1.0 - self.betas[1]) * grad.sqr_()
            m_hat: Tensor = self.m[i] / (1.0 - self.betas[0] ** self.t)
            v_hat: Tensor = self.v[i] / (1.0 - self.betas[1] ** self.t)
            p.x -= self.lr * m_hat / (v_hat.sqrt_() + self.eps)
