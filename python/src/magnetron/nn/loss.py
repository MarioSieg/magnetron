# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from abc import ABC, abstractmethod

from magnetron import Tensor


class Loss(ABC):
    """Base class for all loss functions."""

    @abstractmethod
    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error Loss."""

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        d = y_hat - y
        return (d * d).mean()


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss."""

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        y_hat = y_hat.softmax()
        return -(y * y_hat.log()).sum(dim=-1).mean()
