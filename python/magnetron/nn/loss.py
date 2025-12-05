# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations
from abc import ABC, abstractmethod

from magnetron import Tensor
from magnetron import dtype

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

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        m = logits.max(dim=-1, keepdim=True)
        shifted = logits - m
        logsumexp = shifted.exp().sum(dim=-1, keepdim=True).log()
        log_probs = shifted - logsumexp
        rows = Tensor.arange(target.shape[0], dtype=dtype.int64)
        log_p_true = log_probs[rows, target]
        return -log_p_true.mean()
