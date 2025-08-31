# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import math

from magnetron import Tensor
from magnetron.nn.module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_weights: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if init_weights:
            weight = Tensor.normal(out_features, in_features, mean=0.0, std=1.0)
            weight = weight / math.sqrt(in_features + out_features)
        else:
            weight = Tensor.zeros(out_features, in_features)
        self.weight = Parameter(weight)
        self.bias = None
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight.x.T
        if self.bias is not None:
            x = x + self.bias.x
        return x


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(Tensor.normal(num_embeddings, embedding_dim) / embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight.x[x]


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(Tensor.zeros(dim))

    def _norm(self, x: Tensor) -> Tensor:
        rms = ((x**2).mean(axis=-1, keepdim=True) + self.eps) ** 0.5
        return x / rms

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x)
        return output * self.weight


class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        assert 0 <= p <= 1, 'Bernoulli probability must be between 0 and 1'
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x.dropout_(self.p) if self.inplace else x.dropout(self.p)


class LayerNorm(Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = Parameter(Tensor.ones(ndim))
        self.bias = Parameter(Tensor.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        xm = x - x.mean(dim=-1, keepdim=True)
        var = (xm * xm).mean(dim=-1, keepdim=True)
        x_hat = xm / (var + self.eps).sqrt()
        y = self.weight.x * x_hat
        if self.bias is not None:
            y = y + self.bias.x
        return y
