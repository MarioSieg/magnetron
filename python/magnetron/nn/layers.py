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


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.contiguous().reshape(x.shape[0], -1)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor.normal(out_features, in_features, mean=0.0, std=1.0) / math.sqrt(in_features + out_features))
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
        self.weight = Parameter(Tensor.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        rms = (x.sqr().mean(dim=-1, keepdim=True) + self.eps).sqrt_()
        return x / rms

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x) * self.weight.x


class LayerNorm(Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = Parameter(Tensor.ones(ndim))
        self.bias = Parameter(Tensor.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        xm = x - mean
        var = xm.sqr().mean(dim=-1, keepdim=True)
        x_hat = xm*(var + self.eps).rsqrt()
        y = self.weight.x*x_hat
        if self.bias is not None:
            y = y + self.bias.x
        return y
