# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import math

from .. import Tensor, dtype
from magnetron.nn.module import Module, Parameter


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.contiguous().reshape(x.shape[0], -1)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: dtype.DType = dtype.float32, init: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if init:
            self.weight = Parameter(Tensor.normal(out_features, in_features, mean=0.0, std=1.0, dtype=dtype) / math.sqrt(in_features + out_features))
        else:
            self.weight = Parameter(Tensor.empty(out_features, in_features, dtype=dtype))
        self.bias = None
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return x


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: dtype.DType = dtype.float32, init: bool = True) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if init:
            self.weight = Parameter(Tensor.normal(num_embeddings, embedding_dim, dtype=dtype) / embedding_dim)
        else:
            self.weight = Parameter(Tensor.empty(num_embeddings, embedding_dim, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: dtype.DType = dtype.float32, init: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(Tensor.ones(dim, dtype=dtype) if init else Tensor.empty(dim, dtype=dtype))

    def _norm(self, x: Tensor) -> Tensor:
        rms = (x.sqr().mean(dim=-1, keepdim=True) + self.eps).sqrt_()
        return x / rms

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x) * self.weight


class LayerNorm(Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5, dtype: dtype.DType = dtype.float32, init: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(Tensor.ones(ndim, dtype=dtype) if init else Tensor.empty(ndim, dtype=dtype))
        self.bias = Parameter(Tensor.zeros(ndim, dtype=dtype) if init else Tensor.empty(ndim, dtype=dtype)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        xm = x - mean
        var = xm.sqr().mean(dim=-1, keepdim=True)
        x_hat = xm * (var + self.eps).rsqrt()
        y = self.weight * x_hat
        if self.bias is not None:
            y = y + self.bias
        return y
