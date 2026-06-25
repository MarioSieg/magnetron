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

from .. import Tensor, dtype, context
from .. import dtype as _dtype
from .module import Module, Parameter
from .init import *


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: dtype.DType | None = None,
        weight_init: InitStrategy | None = None,
        bias_init: InitStrategy | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight: Parameter = Parameter(Tensor.empty(out_features, in_features, dtype=dtype))
        if weight_init is None:
            weight_init = KaimingUniformInitStrategy(
                a=math.sqrt(5.0),
                mode=FanMode.FAN_IN,
                activation=Activation.LEAKY_RELU,
            )
        inplace_init(self.weight, weight_init)
        self.bias: Parameter | None = None
        if bias:
            self.bias: Parameter | None = Parameter(Tensor.empty(out_features, dtype=dtype))
            if bias_init is None:
                fan_in, _ = compute_fan_inout(self.weight)
                bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
                inplace_init(self.bias, UniformInitStrategy(-bound, bound))
            else:
                inplace_init(self.bias, bias_init)

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return x


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.contiguous().reshape(x.shape[0], -1)


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: dtype.DType | None = None,
        weight_init: InitStrategy | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(Tensor.empty(num_embeddings, embedding_dim, dtype=dtype))
        if weight_init is None:
            weight_init = NormalInitStrategy(mean=0.0, std=1.0 / embedding_dim)

        inplace_init(self.weight, weight_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight.embedding(x)


class RMSNorm(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: dtype.DType | None = None,
        weight_init: InitStrategy | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.eps = eps
        self.weight = Parameter(Tensor.empty(dim, dtype=dtype))
        if weight_init is None:
            weight_init = OnesInitStrategy()
        inplace_init(self.weight, weight_init)

    def forward(self, x: Tensor) -> Tensor:
        rms = (x.sqr().mean(dim=-1, keepdim=True) + self.eps).sqrt_()
        return (x / rms) * self.weight


class LayerNorm(Module):
    def __init__(
        self,
        ndim: int,
        bias: bool = True,
        eps: float = 1e-5,
        dtype: dtype.DType | None = None,
        weight_init: InitStrategy | None = None,
        bias_init: InitStrategy | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.eps = eps
        self.weight = Parameter(Tensor.empty(ndim, dtype=dtype))
        if weight_init is None:
            weight_init = OnesInitStrategy()
        inplace_init(self.weight, weight_init)
        self.bias = None
        if bias:
            self.bias = Parameter(Tensor.empty(ndim, dtype=dtype))
            if bias_init is None:
                bias_init = ZerosInitStrategy()
            inplace_init(self.bias, bias_init)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        xm = x - mean
        var = xm.sqr().mean(dim=-1, keepdim=True)
        x_hat = xm * (var + self.eps).rsqrt()
        y = self.weight * x_hat
        if self.bias is not None:
            y = y + self.bias
        return y
