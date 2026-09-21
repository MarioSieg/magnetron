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
from collections.abc import Sequence

from .. import Tensor, dtype, context
from .. import dtype as _dtype
from .module import Module, Parameter
from .init import *


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


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
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)


class Unflatten(Module):
    def __init__(self, dim: int, unflattened_size: tuple[int, ...]) -> None:
        super().__init__()
        self.dim = dim
        self.unflattened_size = tuple(unflattened_size)

    def forward(self, x: Tensor) -> Tensor:
        return x.unflatten(self.dim, self.unflattened_size)


class Pad(Module):
    def __init__(self, padding, mode: str = 'constant', value: float = 0.0) -> None:
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        return x.pad(self.padding, self.mode, self.value)


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


class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax()


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class HardSigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.hardsigmoid()


class SiLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class Tanh(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GeLU(Module):
    def __init__(self, use_tanh_approx: bool = False) -> None:
        super().__init__()
        self.use_tanh_approx = use_tanh_approx

    def forward(self, x: Tensor) -> Tensor:
        return x.gelu_approx() if self.use_tanh_approx else x.gelu()


class _ConvND(Module):
    _spatial: int = 0  # spatial dim, 1=conv1d,2=conv2d etc..
    _T: bool = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        output_padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: dtype.DType | None = None,
        weight_init: InitStrategy | None = None,
        bias_init: InitStrategy | None = None,
    ) -> None:
        def _form_tuple_n(value: int | Sequence[int], n: int, what: str) -> tuple[int, ...]:
            if isinstance(value, int):
                return (value,) * n
            value = tuple(value)
            if len(value) != n:
                raise ValueError(f'{what} must be an int or a sequence of {n} ints, but got {value}')
            return value

        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        n = self._spatial
        if groups < 1:
            raise ValueError(f'groups must be >= 1, but got {groups}')
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _form_tuple_n(kernel_size, n, 'kernel_size')
        self.stride = _form_tuple_n(stride, n, 'stride')
        self.padding = _form_tuple_n(padding, n, 'padding')
        self.output_padding = _form_tuple_n(output_padding, n, 'output_padding')
        self.dilation = _form_tuple_n(dilation, n, 'dilation')
        self.groups = groups
        if self._T:
            wshape = (in_channels, out_channels // groups, *self.kernel_size)
        else:
            wshape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight: Parameter = Parameter(Tensor.empty(*wshape, dtype=dtype))
        if weight_init is None:
            weight_init = KaimingUniformInitStrategy(
                a=math.sqrt(5.0),
                mode=FanMode.FAN_IN,
                activation=Activation.LEAKY_RELU,
            )
        inplace_init(self.weight, weight_init)
        self.bias: Parameter | None = None
        if bias:
            self.bias = Parameter(Tensor.empty(out_channels, dtype=dtype))
            if bias_init is None:
                fan_in, _ = compute_fan_inout(self.weight)
                bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
                inplace_init(self.bias, UniformInitStrategy(-bound, bound))
            else:
                inplace_init(self.bias, bias_init)

    def forward(self, x: Tensor) -> Tensor:
        if self._T:
            return getattr(x, f'convT{self._spatial}D')(
                self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
            )
        return getattr(x, f'conv{self._spatial}D')(self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv1D(_ConvND):
    _spatial = 1


class Conv2D(_ConvND):
    _spatial = 2


class Conv3D(_ConvND):
    _spatial = 3


class ConvT1D(_ConvND):
    _spatial = 1
    _T = True


class ConvT2D(_ConvND):
    _spatial = 2
    _T = True


class ConvT3D(_ConvND):
    _spatial = 3
    _T = True
