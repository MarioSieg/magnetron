# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .. import Tensor, dtype, context
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
        weight_init = resolve_weight_init(weight_init)
        bias_init = resolve_bias_init(bias_init)
        if weight_init is None:
            weight_init = KaimingUniformInitStrategy(a=math.sqrt(5.0), mode=FanMode.FAN_IN, activation=Activation.LEAKY_RELU)
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

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

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

    def extra_repr(self) -> str:
        return f'start_dim={self.start_dim}, end_dim={self.end_dim}'

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)


class Unflatten(Module):
    def __init__(self, dim: int, unflattened_size: tuple[int, ...]) -> None:
        super().__init__()
        self.dim = dim
        self.unflattened_size = tuple(unflattened_size)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, unflattened_size={self.unflattened_size}'

    def forward(self, x: Tensor) -> Tensor:
        return x.unflatten(self.dim, self.unflattened_size)


class Pad(Module):
    def __init__(self, padding, mode: str = 'constant', value: float = 0.0) -> None:
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.value = value

    def extra_repr(self) -> str:
        return f'padding={self.padding}, mode={self.mode!r}, value={self.value}'

    def forward(self, x: Tensor) -> Tensor:
        return x.pad(self.padding, self.mode, self.value)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: dtype.DType | None = None, weight_init: InitStrategy | None = None) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(Tensor.empty(num_embeddings, embedding_dim, dtype=dtype))
        weight_init = resolve_weight_init(weight_init)
        if weight_init is None:
            weight_init = NormalInitStrategy(mean=0.0, std=1.0 / embedding_dim)

        inplace_init(self.weight, weight_init)

    def extra_repr(self) -> str:
        return f'{self.num_embeddings}, {self.embedding_dim}'

    def forward(self, x: Tensor) -> Tensor:
        return self.weight.embedding(x)


class RMSNorm(Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, dtype: dtype.DType | None = None, weight_init: InitStrategy | None = None, gamma_offset: float = 0.0
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.eps = eps
        self.weight = Parameter(Tensor.empty(dim, dtype=dtype))
        self.gamma_offset = gamma_offset
        weight_init = resolve_weight_init(weight_init)
        if weight_init is None:
            weight_init = ZerosInitStrategy() if gamma_offset == 1.0 else OnesInitStrategy()
        inplace_init(self.weight, weight_init)

    def extra_repr(self) -> str:
        s = f'{self.weight.shape[0]}, eps={self.eps}'
        if self.gamma_offset != 0.0:
            s += f', gamma_offset={self.gamma_offset}'
        return s

    def forward(self, x: Tensor) -> Tensor:
        dt = x.dtype
        h = x.cast(dtype.float32)
        h = h * (h.sqr().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (h * (self.gamma_offset + self.weight.cast(h.dtype))).cast(dt)


class RMSNormGated(RMSNorm):
    def __init__(
        self, dim: int, eps: float = 1e-5, dtype: dtype.DType | None = None, weight_init: InitStrategy | None = None, gamma_offset: float = 0.0
    ) -> None:
        super().__init__(dim=dim, eps=eps, dtype=dtype, weight_init=weight_init, gamma_offset=gamma_offset)

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        h = super().forward(x)
        return (h.cast(dtype.float32) * gate.cast(dtype.float32).silu()).cast(x.dtype)


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
        weight_init = resolve_weight_init(weight_init)
        bias_init = resolve_bias_init(bias_init)
        if weight_init is None:
            weight_init = OnesInitStrategy()
        inplace_init(self.weight, weight_init)
        self.bias = None
        if bias:
            self.bias = Parameter(Tensor.empty(ndim, dtype=dtype))
            if bias_init is None:
                bias_init = ZerosInitStrategy()
            inplace_init(self.bias, bias_init)

    def extra_repr(self) -> str:
        return f'{self.weight.shape[0]}, eps={self.eps}, bias={self.bias is not None}'

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
        return x.hard_sigmoid()


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


class GroupNorm(Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: dtype.DType | None = None,
        weight_init: InitStrategy | None = None,
        bias_init: InitStrategy | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        if num_groups < 1:
            raise ValueError(f'num_groups must be >= 1, but got {num_groups}')
        if num_channels % num_groups != 0:
            raise ValueError(f'num_channels ({num_channels}) must be divisible by num_groups ({num_groups})')
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight: Parameter | None = None
        self.bias: Parameter | None = None
        if affine:
            weight_init = resolve_weight_init(weight_init)
            bias_init = resolve_bias_init(bias_init)
            self.weight = Parameter(Tensor.empty(num_channels, dtype=dtype))
            inplace_init(self.weight, OnesInitStrategy() if weight_init is None else weight_init)
            self.bias = Parameter(Tensor.empty(num_channels, dtype=dtype))
            inplace_init(self.bias, ZerosInitStrategy() if bias_init is None else bias_init)

    def extra_repr(self) -> str:
        return f'{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.weight is not None}'

    def forward(self, x: Tensor) -> Tensor:
        if x.rank < 2 or x.shape[1] != self.num_channels:
            raise ValueError(f'expected input of shape [N, {self.num_channels}, ...], but got {x.shape}')
        xg = x.reshape(x.shape[0], self.num_groups, -1)
        mean = xg.mean(dim=-1, keepdim=True)
        xm = xg - mean
        var = xm.sqr().mean(dim=-1, keepdim=True)
        y = (xm * (var + self.eps).rsqrt()).reshape(*x.shape)
        if self.weight is not None:
            bshape = (1, self.num_channels) + (1,) * (x.rank - 2)
            y = y * self.weight.reshape(*bshape) + self.bias.reshape(*bshape)
        return y


class Upsample(Module):
    def __init__(
        self,
        size: int | Sequence[int] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        mode: str = 'nearest',
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if (size is None) == (scale_factor is None):
            raise ValueError('exactly one of size or scale_factor must be given')
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def extra_repr(self) -> str:
        s = f'size={self.size}' if self.size is not None else f'scale_factor={self.scale_factor}'
        s += f', mode={self.mode!r}'
        if self.align_corners:
            s += ', align_corners=True'
        return s

    def forward(self, x: Tensor) -> Tensor:
        return x.interpolate(size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class GeLU(Module):
    def __init__(self, use_tanh_approx: bool = False) -> None:
        super().__init__()
        self.use_tanh_approx = use_tanh_approx

    def extra_repr(self) -> str:
        return 'use_tanh_approx=True' if self.use_tanh_approx else ''

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
        weight_init = resolve_weight_init(weight_init)
        bias_init = resolve_bias_init(bias_init)
        if weight_init is None:
            weight_init = KaimingUniformInitStrategy(a=math.sqrt(5.0), mode=FanMode.FAN_IN, activation=Activation.LEAKY_RELU)
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

    def extra_repr(self) -> str:
        n = self._spatial
        s = f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}'
        if self.padding != (0,) * n:
            s += f', padding={self.padding}'
        if self._T and self.output_padding != (0,) * n:
            s += f', output_padding={self.output_padding}'
        if self.dilation != (1,) * n:
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.bias is None:
            s += ', bias=False'
        return s

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


class Dropout(Module):
    _spatial: int = 0

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'Dropout probability must be within [0, 1], got {p}')
        self.p: float = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:  # Only mask during training or when p makes it possible
            return x
        if self.p == 1.0:  # Drop all (full mask)
            return Tensor.zeros_like(x)
        p = 1.0 - self.p
        if self._spatial == 0:  # elementwise mask
            mask = Tensor.bernoulli_like(x, p=p)
        else:  # one mask per val, broadcasted
            if x.rank not in (self._spatial + 1, self._spatial + 2):
                raise ValueError(
                    f'{type(self).__name__} expects (N, C, *spatial) or (C, *spatial) input with {self._spatial} spatial dims, got shape {x.shape}'
                )
            mask = Tensor.bernoulli(*(tuple(x.shape[: x.rank - self._spatial]) + (1,) * self._spatial), p=p)
        return x * mask.cast(x.dtype) * (1.0 / p)

    def extra_repr(self) -> str:
        return f'p={self.p}'


class Dropout1D(Dropout):
    _spatial = 1


class Dropout2D(Dropout):
    _spatial = 2


class Dropout3D(Dropout):
    _spatial = 3


class _RNNCellBase(Module):
    _num_gates: int = 1

    @staticmethod
    def _rnn_init(hidden: int, winit: InitStrategy | None, binit: InitStrategy | None) -> tuple[InitStrategy, InitStrategy]:
        bnd = 1.0 / (float(hidden) ** 0.5) if hidden > 0 else 0.0
        def_init = UniformInitStrategy(-bnd, bnd)  # 𝒰(−1/√H, 1/√H)
        return resolve_weight_init(winit) or def_init, resolve_bias_init(binit) or def_init

    @staticmethod
    def _rnn_gates(x: Tensor, h: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor | None, b_hh: Tensor | None) -> tuple[Tensor, Tensor]:
        gi = x @ w_ih.T
        gh = h @ w_hh.T
        if b_ih is not None:
            gi = gi + b_ih
            gh = gh + b_hh
        return gi, gh

    @staticmethod
    def _lstm_step(x: Tensor, h: Tensor, c: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor | None, b_hh: Tensor | None) -> tuple[Tensor, Tensor]:
        gi, gh = _RNNCellBase._rnn_gates(x, h, w_ih, w_hh, b_ih, b_hh)
        i, f, g, o = (gi + gh).split(h.shape[-1], dim=-1)
        c = c * f.sigmoid() + i.sigmoid() * g.tanh()
        h = c.tanh() * o.sigmoid()
        return h, c

    @staticmethod
    def _gru_step(x: Tensor, h: Tensor, c: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor | None, b_hh: Tensor | None) -> Tensor:
        gi, gh = _RNNCellBase._rnn_gates(x, h, w_ih, w_hh, b_ih, b_hh)
        hid = h.shape[-1]
        i_r, i_z, i_n = gi.split(hid, dim=-1)
        h_r, h_z, h_n = gh.split(hid, dim=-1)
        r = (i_r + h_r).sigmoid()
        z = (i_z + h_z).sigmoid()
        n = (i_n + r * h_n).tanh()
        return n + z * (h - n)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        dtype: dtype.DType | None = None,
        weight_init: InitStrategy | None = None,
        bias_init: InitStrategy | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        gates = hidden_size * self._num_gates
        weight_init, bias_init = self._rnn_init(hidden_size, weight_init, bias_init)
        self.weight_ih: Parameter = Parameter(Tensor.empty(gates, input_size, dtype=dtype))
        inplace_init(self.weight_ih, weight_init)
        self.weight_hh: Parameter = Parameter(Tensor.empty(gates, hidden_size, dtype=dtype))
        inplace_init(self.weight_hh, weight_init)
        self.bias_ih: Parameter | None = None
        self.bias_hh: Parameter | None = None
        if bias:
            self.bias_ih = Parameter(Tensor.empty(gates, dtype=dtype))
            inplace_init(self.bias_ih, bias_init)
            self.bias_hh = Parameter(Tensor.empty(gates, dtype=dtype))
            inplace_init(self.bias_hh, bias_init)

    def _zeros(self, x: Tensor) -> Tensor:
        return Tensor.zeros(x.shape[0], self.hidden_size, dtype=x.dtype)

    def extra_repr(self) -> str:
        return f'{self.input_size}, {self.hidden_size}, bias={self.bias_ih is not None}'


class LSTMCell(_RNNCellBase):
    _num_gates = 4

    def forward(self, x: Tensor, hx: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        unbatched = 1 == x.rank
        if unbatched:
            x = x.unsqueeze(0)
            if hx is not None:
                hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0))
        if hx is None:
            hx = (self._zeros(x), self._zeros(x))
        h, c = self._lstm_step(x, hx[0], hx[1], self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        if unbatched:
            h, c = h.squeeze(0), c.squeeze(0)
        return h, c


class _RNNBase(Module):
    _num_gates: int = 1
    _has_cell_state: bool = False

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        dtype: dtype.DType | None = None,
        weight_init: InitStrategy | None = None,
        bias_init: InitStrategy | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f'num_layers must be >= 1, got {num_layers}')
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f'dropout must be in [0, 1), got {dropout}')
        if dtype is None:
            dtype = context.get_default_dtype()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.bias: bool = bias
        self.batch_first: bool = batch_first
        self.dropout: float = dropout
        self.bidirectional: bool = bidirectional
        self.num_directions: int = 2 if bidirectional else 1
        self.drop: Dropout = Dropout(dropout)
        weight_init, bias_init = _RNNCellBase._rnn_init(hidden_size, weight_init, bias_init)
        gates = self._num_gates * hidden_size

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * self.num_directions
            for d in range(self.num_directions):
                sfx = self._suffix(layer, d)
                w_ih = Parameter(Tensor.empty(gates, in_size, dtype=dtype))
                w_hh = Parameter(Tensor.empty(gates, hidden_size, dtype=dtype))
                inplace_init(w_ih, weight_init)
                inplace_init(w_hh, weight_init)
                setattr(self, f'weight_ih{sfx}', w_ih)
                setattr(self, f'weight_hh{sfx}', w_hh)
                if bias:
                    b_ih = Parameter(Tensor.empty(gates, dtype=dtype))
                    b_hh = Parameter(Tensor.empty(gates, dtype=dtype))
                    inplace_init(b_ih, bias_init)
                    inplace_init(b_hh, bias_init)
                    setattr(self, f'bias_ih{sfx}', b_ih)
                    setattr(self, f'bias_hh{sfx}', b_hh)

    @staticmethod
    def _suffix(layer: int, direction: int) -> str:
        return f'_l{layer}' + ('_reverse' if direction == 1 else '')

    def _weights(self, layer: int, direction: int) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        sfx = self._suffix(layer, direction)
        return (
            getattr(self, f'weight_ih{sfx}'),
            getattr(self, f'weight_hh{sfx}'),
            getattr(self, f'bias_ih{sfx}', None),
            getattr(self, f'bias_hh{sfx}', None),
        )

    def _step(self, x: Tensor, state, weights):
        raise NotImplementedError

    def extra_repr(self) -> str:
        s = f'{self.input_size}, {self.hidden_size}'
        if self.num_layers != 1:
            s += f', num_layers={self.num_layers}'
        if not self.bias:
            s += ', bias=False'
        if self.batch_first:
            s += ', batch_first=True'
        if self.dropout != 0.0:
            s += f', dropout={self.dropout}'
        if self.bidirectional:
            s += ', bidirectional=True'
        return s

    def forward(self, x: Tensor, hx: Tensor | tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        unbatched = x.rank == 2
        if unbatched:
            x = x.unsqueeze(1)
            if hx is not None:
                hx = tuple(s.unsqueeze(1) for s in hx) if self._has_cell_state else hx.unsqueeze(1)
        elif self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch = x.shape[0], x.shape[1]
        if hx is None:
            zeros = Tensor.zeros(self.num_layers * self.num_directions, batch, self.hidden_size, dtype=x.dtype)
            hx = (zeros, zeros) if self._has_cell_state else zeros
        if self._has_cell_state:
            states = list(zip(hx[0].unbind(0), hx[1].unbind(0)))
        else:
            states = list(hx.unbind(0))

        layer_input = list(x.unbind(0))
        final_states = []
        for layer in range(self.num_layers):
            dir_outputs = []
            for d in range(self.num_directions):
                weights = self._weights(layer, d)
                state = states[layer * self.num_directions + d]
                outs = [None] * seq_len
                steps = range(seq_len - 1, -1, -1) if d == 1 else range(seq_len)
                for t in steps:
                    state = self._step(layer_input[t], state, weights)
                    outs[t] = state[0] if self._has_cell_state else state
                dir_outputs.append(outs)
                final_states.append(state)
            if self.num_directions == 1:
                layer_input = dir_outputs[0]
            else:
                layer_input = [Tensor.cat([f, b], dim=-1) for f, b in zip(*dir_outputs)]
            if layer < self.num_layers - 1:
                layer_input = [self.drop(y) for y in layer_input]

        output = Tensor.stack(layer_input, dim=0)
        if self._has_cell_state:
            h_n = Tensor.stack([s[0] for s in final_states], dim=0)
            c_n = Tensor.stack([s[1] for s in final_states], dim=0)
        else:
            h_n = Tensor.stack(final_states, dim=0)
        if unbatched:
            output = output.squeeze(1)
            h_n = h_n.squeeze(1)
            if self._has_cell_state:
                c_n = c_n.squeeze(1)
        elif self.batch_first:
            output = output.transpose(0, 1)
        return output, ((h_n, c_n) if self._has_cell_state else h_n)


class LSTM(_RNNBase):
    _num_gates = 4
    _has_cell_state = True

    def _step(self, x: Tensor, state: tuple[Tensor, Tensor], weights) -> tuple[Tensor, Tensor]:
        return _RNNCellBase._lstm_step(x, state[0], state[1], *weights)


class GRU(_RNNBase):
    _num_gates = 3

    def _step(self, x: Tensor, state: Tensor, weights) -> Tensor:
        return _RNNCellBase._gru_step(x, state, *weights)
