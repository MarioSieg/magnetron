# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import math

from .. import Tensor, no_grad
from dataclasses import dataclass
from abc import ABC
from enum import Enum, unique


@unique
class FanMode(Enum):
    FAN_IN = 'fan_in'
    FAN_OUT = 'fan_out'


@unique
class Activation(Enum):
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'

    def compute_gain(self, param: float | int | None) -> float:
        match self:
            case self.SIGMOID:
                return 1
            case self.TANH:
                return 5.0 / 3
            case self.RELU:
                return math.sqrt(2.0)
            case self.LEAKY_RELU:
                if param is None:
                    negative_slope = 0.01
                elif not isinstance(param, bool) and isinstance(param, int | float):
                    negative_slope = float(param)
                else:
                    raise ValueError(f'Negative slope param must be a number {param}')
                return math.sqrt(2.0 / (1 + negative_slope**2))
            case _:
                raise ValueError(f'Invalid activation {self}')


@dataclass(slots=True)
class InitStrategy(ABC):
    pass


@dataclass(slots=True)
class EmptyInitStrategy(InitStrategy):
    pass


@dataclass(slots=True)
class ConstantInitStrategy(InitStrategy):
    value: float


@dataclass(slots=True)
class ZerosInitStrategy(InitStrategy):
    pass


@dataclass(slots=True)
class OnesInitStrategy(InitStrategy):
    pass


@dataclass(slots=True)
class UniformInitStrategy(InitStrategy):
    low: float
    high: float


@dataclass(slots=True)
class NormalInitStrategy(InitStrategy):
    mean: float = 0.0
    std: float = 1.0


@dataclass(slots=True)
class XavierUniformInitStrategy(InitStrategy):
    gain: float = 1.0


@dataclass(slots=True)
class XavierNormalInitStrategy(InitStrategy):
    gain: float = 1.0


@dataclass(slots=True)
class KaimingUniformInitStrategy(InitStrategy):
    a: float = 0.0
    mode: FanMode = FanMode.FAN_IN
    activation: Activation = Activation.RELU


@dataclass(slots=True)
class KaimingNormalInitStrategy(InitStrategy):
    a: float = 0.0
    mode: FanMode = FanMode.FAN_IN
    activation: Activation = Activation.RELU


def compute_fan_inout(x: Tensor) -> tuple[int, int]:
    rank: int = x.rank
    if rank < 2:
        raise ValueError(f'Fan in and out can not be computed for tensors with fewer than 2 dims')
    if rank == 2:
        fan_in = x.shape[1]
        fan_out = x.shape[0]
    else:
        num_in_fmaps = x.shape[1]
        num_out_fmaps = x.shape[0]
        receptive_field_size = 1
        if rank > 2:
            receptive_field_size = x[0][0].numel()
        fan_in = num_in_fmaps * receptive_field_size
        fan_out = num_out_fmaps * receptive_field_size
    return fan_in, fan_out


def _select_fan(w: Tensor, mode: FanMode) -> int:
    fan_in, fan_out = compute_fan_inout(w)
    match mode:
        case FanMode.FAN_IN:
            return fan_in
        case FanMode.FAN_OUT:
            return fan_out
        case _:
            raise ValueError(f'Invalid fan mode {mode!r}')


@no_grad()
def inplace_init(w: Tensor, init: InitStrategy) -> None:
    match init:
        case EmptyInitStrategy():
            return
        case ConstantInitStrategy(value=v):
            w.fill_(v)
        case ZerosInitStrategy():
            w.zero_()
        case OnesInitStrategy():
            w.one_()
        case UniformInitStrategy(low=a, high=b):
            w.uniform_(a, b)
        case NormalInitStrategy(mean=m, std=s):
            w.normal_(m, s)
        case XavierUniformInitStrategy(gain=g):
            fan_in, fan_out = compute_fan_inout(w)
            std = g * math.sqrt(2.0 / float(fan_in + fan_out))
            bound = math.sqrt(3.0) * std
            w.uniform_(-bound, bound)
        case XavierNormalInitStrategy(gain=g):
            fan_in, fan_out = compute_fan_inout(w)
            std = g * math.sqrt(2.0 / float(fan_in + fan_out))
            w.normal_(0.0, std)
        case KaimingUniformInitStrategy(a=a, mode=mode, activation=activation):
            fan = _select_fan(w, mode)
            gain = activation.compute_gain(a)
            std = gain / math.sqrt(float(fan))
            bound = math.sqrt(3.0) * std
            w.uniform_(-bound, bound)
        case KaimingNormalInitStrategy(a=a, mode=mode, activation=activation):
            fan = _select_fan(w, mode)
            gain = activation.compute_gain(a)
            std = gain / math.sqrt(float(fan))
            w.normal_(0.0, std)
        case _:
            raise TypeError(f'Invalid initializer {init!r}')
