# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

# This file implement a PyTorch-like functional compatibility API over magnetron.

from __future__ import annotations

from ._context import default_dtype
from ._core import *
from ._tensor import Tensor, NestedList


def tensor(
    data: NestedList,
    dtype: DataType | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.of(data, dtype=dtype, requires_grad=requires_grad)


def empty(
    *shape: int | tuple[int, ...],
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.empty(*shape, dtype=dtype, requires_grad=requires_grad)


def empty_like(
    template: Tensor,
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.empty_like(template, dtype=dtype, requires_grad=requires_grad)


def full(
    *shape: int | tuple[int, ...],
    fill_value: int | float | bool,
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.full(*shape, fill_value=fill_value, dtype=dtype, requires_grad=requires_grad)


def full_like(
    template: Tensor,
    *,
    fill_value: int | float,
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.full_like(template, fill_value=fill_value, dtype=dtype, requires_grad=requires_grad)


def zeros(
    *shape: int | tuple[int, ...],
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.zeros(*shape, dtype=dtype, requires_grad=requires_grad)


def zeros_like(
    template: Tensor,
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.zeros_like(template, dtype=dtype, requires_grad=requires_grad)


def ones(
    *shape: int | tuple[int, ...],
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.ones(*shape, dtype=dtype, requires_grad=requires_grad)


def ones_like(
    template: Tensor,
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.ones_like(template, dtype=dtype, requires_grad=requires_grad)


def rand(
    *shape: int | tuple[int, ...],
    low: float | int | None = None,
    high: float | int | None = None,
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.uniform(*shape, low=low, high=high, dtype=dtype, requires_grad=requires_grad)


def randn(
    *shape: int | tuple[int, ...],
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.normal(*shape, mean=0, std=1, dtype=dtype, requires_grad=requires_grad)


def normal(
    *shape: int | tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: DataType = default_dtype(),
    requires_grad: bool = False,
) -> Tensor:
    return Tensor.normal(*shape, mean=mean, std=std, dtype=dtype, requires_grad=requires_grad)


def bernoulli(
    *shape: int | tuple[int, ...],
    p: float = 0.5,
) -> Tensor:
    return Tensor.bernoulli(*shape, p=p)


def clone(x: Tensor) -> Tensor:
    return x.clone()


def detach(x: Tensor) -> Tensor:
    return x.detach()


def view(x: Tensor, *dims: int | tuple[int, ...]) -> Tensor:
    return x.view(*dims)


def reshape(x: Tensor, *dims: int | tuple[int, ...]) -> Tensor:
    return x.reshape(*dims)


def transpose(x: Tensor, dim1: int = 0, dim2: int = 1) -> Tensor:
    return x.transpose(dim1, dim2)


def permute(x: Tensor, *dims: int | tuple[int, ...]) -> Tensor:
    return x.permute(*dims)


def mean(x: Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
    return x.mean(dim, keepdim)


def min(x: Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
    return x.min(dim, keepdim)


def max(x: Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
    return x.max(dim, keepdim)


def sum(x: Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
    return x.sum(dim, keepdim)


def argmin(x: Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
    return x.argmin(dim, keepdim)


def argmax(x: Tensor, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
    return x.mean(dim, keepdim)


def abs(x: Tensor) -> Tensor:
    return x.abs()


def neg(x: Tensor) -> Tensor:
    return x.neg()


def log(x: Tensor) -> Tensor:
    return x.log()


def sqr(x: Tensor) -> Tensor:
    return x.sqr()


def sqrt(x: Tensor) -> Tensor:
    return x.sqrt()


def sin(x: Tensor) -> Tensor:
    return x.sin()


def cos(x: Tensor) -> Tensor:
    return x.cos()


def step(x: Tensor) -> Tensor:
    return x.step()


def exp(x: Tensor) -> Tensor:
    return x.exp()


def floor(x: Tensor) -> Tensor:
    return x.floor()


def ceil(x: Tensor) -> Tensor:
    return x.ceil()


def round(x: Tensor) -> Tensor:
    return x.round()


def softmax(x: Tensor) -> Tensor:
    return x.softmax()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def hardsigmoid(x: Tensor) -> Tensor:
    return x.hardsigmoid()


def silu(x: Tensor) -> Tensor:
    return x.silu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def tril(x: Tensor) -> Tensor:
    return x.tril()


def triu(x: Tensor) -> Tensor:
    return x.triu()


def add(x: Tensor, y: Tensor) -> Tensor:
    return x + y


def sub(x: Tensor, y: Tensor) -> Tensor:
    return x - y


def mul(x: Tensor, y: Tensor) -> Tensor:
    return x * y


def div(x: Tensor, y: Tensor) -> Tensor:
    return x / y


def floordiv(x: Tensor, y: Tensor) -> Tensor:
    return x // y


def matmul(x: Tensor, y: Tensor) -> Tensor:
    return x @ y


def logical_and(x: Tensor, y: Tensor) -> Tensor:
    return x.logical_and(y)


def logical_or(x: Tensor, y: Tensor) -> Tensor:
    return x.logical_or(y)


def logical_xor(x: Tensor, y: Tensor) -> Tensor:
    return x.logical_xor(y)


def logical_not(x: Tensor) -> Tensor:
    return x.logical_not()


def bitwise_and(x: Tensor, y: Tensor) -> Tensor:
    return x.bitwise_and(y)


def bitwise_or(x: Tensor, y: Tensor) -> Tensor:
    return x.bitwise_or(y)


def bitwise_xor(x: Tensor, y: Tensor) -> Tensor:
    return x.bitwise_xor(y)


def bitwise_not(x: Tensor) -> Tensor:
    return x.bitwise_not()


def bitwise_shl(x: Tensor, y: Tensor) -> Tensor:
    return x << y


def bitwise_shr(x: Tensor, y: Tensor) -> Tensor:
    return x >> y
