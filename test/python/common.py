# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import torch
import pytest
from magnetron import *
from collections import deque
from enum import Enum, unique

DTYPE_TORCH_MAP: dict[DataType, torch.dtype] = {
    float16: torch.float16,
    float32: torch.float32,
    int32: torch.int32,
    boolean: torch.bool
}

def totorch(obj: Tensor | int | float | bool, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not isinstance(obj, Tensor) and not isinstance(obj, torch.Tensor):
        return obj
    if dtype is None:
        dtype = DTYPE_TORCH_MAP[obj.dtype]
    return torch.tensor(obj.tolist(), dtype=dtype).reshape(obj.shape)

def square_shape_permutations(f: callable, lim: int) -> None:
    lim += 1
    for i0 in range(1, lim):
        for i1 in range(1, lim):
            for i2 in range(1, lim):
                for i3 in range(1, lim):
                    for i4 in range(1, lim):
                        for i5 in range(1, lim):
                            f((i0, i1, i2, i3, i4, i5))

@unique
class BinaryOpParamKind(Enum):
    TENSOR = 'tensor'
    SCALAR = 'scalar'
    LIST = 'list'

def _allocate_binary_op_args(dtype: DataType, shape: tuple[int, ...], kind: BinaryOpParamKind, low: float | int = 0, high: float | int = 1) -> tuple[Tensor, Tensor | list[Any] | float | int]:
    if dtype == boolean:
        x = Tensor.bernoulli(shape)
        match kind:
            case BinaryOpParamKind.TENSOR:
                return x, Tensor.bernoulli(shape)
            case BinaryOpParamKind.LIST:
                return x, [random.choice([True, False]) for _ in range(nested_len(list(shape)))]
            case BinaryOpParamKind.SCALAR:
                return x, random.choice([True, False])
            case _:
                raise ValueError(f'Unknown BinaryOpParamKind: {kind}')
    else:
        x = Tensor.uniform(shape, dtype=dtype, low=low, high=high)
        match kind:
            case BinaryOpParamKind.TENSOR:
                return x, Tensor.uniform(shape, dtype=dtype, low=low, high=high)
            case BinaryOpParamKind.LIST:
                if dtype.is_integer:
                    return x, [random.randint(low, high) for _ in range(nested_len(list(shape)))]
                else:
                    return x, [random.uniform(low, high) for _ in range(nested_len(list(shape)))]
            case BinaryOpParamKind.SCALAR:
                if dtype.is_integer:
                    return x, random.randint(low, high)
                else:
                    return x, random.uniform(low, high)
            case _:
                raise ValueError(f'Unknown BinaryOpParamKind: {kind}')

def binary_op_square(dtype: DataType, callback: Callable[[Tensor | torch.Tensor, Tensor | torch.Tensor], Tensor | torch.Tensor], lim: int = 4, kind: BinaryOpParamKind = BinaryOpParamKind.TENSOR, low: float | int = 0, high: float | int = 1) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x, y = _allocate_binary_op_args(dtype, shape, kind, low, high)
        r = callback(x, y)
        torch.testing.assert_close(
            totorch(r),
            callback(totorch(x), totorch(y))
        )

    square_shape_permutations(func, lim)

def binary_cmp_op(dtype: DataType, callback: Callable[[Tensor | torch.Tensor, Tensor | torch.Tensor], Tensor | torch.Tensor], lim: int = 4, kind: BinaryOpParamKind = BinaryOpParamKind.TENSOR, low: float | int = 0, high: float | int = 1) -> None:
    def func(shape: tuple[int, ...]) -> None:
        x, y = _allocate_binary_op_args(dtype, shape, kind, low, high)
        r = callback(x, y)
        assert r.dtype == boolean
        torch.testing.assert_close(
            totorch(r, torch.bool),
            callback(totorch(x), totorch(y))
        )

    square_shape_permutations(func, lim)

def unary_op(
    dtype: DataType,
    mag_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor],
    torch_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor],
    lim: int = 4,
    low: float | int | None = None,
    high: float | int | None = None
) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype, low=low, high=high)
        r = mag_callback(x.clone())
        torch.testing.assert_close(totorch(r), torch_callback(totorch(x)))

    square_shape_permutations(func, lim)


def scalar_op(dtype: DataType, callback: Callable[[Tensor | torch.Tensor, int | float | bool], Tensor | torch.Tensor], rhs: bool = True, lim: int = 4) -> None:
    def func(shape: tuple[int, ...]) -> None:  # x op scalar
        xi: float = random.uniform(-1.0, 1.0)
        x = Tensor.uniform(shape, dtype=dtype)
        r = callback(x, xi)
        torch.testing.assert_close(totorch(r), callback(totorch(x), xi))

    square_shape_permutations(func, lim)

    if not rhs:
        return

    def func(shape: tuple[int, ...]) -> None:  # scalar op x
        xi: float = random.uniform(-1.0, 1.0)
        x = Tensor.uniform(shape)
        r = callback(xi, x)
        torch.testing.assert_close(totorch(r), callback(xi, totorch(x)))

    square_shape_permutations(func, lim)

def nested_len(obj: list[Any]) -> int:
    total = 0
    stack = deque([obj])
    seen = set()
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        for item in current:
            if isinstance(item, list):
                stack.append(item)
            else:
                total += 1
    return total

def flatten(nested: Any) -> list[Any]:
    out: list[Any] = []
    stack = deque([iter(nested)])
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(item, list) or isinstance(item, tuple):
            stack.append(iter(item))
        else:
            out.append(item)
    return out
