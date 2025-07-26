# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from typing import Any, Iterator

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

def totorch_dtype(dtype: DataType) -> torch.dtype:
    if dtype not in DTYPE_TORCH_MAP:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return DTYPE_TORCH_MAP[dtype]

def totorch(obj: Tensor | int | float | bool, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not isinstance(obj, Tensor) and not isinstance(obj, torch.Tensor):
        return obj
    if dtype is None:
        dtype = totorch_dtype(obj.dtype)
    return torch.tensor(obj.tolist(), dtype=dtype).reshape(obj.shape)

def broadcastable(a: tuple[int,...], b: tuple[int,...]) -> bool:
    for x, y in zip(a[::-1], b[::-1]):
        if not (x == y or x == 1 or y == 1):
            return False
    return True

def broadcast_shape(a: tuple[int,...], b: tuple[int,...]) -> tuple[int,...]:
    rev = []
    for x, y in zip(a[::-1], b[::-1]):
        rev.append(max(x, y))
    longer = a if len(a) > len(b) else b
    rev.extend(longer[:abs(len(a)-len(b))][::-1])
    return tuple(rev[::-1])

def matmul_shape_pairs(lim: int, max_total_rank: int = 6) -> Iterator[tuple[tuple[int,...], tuple[int,...]]]:
    max_batch_rank = max_total_rank-2
    for batch_rank in range(max_batch_rank+1):
        for batch_a in itertools.product(range(1, lim+1), repeat=batch_rank):
            for batch_b in itertools.product(range(1, lim+1), repeat=batch_rank):
                if not broadcastable(batch_a, batch_b):
                    continue
                batched = broadcast_shape(batch_a, batch_b)
                for M, K, N in itertools.product(range(1, lim+1), repeat=3):
                    shape_A = (*batched, M, K)
                    shape_B = (*batched, K, N)
                    yield shape_A, shape_B

def square_shape_permutations(f: callable, lim: int) -> None:
    lim += 1
    for shape in itertools.product(range(1, lim+1), repeat=MAX_DIMS):
        f(shape)

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
