# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import itertools
import multiprocessing
import random
import pytest
from typing import Any, Iterator

import torch
import numpy as np
from magnetron import *
from collections import deque

# Give torch 1//4 of the total cores to not overload the CPU with inner parallelism threads + parallel spawned pytests
torch.set_num_threads(max(4, multiprocessing.cpu_count() // 8))
torch.set_num_interop_threads(max(4, multiprocessing.cpu_count() // 8))

DTYPE_TORCH_MAP: dict[DataType, torch.dtype] = {
    float16: torch.float16,
    float32: torch.float32,
    boolean: torch.bool,
    uint8: torch.uint8,
    int8: torch.int8,
    uint16: torch.uint16,
    int16: torch.int16,
    uint32: torch.uint32,
    int32: torch.int32,
    uint64: torch.uint64,
    int64: torch.int64,
}

NUMPY_DTYPE_MAP: dict[DataType, np.dtype] = {
    float16: np.float16,
    float32: np.float32,
    boolean: np.bool_,
    uint8: np.uint8,
    int8: np.int8,
    uint16: np.uint16,
    int16: np.int16,
    uint32: np.uint32,
    int32: np.int32,
    uint64: np.uint64,
    int64: np.int64,
}

def totorch_dtype(dtype: DataType) -> torch.dtype:
    if dtype not in DTYPE_TORCH_MAP:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return DTYPE_TORCH_MAP[dtype]

def tonumpy_dtype(dtype: DataType) -> np.dtype:
    if dtype not in NUMPY_DTYPE_MAP:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return NUMPY_DTYPE_MAP[dtype]


def totorch(obj: Tensor | int | float | bool, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype) if dtype is not None else obj
    if not isinstance(obj, Tensor):
        return torch.tensor(obj, dtype=dtype) if dtype is not None else torch.tensor(obj)
    if dtype is None:
        dtype = totorch_dtype(obj.dtype)
    t = torch.tensor(obj.tolist(), dtype=dtype)
    return t.reshape(obj.shape)

def tonumpy(obj: Tensor | int | float | bool, dtype: np.dtype | None = None) -> np.array:
    if isinstance(obj, np.ndarray):
        return obj.astype(dtype) if dtype is not None else obj
    if not isinstance(obj, Tensor):
        return np.array(obj, dtype=dtype) if dtype is not None else np.array(obj)
    if dtype is None:
        dtype = tonumpy_dtype(obj.dtype)
    a = np.array(obj.tolist(), dtype=dtype)
    return a.reshape(obj.shape)

def broadcastable(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    for x, y in zip(a[::-1], b[::-1]):
        if not (x == y or x == 1 or y == 1):
            return False
    return True


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    rev = []
    for x, y in zip(a[::-1], b[::-1]):
        rev.append(max(x, y))
    longer = a if len(a) > len(b) else b
    rev.extend(longer[: abs(len(a) - len(b))][::-1])
    return tuple(rev[::-1])


def iter_shapes(rank: int, lim: int) -> Iterator[tuple[int, ...]]:
    if rank == 0:
        yield ()
        return
    tup = [1] * rank
    while True:
        yield tuple(tup)
        i = rank - 1
        while i >= 0 and tup[i] == lim:
            tup[i] = 1
            i -= 1
        if i < 0:
            break
        tup[i] += 1

def matmul_shape_pairs(lim: int, max_total_rank: int = 6) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    max_batch_rank = max_total_rank - 2
    rng = range(1, lim + 1)
    for batch_rank in range(max_batch_rank + 1):
        for batch_a in iter_shapes(batch_rank, lim):
            for batch_b in iter_shapes(batch_rank, lim):
                if not broadcastable(batch_a, batch_b):
                    continue
                batched = broadcast_shape(batch_a, batch_b)
                for M in rng:
                    for K in rng:
                        for N in rng:
                            shape_A = (*batched, M, K)
                            shape_B = (*batched, K, N)
                            yield shape_A, shape_B

def random_tensor(shape: tuple[int, ...], dtype: DataType) -> Tensor:
    if dtype == boolean:
        return Tensor.bernoulli(shape)
    else:
        lim = 100 if dtype.is_integer else 1.0
        return Tensor.uniform(shape, low=-lim, high=lim, dtype=dtype)

DETAILED_TEST_SHAPES: tuple[tuple[int, ...], ...] = (
    (),
    (1,),
    (1, 1),
    (1, 1, 1),
    (1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (2,),
    (3,),
    (4,),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (1, 1, 2),
    (1, 2, 1),
    (2, 1, 1),
    (2, 3),
    (3, 2),
    (1, 2, 3),
    (3, 2, 1),
    (1, 3, 1, 3),
    (3, 5),
    (4, 7),
    (7, 4),
    (5, 3, 4),
    (4, 5, 3),
    (3, 4, 5),
    (1, 16),
    (16, 1),
    (1, 1, 16),
    (8, 1, 16),
    (1, 8, 16),
    (2, 3, 1, 16),
    (2, 1, 3, 16),
    (1, 2, 3, 1),
    (1, 2, 1, 3),
    (2, 1, 1, 3),
    (1, 32, 128),
    (32, 1, 128),
    (32, 128, 1),
    (16,),
    (32,),
    (64,),
    (128,),
    (256,),
    (512,),
    (1024,),
    (2048,),
    (4096,),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (1, 3, 224, 224),
    (1, 3, 512, 512),
    (8, 3, 32, 32),
    (4, 3, 256, 256),
    (64, 3, 7, 7),
    (128, 64, 3, 3),
    (256, 128, 1, 1),
    (512, 256, 3, 3),
    (1, 197, 768),
    (1, 577, 768),
    (1, 197, 1024),
    (1, 4096),
    (1, 5120),
    (1, 8192),
    (1, 12288),
    (16, 4096),
    (32, 4096),
    (8, 8192),
    (8, 12288),
    (32, 128),
    (40, 128),
    (64, 128),
    (32, 96),
    (1024, 3072),
    (2048, 2048),
    (1024, 1536),
    (1376, 1024),
    (1024, 1376),
    (512, 917),
    (768, 1536),
    (4, 32, 128, 64),
    (1, 40, 512, 48),
    (8, 64, 128, 16),
    (2, 32, 256, 64),
    (16, 128),
    (128, 128),
    (2048, 128),
    (3, 1, 7),
    (1, 3, 7),
    (7, 1, 3),
    (3, 7, 1),
    (1, 7, 3),
    (7, 3, 1),
    (2, 3, 5, 7),
    (7, 5, 3, 2),
    (17,),
    (31,),
    (67,),
    (127,),
    (257,),
    (2, 3, 4, 5, 6),
    (6, 5, 4, 3, 2),
    (1, 2, 3, 4, 5, 6),
    (1, 1, 1024, 1024),  # smaller
    (512, 512),
    (1024, 1024),
    (2048, 512),          # smaller
    (512, 2048),          # smaller
    (2048, 512),          # smaller
    (512, 2048),          # smaller
    (6, 66, 666),
    (4, 4, 256, 256),     # smaller
    (2, 8, 256, 256),     # smaller
    (5,),
    (7,),
    (9,),
    (11,),
    (13,),
    (2, 2),
    (2, 2, 2),
    (2, 2, 2, 2),
    (3, 3),
    (3, 3, 3),
    (4, 1, 4),
    (1, 4, 1, 4),
    (1, 2, 2, 1),
    (2, 1, 2, 1),
    (1, 1, 2, 2),
    (2, 2, 1, 1),
    (1, 3, 5),
    (5, 3, 1),
    (1, 5, 3),
    (3, 1, 5),
    (1, 1, 8),
    (8, 1, 1),
    (1, 8, 1),
    (1, 1, 1, 8),
    (8, 1, 1, 1),
    (1, 8, 1, 1),
    (1, 1, 8, 1),
    (1, 1, 16, 1),
    (1, 16, 1, 1),
    (1, 4, 1, 16),
    (4, 1, 16, 1),
    (1, 1, 4, 16),
    (1, 7, 1, 1, 7),
    (7, 1, 7, 1, 1),
    (1, 1, 7, 1, 7),
    (1, 3, 7, 7),
    (1, 3, 9, 9),
    (1, 3, 11, 11),
    (2, 3, 17, 17),
    (1, 1, 15, 15),
    (4, 3, 13, 13),
    (8, 3, 17, 17),
    (1, 16, 64),
    (2, 16, 64),
    (4, 16, 64),
    (1, 32, 64),
    (2, 32, 64),
    (4, 32, 64),
    (1, 32, 128),
    (2, 32, 128),
    (4, 32, 128),
    (1, 64, 64),
    (2, 64, 64),
    (4, 64, 64),
    (1, 64, 128),
    (2, 64, 128),
    (4, 64, 128),
    (1, 1, 16, 16),
    (1, 2, 16, 32),
    (2, 2, 16, 32),
    (1, 4, 32, 32),
    (2, 4, 32, 32),
    (4, 4, 32, 32),
    (1, 8, 64, 32),
    (2, 8, 64, 32),
    (1, 8, 32, 64),
    (2, 8, 32, 64),
    (7, 13),
    (13, 7),
    (15, 33),
    (33, 15),
    (31, 64),
    (64, 31),
    (63, 64),
    (64, 63),
    (127, 64),
    (64, 127),
    (255, 64),
    (64, 255),
    (16, 20),
    (20, 16),
    (8, 24),
    (24, 8),
    (7, 17),
    (17, 7),
    (5, 21),
    (21, 5),
    (4, 4, 16),
    (4, 8, 16),
    (8, 4, 16),
    (2, 8, 32),
    (2, 16, 32),
    (4, 8, 32),
    (2, 4, 8, 16),
    (2, 4, 16, 8),
    (4, 2, 8, 16),
    (1, 4, 8, 16),
    (1, 8, 4, 16),
    (1, 2, 3, 4, 5),
    (5, 4, 3, 2, 1),
    (1, 1, 2, 3, 4),
    (1, 2, 1, 3, 4),
    (2, 1, 3, 1, 4),
    (1, 2, 3, 1, 4),
    (1, 2, 3, 4, 1),
    (1, 2, 3, 4, 5, 1),
    (1, 1, 2, 3, 4, 5),
    (2, 3, 1, 4, 1, 5),
    (1, 16, 8, 8),
    (1, 8, 8, 16),
    (2, 16, 8, 8),
    (2, 8, 8, 16),
    (4, 16, 4, 4),
    (4, 4, 4, 16),
    (1, 1, 32),
    (32, 1, 1),
    (1, 32, 1),
    (2, 1, 32),
    (2, 32, 1),
    (1, 2, 32),
    (1, 2, 1, 32),
    (2, 1, 1, 32),
    (1, 2, 32, 1),
    (1, 15),
    (1, 31),
    (1, 63),
    (2, 63),
    (4, 63),
    (3, 21),
    (3, 63),
    (5, 19),
    (7, 8),
    (8, 7),
    (15, 16),
    (16, 15),
    (31, 32),
    (32, 31),
    (1, 2, 2, 2, 2, 2, 2),
    (2, 1, 2, 2, 2, 2, 2),
    (1, 3, 1, 3, 1, 3, 1),
    (2, 3, 4, 1, 2, 3, 4),
    (1, 2, 3, 4, 1, 2, 3, 4),
)

def for_all_shapes(f: Callable[tuple[int, ...]]) -> None:
    for shape in DETAILED_TEST_SHAPES: # We use small shapes for faster tests
        f(shape)

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

def random_dim(shape: tuple[int, ...]) -> int | None:
    if len(shape) == 0:
        return None
    elif len(shape) == 1:
        return 0
    else:
        return random.randrange(len(shape))

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
