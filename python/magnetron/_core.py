# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from enum import Enum, unique
from os import getenv

from ._bootstrap import load_native_module

_ffi, _C = load_native_module()

MAX_DIMS: int = 6
DIM_MAX: int = (1 << 63) - 1  # INT64_MAX


class ComputeDevice:
    class CPU:
        def __init__(self, num_threads: int = 0) -> None:
            self.num_threads = num_threads

    class CUDA:
        def __init__(self, device_id: int = 0) -> None:
            self.device_id = device_id


@unique
class PRNGAlgorithm(Enum):
    MERSENNE_TWISTER = _C.MAG_PRNG_MERSENNE_TWISTER
    PCG = _C.MAG_PRNG_PCG


# Includes all floating-point types.
FLOATING_POINT_DTYPES: set[DataType] = set()

# Includes all integral types (integers + boolean).
INTEGRAL_DTYPES: set[DataType] = set()

# Include all integer types (integers - boolean).
INTEGER_DTYPES: set[DataType] = set()

# Include all numeric dtypes (floating point + integers - boolean)
NUMERIC_DTYPES: set[DataType] = set()


@dataclass(frozen=True)
class DataType:
    enum_value: int
    size: int
    name: str
    native_type: str | None
    fill_fn: _ffi.CData

    @property
    def is_floating_point(self) -> bool:
        return self in FLOATING_POINT_DTYPES

    @property
    def is_integral(self) -> bool:
        return self in INTEGRAL_DTYPES

    @property
    def is_integer(self) -> bool:
        return self in INTEGER_DTYPES

    @property
    def is_numeric(self) -> bool:
        return self in NUMERIC_DTYPES

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


float32: DataType = DataType(_C.MAG_DTYPE_E8M23, 4, 'float32', 'float', _C.mag_tensor_fill_from_floats)
float16: DataType = DataType(_C.MAG_DTYPE_E5M10, 2, 'float16', None, _C.mag_tensor_fill_from_floats)
boolean: DataType = DataType(_C.MAG_DTYPE_BOOL, 1, 'bool', 'bool', _C.mag_tensor_fill_from_raw_bytes)
int32: DataType = DataType(_C.MAG_DTYPE_I32, 4, 'int32', 'int32_t', _C.mag_tensor_fill_from_raw_bytes)

DTYPE_ENUM_MAP: dict[int, DataType] = {
    float32.enum_value: float32,
    float16.enum_value: float16,
    boolean.enum_value: boolean,
    int32.enum_value: int32,
}
FLOATING_POINT_DTYPES = {float32, float16}
INTEGRAL_DTYPES = {boolean, int32}
INTEGER_DTYPES = INTEGRAL_DTYPES - {boolean}
NUMERIC_DTYPES = FLOATING_POINT_DTYPES | INTEGER_DTYPES


@dataclass
class Config:
    verbose: bool = getenv('MAGNETRON_VERBOSE', '0') == '1'
    compute_device: ComputeDevice.CPU | ComputeDevice.CUDA = ComputeDevice.CPU()
    default_dtype: DataType = float32


NestedList = float | bool | int | list['NestedData']


def flatten_nested_lists(nested: list[Any]) -> tuple[tuple[int], list[Any]]:
    flat, dims = [], []

    def walk(node: list[Any], depth: int = 0) -> None:
        if isinstance(node, list):
            if len(dims) <= depth:
                dims.append(len(node))
            elif dims[depth] is None or dims[depth] != len(node):
                raise ValueError('All sub-lists must have the same shape')
            for child in node:
                walk(child, depth + 1)
        else:
            if len(dims) <= depth:
                dims.append(None)
            elif dims[depth] is not None:
                raise ValueError('All sub-lists must have the same shape')
            flat.append(node)

    walk(nested)
    return tuple(d for d in dims if d is not None), flat


def unpack_shape(*shape: int | tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], tuple):
        return shape[0]
    return shape


def build_nested_lists(flat: list[Any], shape: tuple[int], strides: tuple[int], offset: int, dim: int) -> list[Any, ...]:
    if dim == len(shape):
        return flat[offset]
    size = shape[dim]
    stride = strides[dim]
    return [build_nested_lists(flat, shape, strides, offset + i * stride, dim + 1) for i in range(size)]
