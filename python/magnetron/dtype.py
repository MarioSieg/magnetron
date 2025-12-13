# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations
from dataclasses import dataclass

from ._bootstrap import _FFI, _C

FLOATING_POINT_DTYPES: set[DataType] = set()  # Includes all floating-point types.
INTEGRAL_DTYPES: set[DataType] = set()  # Includes all integral types (integers + boolean).
INTEGER_DTYPES: set[DataType] = set()  # Include all integer types (integers - boolean).
NUMERIC_DTYPES: set[DataType] = set()  # Include all numeric dtypes (floating point + integers - boolean)
ALL_DTYPES: set[DataType] = set()

@dataclass(frozen=True)
class DataType:
    enum_value: int
    size: int
    name: str

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
    def is_signed_integer(self) -> bool:
        return self in SIGNED_INTEGER_DTYPES

    @property
    def is_unsigned_integer(self) -> bool:
        return self in UNSIGNED_INTEGER_DTYPES

    @property
    def is_numeric(self) -> bool:
        return self in NUMERIC_DTYPES

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


float32: DataType = DataType(_C.MAG_DTYPE_FLOAT32, 4, 'float32')
float16: DataType = DataType(_C.MAG_DTYPE_FLOAT16, 2, 'float16')
boolean: DataType = DataType(_C.MAG_DTYPE_BOOLEAN, 1, 'uint8')
uint8: DataType = DataType(_C.MAG_DTYPE_UINT8, 1, 'uint8')
int8: DataType = DataType(_C.MAG_DTYPE_INT8, 1, 'int8')
uint16: DataType = DataType(_C.MAG_DTYPE_UINT16, 2, 'uint16')
int16: DataType = DataType(_C.MAG_DTYPE_INT16, 2, 'int16')
uint32: DataType = DataType(_C.MAG_DTYPE_UINT32, 4, 'uint32')
int32: DataType = DataType(_C.MAG_DTYPE_INT32, 4, 'int32')
uint64: DataType = DataType(_C.MAG_DTYPE_UINT64, 8, 'uint64')
int64: DataType = DataType(_C.MAG_DTYPE_INT64, 8, 'int64')

DTYPE_ENUM_MAP: dict[int, DataType] = {
    float32.enum_value: float32,
    float16.enum_value: float16,
    boolean.enum_value: boolean,
    uint8.enum_value: uint8,
    int8.enum_value: int8,
    uint16.enum_value: uint16,
    int16.enum_value: int16,
    uint32.enum_value: uint32,
    int32.enum_value: int32,
    uint64.enum_value: uint64,
    int64.enum_value: int64,
}
FLOATING_POINT_DTYPES = {float32, float16}
UNSIGNED_INTEGER_DTYPES = {uint8, uint16, uint32, uint64}
SIGNED_INTEGER_DTYPES = {int8, int16, int32, int64}
INTEGER_DTYPES = UNSIGNED_INTEGER_DTYPES | SIGNED_INTEGER_DTYPES
NUMERIC_DTYPES = FLOATING_POINT_DTYPES | INTEGER_DTYPES
INTEGRAL_DTYPES = INTEGER_DTYPES | {boolean}
ALL_DTYPES = FLOATING_POINT_DTYPES | INTEGRAL_DTYPES
