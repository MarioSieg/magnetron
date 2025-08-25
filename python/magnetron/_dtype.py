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

from ._bootstrap import FFI, C

FLOATING_POINT_DTYPES: set[DataType] = set()  # Includes all floating-point types.
INTEGRAL_DTYPES: set[DataType] = set()  # Includes all integral types (integers + boolean).
INTEGER_DTYPES: set[DataType] = set()  # Include all integer types (integers - boolean).
NUMERIC_DTYPES: set[DataType] = set()  # Include all numeric dtypes (floating point + integers - boolean)


@dataclass(frozen=True)
class DataType:
    enum_value: int
    size: int
    name: str
    native_type: str | None
    fill_fn: FFI.CData

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


float32: DataType = DataType(C.MAG_DTYPE_E8M23, 4, 'float32', 'float', C.mag_tensor_fill_from_floats)
float16: DataType = DataType(C.MAG_DTYPE_E5M10, 2, 'float16', None, C.mag_tensor_fill_from_floats)
boolean: DataType = DataType(C.MAG_DTYPE_BOOL, 1, 'bool', 'bool', C.mag_tensor_fill_from_raw_bytes)
int32: DataType = DataType(C.MAG_DTYPE_I32, 4, 'int32', 'int32_t', C.mag_tensor_fill_from_raw_bytes)

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
