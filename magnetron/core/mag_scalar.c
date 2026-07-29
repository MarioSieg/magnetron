/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "mag_def.h"

mag_scalar_t mag_scalar_from_float64(double value) {
  return (mag_scalar_t){.type = MAG_SCALAR_TYPE_F64, .value.float64 = value};
}

mag_scalar_t mag_scalar_from_int64(int64_t value) {
  return (mag_scalar_t){.type = MAG_SCALAR_TYPE_I64, .value.int64 = value};
}

mag_scalar_t mag_scalar_from_uint64(uint64_t value) {
  return (mag_scalar_t){.type = MAG_SCALAR_TYPE_U64, .value.uint64 = value};
}

bool mag_scalar_is_float64(mag_scalar_t s) { return s.type == MAG_SCALAR_TYPE_F64; }
bool mag_scalar_is_int64(mag_scalar_t s) { return s.type == MAG_SCALAR_TYPE_I64; }
bool mag_scalar_is_uint64(mag_scalar_t s) { return s.type == MAG_SCALAR_TYPE_U64; }

double mag_scalar_as_float64(mag_scalar_t s) {
  switch (s.type) {
    case MAG_SCALAR_TYPE_F64: return s.value.float64;
    case MAG_SCALAR_TYPE_I64: return (double)s.value.int64;
    case MAG_SCALAR_TYPE_U64: return (double)s.value.uint64;
    default: mag_panic("scalar: invalid type tag %d.", s.type);
  }
}

int64_t mag_scalar_as_int64(mag_scalar_t s) {
  switch (s.type) {
    case MAG_SCALAR_TYPE_I64: return s.value.int64;
    case MAG_SCALAR_TYPE_U64: return (int64_t)s.value.uint64;
    case MAG_SCALAR_TYPE_F64: return (int64_t)s.value.float64;
    default: mag_panic("scalar: invalid type tag %d.", s.type);
  }
}

uint64_t mag_scalar_as_uint64(mag_scalar_t s) {
  switch (s.type) {
    case MAG_SCALAR_TYPE_U64: return s.value.uint64;
    case MAG_SCALAR_TYPE_I64: return (uint64_t)s.value.int64;
    case MAG_SCALAR_TYPE_F64: return (uint64_t)s.value.float64;
    default: mag_panic("scalar: invalid type tag %d.", s.type);
  }
}

bool mag_scalar_same_type(mag_scalar_t a, mag_scalar_t b) {
  return a.type == b.type;
}

bool mag_scalar_same_type_and_value(mag_scalar_t a, mag_scalar_t b) {
  return mag_scalar_same_type(a, b) && a.value.uint64 == b.value.uint64;
}
