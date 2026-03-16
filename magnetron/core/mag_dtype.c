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

#include "mag_context.h"
#include "mag_alloc.h"
#include "mag_tensor.h"
#include "mag_float16.h"
#include "mag_bfloat16.h"

#include <float.h>
#include <inttypes.h>

const mag_type_traits_t *mag_type_trait(mag_dtype_t type) {
    static const mag_type_traits_t infos[MAG_DTYPE__NUM] = {
        [MAG_DTYPE_FLOAT32] = {
            .name = "float32",
            .short_name = "f32",
            .size = sizeof(float),
            .alignment = __alignof(float),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_F64,
                .value = {.f64 = FLT_MIN}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_F64,
                .value = {.f64 = FLT_MAX}
            },
        },
        [MAG_DTYPE_FLOAT16] = {
            .name = "float16",
            .short_name = "f16",
            .size = sizeof(mag_float16_t),
            .alignment = __alignof(mag_float16_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_F64,
                .value = {.f64 = -65504.0}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_F64,
                .value = {.f64 = 65504.0}
            },
        },
        [MAG_DTYPE_BFLOAT16] = {
            .name = "bfloat16",
            .short_name = "bf16",
            .size = sizeof(mag_bfloat16_t),
            .alignment = __alignof(mag_bfloat16_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_F64,
                .value = {.f64 = -3.3895313892515355e38}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_F64,
                .value = {.f64 = 3.3895313892515355e38}
            },
        },
        [MAG_DTYPE_BOOLEAN] = {
            .name = "boolean",
            .short_name = "b8",
            .size = sizeof(uint8_t),
            .alignment = __alignof(uint8_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = 0}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = 1}
            },
        },
        [MAG_DTYPE_UINT8] = {
            .name = "uint8",
            .short_name = "u8",
            .size = sizeof(uint8_t),
            .alignment = __alignof(uint8_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = 0}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = UINT8_MAX}
            },
        },
        [MAG_DTYPE_INT8] = {
            .name = "int8",
            .short_name = "i8",
            .size = sizeof(int8_t),
            .alignment = __alignof(int8_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_I64,
                .value = {.i64 = INT8_MIN}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_I64,
                .value = {.i64 = INT8_MAX}
            },
        },
        [MAG_DTYPE_UINT16] = {
            .name = "uint16",
            .short_name = "u16",
            .size = sizeof(uint16_t),
            .alignment = __alignof(uint16_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = 0}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = UINT16_MAX}
            },
        },
        [MAG_DTYPE_INT16] = {
            .name = "int16",
            .short_name = "i16",
            .size = sizeof(int16_t),
            .alignment = __alignof(int16_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_I64,
                .value = {.i64 = INT16_MIN}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_I64,
                .value = {.i64 = INT16_MAX}
            },
        },
        [MAG_DTYPE_UINT32] = {
            .name = "uint32",
            .short_name = "u32",
            .size = sizeof(uint32_t),
            .alignment = __alignof(uint32_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = 0}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = UINT32_MAX}
            },
        },
        [MAG_DTYPE_INT32] = {
            .name = "int32",
            .short_name = "i32",
            .size = sizeof(int32_t),
            .alignment = __alignof(int32_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_I64,
                .value = {.i64 = INT32_MIN}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_I64,
                .value = {.i64 = INT32_MAX}
            },
        },
        [MAG_DTYPE_UINT64] = {
            .name = "uint64",
            .short_name = "u64",
            .size = sizeof(uint64_t),
            .alignment = __alignof(uint64_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = 0}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_U64,
                .value = {.u64 = UINT64_MAX}
            },
        },
        [MAG_DTYPE_INT64] = {
            .name = "int64",
            .short_name = "i64",
            .size = sizeof(int64_t),
            .alignment = __alignof(int64_t),
            .min_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_I64,
                .value = {.i64 = INT64_MIN}
            },
            .max_val = (mag_scalar_t) {
                .type = MAG_SCALAR_TYPE_I64,
                .value = {.i64 = INT64_MAX}
            },
        },
    };
    return &infos[type];
}

bool mag_type_category_is_floating_point(mag_dtype_t type) {
    return mag_dtype_bit(type) & MAG_DTYPE_MASK_FP;
}

bool mag_type_category_is_unsigned_integer(mag_dtype_t type) {
    return mag_dtype_bit(type) & MAG_DTYPE_MASK_UINT;
}

bool mag_type_category_is_signed_integer(mag_dtype_t type) {
    return mag_dtype_bit(type) & MAG_DTYPE_MASK_SINT;
}

bool mag_type_category_is_integer(mag_dtype_t type) {
    return mag_dtype_bit(type) & MAG_DTYPE_MASK_INTEGER;
}

bool mag_type_category_is_integral(mag_dtype_t type) {
    return mag_dtype_bit(type) & MAG_DTYPE_MASK_INTEGRAL;
}

bool mag_type_category_is_numeric(mag_dtype_t type) {
    return mag_dtype_bit(type) & MAG_DTYPE_MASK_NUMERIC;
}

static const mag_dtype_t mag_type_promotion_rules[MAG_DTYPE__NUM][MAG_DTYPE__NUM] = {
    [MAG_DTYPE_FLOAT32] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_FLOAT32,
    },
    [MAG_DTYPE_FLOAT16] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_FLOAT32,
    },
    [MAG_DTYPE_BFLOAT16] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_BFLOAT16]= MAG_DTYPE_BFLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_BFLOAT16,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_BFLOAT16,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_BFLOAT16,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_BFLOAT16,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_BFLOAT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_FLOAT32,
    },
    [MAG_DTYPE_BOOLEAN] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_BOOLEAN,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT8,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT8,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT16,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_UINT8] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_UINT8,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT8,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT16,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_INT8] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_INT8,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT8,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_INT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_UINT16] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_UINT16,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT16,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT16,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_INT16] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_INT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_UINT32] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_UINT32,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_INT32] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_INT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_UINT64] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_UINT64,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_INT64] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
};

bool mag_promote_type(mag_dtype_t *out, mag_dtype_t lhs, mag_dtype_t rhs) {
    if (mag_unlikely(lhs >= MAG_DTYPE__NUM || rhs >= MAG_DTYPE__NUM)) return false;
    *out = mag_type_promotion_rules[lhs][rhs];
    return *out < MAG_DTYPE__NUM;
}