/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#ifndef MAG_OPPARAM_H
#define MAG_OPPARAM_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Operation parameter type tag. */
typedef enum mag_opparam_type_t {
    MAG_OPP_NONE  = 0,
    MAG_OPP_E8M23 = 1,    /* fp32 */
    MAG_OPP_I64 = 2,    /* 64-bit signed integer. */

    MAG_OPP__NUM
} mag_opparam_type_t;
mag_static_assert(((MAG_OPP__NUM-1) & ~3) == 0); /* Must fit in 2 bits */

/*
** The opp (Operation Parameter) is used to pass additional data to the operation. For example:
**      The FILL init op uses the opp to pass the value to fill the tensor buffer with.
**      The permute op receives all permutation axes in the operation parameters.
** The opp is a tagged union (variant).
*/
typedef struct mag_opparam_t {
    uint8_t type : 2;
    int64_t i62 : 62;
} mag_opparam_t;

static MAG_AINLINE mag_opparam_t mag_op_param_none() {
    mag_opparam_t p;
    p.type = MAG_OPP_NONE;
    p.i62 = 0;
    return p;
}

static MAG_AINLINE mag_opparam_t mag_op_param_wrap_e8m23(mag_e8m23_t x) {
    uint32_t u32;
    memcpy(&u32, &x, sizeof(u32));
    mag_opparam_t p;
    p.type = MAG_OPP_E8M23;
    p.i62 = u32;
    return p;
}

static MAG_AINLINE mag_opparam_t mag_op_param_wrap_i64(int64_t x) {
    mag_assert(!((x < 0 ? ~x+1 : x) & ~((1ULL<<62)-1)), "op param int64 value must fit in 62 bit, but is: %" PRIi64, x);
    mag_opparam_t p;
    p.type = MAG_OPP_I64;
    p.i62 = x;
    return p;
}

/* Unpack value from packed opp. Panics if the type is not the expected type. */
static MAG_AINLINE mag_e8m23_t mag_op_param_unpack_e8m23_or_panic(mag_opparam_t pa) {
    mag_assert(pa.type == MAG_OPP_E8M23, "invalid op param type: %d", pa.type);
    mag_e8m23_t e8m23 = 0.f;
    uint32_t u32 = (uint32_t)pa.i62;
    memcpy(&e8m23, &u32, sizeof(e8m23));
    return e8m23;
}

static MAG_AINLINE int64_t mag_op_param_unpack_i64_or_panic(mag_opparam_t pa) {
    mag_assert(pa.type == MAG_OPP_I64, "invalid op param type: %d", pa.type);
    return pa.i62;
}

static MAG_AINLINE mag_e8m23_t mag_op_param_unpack_e8m23_or(mag_opparam_t pa, mag_e8m23_t fallback) {
    return pa.type == MAG_OPP_E8M23 ? mag_op_param_unpack_e8m23_or_panic(pa) : fallback;
}

static MAG_AINLINE int64_t mag_op_param_unpack_i64_or(mag_opparam_t pa, int64_t fallback) {
    return pa.type == MAG_OPP_I64 ? mag_op_param_unpack_i64_or_panic(pa) : fallback;
}

/* Helper for filling the operation parameters array and validating the amount. */
typedef struct mag_opparam_layout_t {
    mag_opparam_t slots[MAG_MAX_OP_PARAMS];
    uint32_t count;
} mag_op_param_layout_t;

static inline void mag_op_param_layout_init(mag_op_param_layout_t *set) {
    set->count = 0;
    for (int i=0; i < MAG_MAX_OP_PARAMS; ++i)
        set->slots[i] = mag_op_param_none();
}

static inline size_t mag_op_param_layout_insert(mag_op_param_layout_t *set, mag_opparam_t param) {
    mag_assert(set->count < MAG_MAX_OP_PARAMS, "Too many operation parameters");
    set->slots[set->count] = param;
    return set->count++;
}

static inline void mag_op_param_layout_store(mag_op_param_layout_t *set, size_t idx, mag_opparam_t param) {
    mag_assert(idx < set->count, "Invalid operation parameter index");
    mag_assert(set->slots[idx].type == MAG_OPP_NONE, "Operation parameter already set");
    set->slots[idx] = param;
}

static inline void mag_op_param_layout_transfer(const mag_op_param_layout_t *set, mag_opparam_t (*out)[MAG_MAX_OP_PARAMS]) {
    memcpy(*out, set->slots, set->count*sizeof(*set->slots));
    for (size_t i=set->count; i < MAG_MAX_OP_PARAMS; ++i)
        (*out)[i] = mag_op_param_none();
}

#ifdef __cplusplus
}
#endif

#endif
