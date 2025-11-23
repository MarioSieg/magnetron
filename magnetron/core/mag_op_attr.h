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

#ifndef MAG_OP_ATTR_H
#define MAG_OP_ATTR_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_MAX_OP_PARAMS 8 /* Maximum number of parameters for an operation */

typedef enum mag_op_attr_type_tag_t {
    MAG_OP_ATTR_TYPE_EMPTY,
    MAG_OP_ATTR_TYPE_BOOL,
    MAG_OP_ATTR_TYPE_U64,
    MAG_OP_ATTR_TYPE_I64,
    MAG_OP_ATTR_TYPE_E8M23,
    MAG_OP_ATTR_TYPE_E5M10,
    MAG_OP_ATTR_TYPE_PTR,
} mag_op_attr_type_tag_t;

typedef struct mag_op_attr_t {
    mag_op_attr_type_tag_t tag;
    union {
        bool b;
        uint64_t u64;
        int64_t i64;
        mag_e8m23_t e8m23;
        mag_e5m10_t e5m10;
        void *ptr;
    } value;
} mag_op_attr_t;
mag_static_assert(sizeof(mag_op_attr_t) == 1+3+4 + 8);

static inline bool mag_op_attr_is_type(mag_op_attr_t opt, mag_op_attr_type_tag_t type) { return opt.tag == type; }
static inline bool mag_op_attr_is_empty(mag_op_attr_t opt) { return mag_op_attr_is_type(opt, MAG_OP_ATTR_TYPE_EMPTY); }

static inline mag_op_attr_t mag_op_attr_empty(void) { return (mag_op_attr_t){.tag=MAG_OP_ATTR_TYPE_EMPTY, .value={.i64=0}}; }
static inline mag_op_attr_t mag_op_attr_bool(bool v) { return (mag_op_attr_t){.tag=MAG_OP_ATTR_TYPE_BOOL, .value={.b=v}}; }
static inline mag_op_attr_t mag_op_attr_u64(uint64_t v) { return (mag_op_attr_t){.tag=MAG_OP_ATTR_TYPE_U64, .value={.u64=v}}; }
static inline mag_op_attr_t mag_op_attr_i64(int64_t v) { return (mag_op_attr_t){.tag=MAG_OP_ATTR_TYPE_I64, .value={.i64=v}}; }
static inline mag_op_attr_t mag_op_attr_e8m23(mag_e8m23_t v) { return (mag_op_attr_t){.tag=MAG_OP_ATTR_TYPE_E8M23, .value={.e8m23=v}}; }
static inline mag_op_attr_t mag_op_attr_e5m10(mag_e5m10_t v) { return (mag_op_attr_t){.tag=MAG_OP_ATTR_TYPE_E5M10, .value={.e5m10=v}}; }
static inline mag_op_attr_t mag_op_attr_ptr(void *v) { return (mag_op_attr_t){.tag=MAG_OP_ATTR_TYPE_PTR, .value={.ptr=v}}; }

#define mag_check_tag(T) mag_assert(opt.tag==MAG_OP_ATTR_TYPE_##T, "Op attribute stores wrong type code: %d", opt.tag);
static inline bool mag_op_attr_unwrap_bool(mag_op_attr_t opt) { mag_check_tag(BOOL);  return opt.value.b; }
static inline uint64_t mag_op_attr_unwrap_u64(mag_op_attr_t opt) { mag_check_tag(U64);return opt.value.u64; }
static inline int64_t mag_op_attr_unwrap_i64(mag_op_attr_t opt) { mag_check_tag(I64); return opt.value.i64; }
static inline mag_e8m23_t mag_op_attr_unwrap_e8m23(mag_op_attr_t opt) { mag_check_tag(E8M23); return opt.value.e8m23; }
static inline mag_e5m10_t mag_op_attr_unwrap_e5m10(mag_op_attr_t opt) { mag_check_tag(E5M10); return opt.value.e5m10; }
static inline void *mag_op_attr_unwrap_ptr(mag_op_attr_t opt) { mag_check_tag(PTR); return opt.value.ptr; }
#undef mag_check_tag

/* Helper for filling the operation parameters array and validating the amount. */
typedef struct mag_op_attr_registry_t {
    mag_op_attr_t slots[MAG_MAX_OP_PARAMS];
    uint32_t count;
} mag_op_attr_registry_t;

static inline void mag_op_attr_registry_init(mag_op_attr_registry_t *set) {
    set->count = 0;
    for (int i=0; i < MAG_MAX_OP_PARAMS; ++i)
        set->slots[i] = mag_op_attr_empty();
}

static inline size_t mag_op_attr_registry_insert(mag_op_attr_registry_t *set, mag_op_attr_t param) {
    mag_assert(set->count < MAG_MAX_OP_PARAMS, "Too many operation parameters");
    set->slots[set->count] = param;
    return set->count++;
}

static inline void mag_op_attr_registry_store(mag_op_attr_registry_t *set, size_t i, mag_op_attr_t param) {
    mag_assert(i < set->count, "Invalid operation parameter index: #%d", i);
    mag_assert(mag_op_attr_is_empty(set->slots[i]), "Operation parameter at #%d already set", i);
    set->slots[i] = param;
}

static inline void mag_op_attr_registry_transfer(const mag_op_attr_registry_t *set, mag_op_attr_t (*out)[MAG_MAX_OP_PARAMS]) {
    memcpy(*out, set->slots, set->count*sizeof(*set->slots));
    for (size_t i=set->count; i < MAG_MAX_OP_PARAMS; ++i)
        (*out)[i] = mag_op_attr_empty();
}

#ifdef __cplusplus
}
#endif

#endif
