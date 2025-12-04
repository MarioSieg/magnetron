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

#ifndef MAG_RC_H
#define MAG_RC_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Header for all objects that are reference counted. */
typedef struct mag_rc_control_block_t {
    #ifdef MAG_DEBUG
        uint32_t __sentinel;
    #endif
    volatile mag_atomic32_t rc_strong; /* Strong atomic RC */
    void (*dtor)(void *); /* Destructor (required). */
} mag_rc_control_block_t;

#ifdef MAG_DEBUG
mag_static_assert(offsetof(mag_rc_control_block_t, __sentinel) == 0);
#endif

#define MAG_RC_INJECT_HEADER mag_rc_control_block_t __rcb
#define MAG_RC_OBJECT_IS_VALID(T) mag_static_assert(offsetof(T, __rcb) == 0)

/* Initialize reference count header for a new object. Object must have MAG_RC_INJECT_HEADER as first field. */
static inline void mag_rc_init_object(void *obj, void (*dtor)(void *)) {
    mag_rc_control_block_t *rc = (mag_rc_control_block_t *)obj;
    mag_atomic32_store(&rc->rc_strong, 1, MAG_MO_RELAXED);
    rc->dtor = dtor;
    #ifdef MAG_DEBUG
        rc->__sentinel = 0xDEADBEEF;
    #endif
}

/* Increment reference count (retain). Object must have MAG_RC_INJECT_HEADER as first field. */
static MAG_AINLINE void mag_rc_incref(void *obj) {
    mag_rc_control_block_t *rc = (mag_rc_control_block_t *)obj;
    #ifdef MAG_DEBUG /* Verify that object has a valid control block header using the sentinel */
        mag_assert2(rc->__sentinel == 0xDEADBEEF);
    #endif
    mag_atomic32_t prev = mag_atomic32_fetch_add(&rc->rc_strong, 1, MAG_MO_RELAXED);
    mag_assert(prev < INT32_MAX, "RC inc/dec imbalance detected"); /* Catch overflow */
}

/* Decrement reference count (release). Object must have MAG_RC_INJECT_HEADER as first field. */
static MAG_AINLINE bool mag_rc_decref(void *obj) {
    mag_rc_control_block_t *rc = (mag_rc_control_block_t *)obj;
    #ifdef MAG_DEBUG /* Verify that object has a valid control block header using the sentinel */
        mag_assert2(rc->__sentinel == 0xDEADBEEF);
    #endif
    mag_atomic32_t prev = mag_atomic32_fetch_sub(&rc->rc_strong, 1, MAG_MO_ACQ_REL);
    mag_assert(prev > 0, "RC inc/dec imbalance detected"); /* Catch underflow */
    if (mag_unlikely(1 == prev)) { /* Decref and invoke destructor. */
        (*rc->dtor)(obj);
        return true; /* Object was destroyed. */
    }
    return false;
}

#ifdef __cplusplus
}
#endif

#endif
