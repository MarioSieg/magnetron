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

typedef uint64_t mag_rcint_t;
#define MAG_RCINTEGRAL_MAX UINT64_MAX
#define MAG_RCINTEGRAL_PRI PRIu64

/* Header for all objects that are reference counted. */
typedef struct mag_rccontrol_t {
    mag_rcint_t rc;                      /* Strong reference count. Object is deallocated if this reaches zero. */
    void *self;                    /* Pointer to the self. */
    void (*dtor)(void *); /* Destructor function (required). */
} mag_rccontrol_t;

/* Initialize reference count header for a new object. Self-reference and destructor functon must be provided. */
static MAG_AINLINE mag_rccontrol_t mag_rc_control_init(void *self, void (*dtor)(void *)) {
    mag_assert2(self && dtor); /* Self and destructor must be set. */
    mag_rccontrol_t control;
    control.rc = 1;
    control.self = self;
    control.dtor = dtor;
    return control;
}

static MAG_AINLINE void mag_rc_control_incref(mag_rccontrol_t *rcb) { /* Increment reference count (retain). */
    mag_assert(++rcb->rc < MAG_RCINTEGRAL_MAX, "reference count overflow, max RC: %" MAG_RCINTEGRAL_PRI, MAG_RCINTEGRAL_MAX);
}
static MAG_AINLINE bool mag_rc_control_decref(mag_rccontrol_t *rcb) { /* Decrement reference count (release). */
    mag_assert(rcb->rc, "reference count underflow (double free)");
    if (!--rcb->rc) { /* Decref and invoke destructor. */
        (*rcb->dtor)(rcb->self);
        return true; /* Object was destroyed. */
    }
    return false;
}

#ifdef __cplusplus
}
#endif

#endif
