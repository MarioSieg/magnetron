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

#ifndef MAG_CPU_PHASE_FENCE_H
#define MAG_CPU_PHASE_FENCE_H

#include <core/mag_thread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_phase_fence_t {
    mag_atomic32_t phase;
    mag_atomic32_t remaining;
} mag_phase_fence_t;

extern void mag_phase_fence_init(mag_phase_fence_t *fence);
extern void mag_phase_fence_kick(mag_phase_fence_t *fence, int32_t workers_active);
extern void mag_phase_fence_wait(mag_phase_fence_t *fence, int32_t *pha);
extern void mag_phase_fence_done(mag_phase_fence_t *fence);
extern void mag_phase_fence_barrier(mag_phase_fence_t *fence);

#ifdef __cplusplus
}
#endif

#endif
