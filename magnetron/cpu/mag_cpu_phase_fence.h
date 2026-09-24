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

#ifndef MAG_CPU_PHASE_FENCE_H
#define MAG_CPU_PHASE_FENCE_H

#include <core/mag_threadlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_SPIN_MIN_NS 1000u
#define MAG_SPIN_MAX_NS 1000000u
#define MAG_SPIN_INIT_NS 50000u

typedef struct mag_spin_ctrl_t {
  uint32_t budget_ns;
} mag_spin_ctrl_t;

typedef struct mag_phase_fence_t {
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_atomic32_t phase;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_atomic32_t sleepers;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_atomic32_t remaining;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_atomic32_t master_parked;
} mag_phase_fence_t;

extern void mag_spin_ctrl_init(mag_spin_ctrl_t *ctrl);
extern void mag_phase_fence_init(mag_phase_fence_t *fence);
extern void mag_phase_fence_kick(mag_phase_fence_t *fence, int32_t workers_active);
extern void mag_phase_fence_wait(mag_phase_fence_t *fence, int32_t *pha, mag_spin_ctrl_t *spin);
extern void mag_phase_fence_done(mag_phase_fence_t *fence);
extern void mag_phase_fence_barrier(mag_phase_fence_t *fence, mag_spin_ctrl_t *spin);

#ifdef __cplusplus
}
#endif

#endif
