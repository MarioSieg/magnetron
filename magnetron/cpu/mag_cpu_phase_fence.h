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

/* Per-worker wake-up gate */
typedef struct mag_worker_gate_t {
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_atomic32_t phase;  /* Incremented once per op the worker is asked to run */
  mag_atomic32_t parked;                                                /* 1 while the worker is about to be blocked in futex_wait */
} mag_worker_gate_t;

/* Completion fence shared by the master and the active workers of one op. */
typedef struct mag_phase_fence_t {
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_atomic32_t remaining;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_atomic32_t master_parked;
} mag_phase_fence_t;

extern void mag_spin_ctrl_init(mag_spin_ctrl_t *ctrl);
extern void mag_worker_gate_init(mag_worker_gate_t *gate);
extern void mag_worker_gate_wait(mag_worker_gate_t *gate, int32_t *pha, mag_spin_ctrl_t *spin);
extern void mag_worker_gate_open(mag_worker_gate_t *gate);
extern void mag_phase_fence_init(mag_phase_fence_t *fence);
extern void mag_phase_fence_arm(mag_phase_fence_t *fence, int32_t workers_active);
extern void mag_phase_fence_done(mag_phase_fence_t *fence);
extern void mag_phase_fence_barrier(mag_phase_fence_t *fence, mag_spin_ctrl_t *spin);

#ifdef __cplusplus
}
#endif

#endif
