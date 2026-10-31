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

typedef struct mag_spin_timer_t {
  uint64_t start;
  uint64_t deadline;
  uint64_t tick;
} mag_spin_timer_t;

static MAG_AINLINE mag_spin_timer_t mag_spin_timer_start(const mag_spin_ctrl_t *ctrl) {
  uint64_t now = mag_hpc_clock_ns();
  return (mag_spin_timer_t) {
    .start = now,
    .deadline = now+ctrl->budget_ns,
    .tick = 0
  };
}

static MAG_AINLINE bool mag_spin_timer_expired(mag_spin_timer_t *timer) {
  mag_cpu_pause();
  if (mag_likely(++timer->tick&7)) return false;
  return mag_hpc_clock_ns() >= timer->deadline;
}

static MAG_AINLINE void mag_spin_ctrl_update(mag_spin_ctrl_t *ctrl, const mag_spin_timer_t *timer, bool parked) {
  uint64_t waited = mag_hpc_clock_ns()-timer->start;
  uint64_t budget = ctrl->budget_ns;
  if (waited >= MAG_SPIN_MAX_NS) budget>>=1;
  else if (parked || (waited<<1) > budget) budget = waited<<1;
  budget = mag_vclamp(budget, MAG_SPIN_MIN_NS, MAG_SPIN_MAX_NS);
  ctrl->budget_ns = (uint32_t)budget;
}

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
