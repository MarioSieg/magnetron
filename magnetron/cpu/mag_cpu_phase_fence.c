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

#include "mag_cpu_phase_fence.h"

#include <core/mag_def.h>

typedef struct mag_spin_timer_t {
  uint64_t start;
  uint64_t deadline;
  uint32_t tick;
} mag_spin_timer_t;

static MAG_AINLINE mag_spin_timer_t mag_spin_timer_start(const mag_spin_ctrl_t *ctrl) {
  uint64_t now = mag_hpc_clock_ns();
  return (mag_spin_timer_t){.start = now, .deadline = now + ctrl->budget_ns, .tick = 0};
}

static MAG_AINLINE bool mag_spin_timer_expired(mag_spin_timer_t *timer) {
  mag_cpu_pause();
  if (mag_likely(++timer->tick & 7)) return false;
  return mag_hpc_clock_ns() >= timer->deadline;
}

static MAG_AINLINE void mag_spin_ctrl_update(mag_spin_ctrl_t *ctrl, const mag_spin_timer_t *timer, bool parked) {
  uint64_t waited = mag_hpc_clock_ns()-timer->start;
  uint64_t budget = ctrl->budget_ns;
  if (waited >= MAG_SPIN_MAX_NS) budget >>= 1;
  else if (parked || (waited<<1) > budget) budget = waited*2;
  budget = mag_vclamp(budget, MAG_SPIN_MIN_NS, MAG_SPIN_MAX_NS);
  ctrl->budget_ns = (uint32_t)budget;
}

void mag_spin_ctrl_init(mag_spin_ctrl_t *ctrl) {
  ctrl->budget_ns = MAG_SPIN_INIT_NS;
}

void mag_worker_gate_init(mag_worker_gate_t *gate) {
  gate->phase = gate->parked = 0;
}

void mag_worker_gate_wait(mag_worker_gate_t *gate, int32_t *pha, mag_spin_ctrl_t *spin) {
  int32_t p = *pha;
  mag_spin_timer_t timer = mag_spin_timer_start(spin);
  bool parked = false;
  while (mag_atomic32_load(&gate->phase, MAG_MO_ACQUIRE) == p) {
    if (mag_likely(!mag_spin_timer_expired(&timer))) continue;
    parked = true;
    mag_atomic32_store(&gate->parked, 1, MAG_MO_SEQ_CST);    /* Publish intent to sleep BEFORE re-checking, pairs with the load in gate_open */
    while (mag_atomic32_load(&gate->phase, MAG_MO_SEQ_CST) == p)
      mag_futex_wait(&gate->phase, p);
    mag_atomic32_store(&gate->parked, 0, MAG_MO_SEQ_CST);
    break;
  }
  mag_spin_ctrl_update(spin, &timer, parked);
  *pha = p+1;
}

void mag_worker_gate_open(mag_worker_gate_t *gate) {
  mag_atomic32_fetch_add(&gate->phase, 1, MAG_MO_SEQ_CST);  /* Also publishes the payload written before the call */
  if (mag_atomic32_load(&gate->parked, MAG_MO_SEQ_CST))      /* Only do the syscall if the worker is actually asleep */
    mag_futex_wake1(&gate->phase);
}

void mag_phase_fence_init(mag_phase_fence_t *fence) {
  fence->remaining = fence->master_parked = 0;
}

void mag_phase_fence_arm(mag_phase_fence_t *fence, int32_t workers_active) {
  mag_atomic32_store(&fence->remaining, workers_active, MAG_MO_RELAXED); /* Ordered before the gates open by their SEQ_CST RMW */
}

void mag_phase_fence_done(mag_phase_fence_t *fence) {
  if (mag_atomic32_fetch_sub(&fence->remaining, 1, MAG_MO_SEQ_CST) != 1) return;
  if (mag_atomic32_load(&fence->master_parked, MAG_MO_SEQ_CST))
    mag_futex_wake1(&fence->remaining);
}

void mag_phase_fence_barrier(mag_phase_fence_t *fence, mag_spin_ctrl_t *spin) {
  mag_spin_timer_t timer = mag_spin_timer_start(spin);
  bool parked = false;
  int32_t val;
  while ((val = mag_atomic32_load(&fence->remaining, MAG_MO_ACQUIRE)) != 0) {
    if (mag_likely(!mag_spin_timer_expired(&timer))) continue;
    parked = true;
    mag_atomic32_store(&fence->master_parked, 1, MAG_MO_SEQ_CST);
    while ((val = mag_atomic32_load(&fence->remaining, MAG_MO_SEQ_CST)) != 0)
      mag_futex_wait(&fence->remaining, val);
    mag_atomic32_store(&fence->master_parked, 0, MAG_MO_RELAXED);
    break;
  }
  mag_spin_ctrl_update(spin, &timer, parked);
}
