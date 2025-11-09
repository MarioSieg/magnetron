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

#include "mag_cpu_phase_fence.h"

void mag_phase_fence_init(mag_phase_fence_t *fence) {
    fence->phase = fence->remaining = 0;
}

void mag_phase_fence_kick(mag_phase_fence_t *fence, int32_t workers_active) {
    mag_atomic32_store(&fence->remaining, workers_active, MAG_MO_RELAXED);
    int32_t cur = mag_atomic32_load(&fence->phase, MAG_MO_RELAXED);
    mag_atomic32_store(&fence->phase, cur+1, MAG_MO_RELEASE);
    mag_futex_wakeall(&fence->phase);
}

void mag_phase_fence_wait(mag_phase_fence_t *fence, int32_t *pha) {
    int32_t p = *pha;
    for (int spin=0; spin < 20; ++spin) {
        if (mag_atomic32_load(&fence->phase, MAG_MO_ACQUIRE) != p) goto ready;
        mag_cpu_pause();
    }
    while (mag_atomic32_load(&fence->phase, MAG_MO_ACQUIRE) == p)
        mag_futex_wait(&fence->phase, p);
    ready: *pha = p+1;
}

void mag_phase_fence_done(mag_phase_fence_t *fence) {
    if (mag_atomic32_fetch_sub(&fence->remaining, 1, MAG_MO_ACQ_REL) == 1)
        mag_futex_wake1(&fence->remaining);
}

void mag_phase_fence_barrier(mag_phase_fence_t *fence) {
    int32_t val = mag_atomic32_load(&fence->remaining, MAG_MO_ACQUIRE);
    while (val != 0) {
        mag_futex_wait(&fence->remaining, val);
        val = mag_atomic32_load(&fence->remaining, MAG_MO_ACQUIRE);
    }
}
