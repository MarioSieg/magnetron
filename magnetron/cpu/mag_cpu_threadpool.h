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

#ifndef MAG_CPU_THREADPOOL_H
#define MAG_CPU_THREADPOOL_H

#include "mag_cpu_kernel_data.h"
#include "mag_cpu_phase_fence.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_worker_t mag_worker_t;
typedef struct mag_thread_pool_t {
    mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile bool interrupt;   /* Interrupt flag, 1=stop */
    mag_phase_fence_t fence;
    int32_t num_allocated_workers;                 /* Number of intra-op workers allocated */
    uint32_t num_active_workers;                    /* Number of intra-op workers that are actively used in this compute step. */
    volatile mag_atomic32_t num_workers_online;       /* Number of workers that are online */
    mag_worker_t *workers;                          /* Array of workers */
    const mag_kernel_registry_t *kernels;           /* Specialized compute kernel registry */
    mag_thread_prio_t sched_prio;             /* Scheduling priority */
    mag_context_t *host_ctx;                            /* Host context */
} mag_thread_pool_t;

struct mag_worker_t {
    int32_t phase;                          /* Current compute phase */
    mag_kernel_payload_t payload;           /* Compute op payload */
    mag_philox4x32_stream_t prng;           /* Thread local prng */
    mag_thread_pool_t *pool;                /* Host thread pool */
    bool is_async;                          /* True if worker is async (executed on a different thread)  */
    mag_thread_t thread;                    /* Thread handle */
} mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE);

extern mag_thread_pool_t *mag_threadpool_create(mag_context_t *host_ctx, uint32_t num_workers, const mag_kernel_registry_t *kernels, mag_thread_prio_t prio);
extern void mag_worker_exec_thread_local(const mag_kernel_registry_t *kernels, mag_kernel_payload_t *payload);
extern void mag_threadpool_parallel_compute(mag_thread_pool_t *pool, const mag_command_t *cmd, uint32_t num_active_workers);
extern void mag_threadpool_destroy(mag_thread_pool_t *pool);

#ifdef __cplusplus
}
#endif

#endif
