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

#include "mag_cpu_threadpool.h"
#include "mag_cpu_phase_fence.h"
#include "mag_cpu_kernel_data.h"

#include <core/mag_tensor.h>
#include <core/mag_alloc.h>

/* Await signal to start work */
static bool mag_worker_await_work(mag_worker_t *worker, mag_thread_pool_t *pool) {
    if (mag_unlikely(pool->interrupt))
        return false;
    mag_phase_fence_wait(&pool->fence, &worker->phase);
    return !pool->interrupt;
}

/* Execute the operation on the current thread */
void mag_worker_exec_thread_local(const mag_kernel_registry_t *kernels, mag_kernel_payload_t *payload) {
    if (mag_unlikely(!payload->cmd)) return;
    mag_opcode_t op = payload->cmd->op;
    mag_dtype_t dtype = payload->cmd->in && *payload->cmd->in ? (*payload->cmd->in)->dtype : (*payload->cmd->out)->dtype;
    mag_assert2(op >= 0 && op < MAG_OP__NUM);
    mag_assert2(dtype >= 0 && dtype < MAG_DTYPE__NUM);
    void (*kernel)(const mag_kernel_payload_t *) = kernels->operators[op][dtype];
    mag_assert(kernel, "No kernel found for op '%s' with dtype %s", mag_op_traits(op)->mnemonic, mag_type_trait(dtype)->name);
    (*kernel)(payload);
    payload->cmd = NULL;
}

/* Execute the operation and broadcast completion if last chunk was done */
static void mag_worker_exec_and_broadcast(mag_thread_pool_t *pool, const mag_kernel_registry_t *kernels, mag_kernel_payload_t *payload) {
    if (mag_likely(payload->thread_idx < pool->num_active_workers))
        mag_worker_exec_thread_local(kernels, payload);

    /* signal completion to master */
    mag_phase_fence_done(&pool->fence);
}

/* Worker thread entry point */
static MAG_HOTPROC void *mag_worker_thread_entry(void *arg) {
    mag_worker_t *worker = arg;
    mag_thread_pool_t *pool = worker->pool;
    mag_kernel_payload_t *payload = &worker->payload;
    const mag_kernel_registry_t *kernels = pool->kernels;
    char name[32];
    snprintf(name, sizeof(name), "mag_worker_%" PRIx64, payload->thread_idx);
    mag_thread_set_name(name);
    /*mag_thread_set_prio(pool->sched_prio);*/
    mag_atomic32_fetch_add(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
    while (mag_likely(mag_worker_await_work(worker, pool)))  /* Main work loop: wait, work, signal status */
        mag_worker_exec_and_broadcast(pool, kernels, payload);
    mag_atomic32_fetch_sub(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
    return MAG_THREAD_RET_NONE;
}

/* Create thread pool and allocate threads */
mag_thread_pool_t *mag_threadpool_create(mag_context_t *host_ctx, uint32_t num_workers, const mag_kernel_registry_t *kernels, mag_thread_prio_t prio) { /* Create a thread pool */
    mag_thread_pool_t *pool = (*mag_alloc)(NULL, sizeof(*pool), __alignof(mag_thread_pool_t));
    memset(pool, 0, sizeof(*pool));
    mag_worker_t *workers = (*mag_alloc)(NULL, num_workers*sizeof(*workers), __alignof(mag_worker_t));
    memset(workers, 0, num_workers*sizeof(*workers));
    *pool = (mag_thread_pool_t) {
        .interrupt = false,
        .num_allocated_workers = num_workers,
        .num_active_workers = num_workers,
        .num_workers_online = 0,  /* Main thread as worker 0 */
        .workers = workers,
        .kernels = kernels,
        .sched_prio = prio,
        .host_ctx = host_ctx
    };
    mag_phase_fence_init(&pool->fence);
    for (uint32_t ti=0; ti < num_workers; ++ti) { /* Initialize workers */
        mag_worker_t *worker = workers+ti;
        *worker = (mag_worker_t) {
            .phase = 0,
            .prng = {},
            .payload = (mag_kernel_payload_t) {
                .cmd = NULL, /* Will be set later */
                .thread_num = num_workers,
                .thread_idx = ti,
                .prng = NULL
            },
            .pool = pool,
            .is_async = ti != 0 /* Main thread is worker but without thread */
        };
        worker->payload.prng = &worker->prng;
        if (worker->is_async)
            mag_thread_create(&worker->thread, &mag_worker_thread_entry, workers+ti);
    }
    while (mag_atomic32_load(&pool->num_workers_online, MAG_MO_SEQ_CST) != num_workers-1)  /* Wait for all workers to come online */
        mag_thread_yield();
    return pool;
}

/* Destroy thread pool */
void mag_threadpool_destroy(mag_thread_pool_t *pool) {
    pool->interrupt = true;
    mag_phase_fence_kick(&pool->fence, pool->num_allocated_workers);
    while (mag_atomic32_load(&pool->num_workers_online, MAG_MO_SEQ_CST))  /* Wait for all workers to exit */
        mag_thread_yield();
    for (uint32_t i=0; i < pool->num_allocated_workers; ++i) /* Join all worker threads */
        if (pool->workers[i].is_async)
            mag_thread_join(pool->workers[i].thread);
    (*mag_alloc)(pool->workers, 0, __alignof(mag_worker_t));
    (*mag_alloc)(pool, 0, __alignof(mag_thread_pool_t));
}

/* Submits work payload and awakens all threads */
static void mag_threadpool_kickoff(mag_thread_pool_t *pool, const mag_command_t *cmd, uint32_t num_active_workers, volatile mag_atomic64_t *next_tile) {
    pool->num_active_workers = num_active_workers;
    for (uint32_t i=0; i < pool->num_allocated_workers; ++i) { /* Set up payload */
        mag_kernel_payload_t *payload = &pool->workers[i].payload;
        payload->cmd = cmd;
        payload->thread_num = num_active_workers;
        payload->mm_next_tile = next_tile;
    }
    mag_phase_fence_kick(&pool->fence, pool->num_allocated_workers);
}

/* Blocks until all threads have completed their work */
static void mag_threadpool_barrier(mag_thread_pool_t *pool) {
    mag_phase_fence_barrier(&pool->fence);
}

/* Execute an operator tensor on the CPU */
void mag_threadpool_parallel_compute(mag_thread_pool_t *pool, const mag_command_t *cmd, uint32_t num_active_workers) {
    mag_assert2(pool != NULL);
    volatile mag_atomic64_t next_tile = 0;
    mag_threadpool_kickoff(pool, cmd, num_active_workers, &next_tile);                  /* Kick off workers */
    mag_worker_exec_and_broadcast(pool, pool->kernels, &pool->workers->payload);        /* Main thread does work too */
    mag_threadpool_barrier(pool);                                                       /* Wait for all workers to finish */
}
