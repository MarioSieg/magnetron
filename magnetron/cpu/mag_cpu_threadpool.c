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

#include "mag_cpu_threadpool.h"
#include "mag_cpu_kernel_data.h"
#include "mag_cpu_numa.h"
#include "mag_cpu_phase_fence.h"

#include <core/mag_alloc.h>
#include <core/mag_tensor.h>

/* Await signal to start work */
static bool mag_worker_await_work(mag_worker_t *worker, mag_thread_pool_t *pool) {
  if (mag_unlikely(pool->interrupt))
    return false;
  mag_phase_fence_wait(&pool->fence, &worker->phase);
  return !pool->interrupt;
}

static mag_dtype_t mag_command_dispatch_dtype(const mag_command_t *cmd) {
  switch (cmd->op) {
    case MAG_OP_MASKED_FILL:
    case MAG_OP_WHERE: return cmd->out[0]->dtype; /* Where kernel is determined by output dtype */
    default: return cmd->in && cmd->in[0] ? cmd->in[0]->dtype : cmd->out[0]->dtype; /* Most ops use input[0] dtype */
  }
}

/* Execute the operation on the current thread */
mag_status_t mag_worker_exec_thread_local(mag_error_t *err, const mag_kernel_registry_t *kernels, mag_kernel_payload_t *payload) {
  if (mag_unlikely(!(payload->cmd != NULL))) {
    return mag_set_error(err, MAG_ERR_KERNEL, "cpu: missing kernel command descriptor.");
  }
  mag_opcode_t op = payload->cmd->op;
  mag_dtype_t dtype = mag_command_dispatch_dtype(payload->cmd);
  mag_assert2(op >= 0 && op < MAG_OP__NUM);
  mag_assert2(dtype >= 0 && dtype < MAG_DTYPE__NUM);
  mag_status_t (*kernel)(mag_error_t *, const mag_kernel_payload_t *) = kernels->operators[op][dtype];
  if (mag_unlikely(!(kernel != NULL))) {
    return mag_set_error(err, MAG_ERR_KERNEL, "cpu: no kernel found for operator '%s' with dtype '%s'.", mag_op_trait(op)->mnemonic, mag_type_trait(dtype)->name);
  }
  mag_status_t stat = (*kernel)(err, payload);
  payload->cmd = NULL;
  return stat;
}

/* Execute the operation and broadcast completion if last chunk was done */
static mag_status_t mag_worker_exec_and_broadcast(mag_error_t *err, mag_thread_pool_t *pool, const mag_kernel_registry_t *kernels, mag_kernel_payload_t *payload) {
  mag_status_t stat = MAG_OK;
  if (mag_likely(payload->thread_idx < pool->num_active_workers))
    stat = mag_worker_exec_thread_local(err, kernels, payload);
  mag_phase_fence_done(&pool->fence);   /* signal completion to master */
  return stat;
}

/* Worker thread entry point */
static MAG_HOTPROC void mag_worker_thread_entry(void *arg) {
  mag_worker_t *worker = arg;
  mag_thread_pool_t *pool = worker->pool;
  mag_kernel_payload_t *payload = &worker->payload;
  mag_error_t *err = &worker->err;
  mag_status_t *stat = &worker->stat;
  const mag_kernel_registry_t *kernels = pool->kernels;
  char name[32];
  snprintf(name, sizeof(name), "mag_worker_thread_%" PRIx64, payload->thread_idx);
  mag_curr_thread_set_name(name);
  /*mag_thread_set_prio(pool->sched_prio);*/
  if (mag_numa_is_numa(pool->numa_ctrl)) /* Pin numa affinity if numa system */
    mag_numa_pin_thread_affinity(pool->numa_ctrl, payload->thread_idx);
  mag_atomic32_fetch_add(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
  while (mag_likely(mag_worker_await_work(worker, pool)))  /* Main work loop: wait, work, signal status */
    *stat = mag_worker_exec_and_broadcast(err, pool, kernels, payload);
  mag_atomic32_fetch_sub(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
}

/* Create thread pool and allocate threads */
mag_status_t mag_threadpool_create(
  mag_error_t *err,
  mag_thread_pool_t **out_pool,
  mag_context_t *host_ctx,
  uint32_t num_workers,
  const mag_kernel_registry_t *kernels,
  mag_numa_node_controller_t *numa,
  mag_thread_prio_t sched_prio
) {
  *out_pool = NULL;
  mag_thread_pool_t *pool = (*mag_try_alloc)(NULL, sizeof(*pool), __alignof(mag_thread_pool_t));
  if (mag_unlikely(!pool))
    return mag_set_error(err, MAG_ERR_OOM, "cpu: failed to allocate thread pool.");
  memset(pool, 0, sizeof(*pool));
  mag_worker_t *workers = (*mag_try_alloc)(NULL, num_workers*sizeof(*workers), __alignof(mag_worker_t));
  if (mag_unlikely(!workers)) {
    (*mag_try_alloc)(pool, 0, __alignof(mag_thread_pool_t));
    return mag_set_error(err, MAG_ERR_OOM, "cpu: failed to allocate %u worker threads.", num_workers);
  }
  memset(workers, 0, num_workers*sizeof(*workers));
  *pool = (mag_thread_pool_t) {
    .interrupt = false,
    .num_allocated_workers = (int32_t)num_workers,
    .num_active_workers = num_workers,
    .num_workers_online = 0,  /* Main thread as worker 0 */
    .workers = workers,
    .kernels = kernels,
    .host_ctx = host_ctx,
    .numa_ctrl = numa
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
    };
    worker->payload.prng = &worker->prng;
    bool is_main = ti == 0;
    if (!is_main) { /* Main thread is worker 0 but runs inline without its own thread */
      if (mag_iserr(mag_thread_create(
          err,
          &worker->thread,
          &mag_worker_thread_entry,
          sched_prio,
          NULL,
          worker
        ))) {
          pool->num_allocated_workers = (int32_t)ti;    /* Only workers [0, ti) may have live threads - reuse the proven teardown. */
          mag_threadpool_destroy(pool);
          return err->code;
      }
    }
  }
  while (mag_atomic32_load(&pool->num_workers_online, MAG_MO_SEQ_CST) != num_workers-1)  /* Wait for all workers to come online */
    mag_curr_thread_yield();
  *out_pool = pool;
  return MAG_OK;
}

/* Destroy thread pool */
void mag_threadpool_destroy(mag_thread_pool_t *pool) {
  pool->interrupt = true;
  mag_phase_fence_kick(&pool->fence, pool->num_allocated_workers);
  while (mag_atomic32_load(&pool->num_workers_online, MAG_MO_SEQ_CST))  /* Wait for all workers to exit */
    mag_curr_thread_yield();
  for (uint32_t i=0; i < pool->num_allocated_workers; ++i) /* Join all worker threads */
    if (pool->workers[i].thread)
      mag_thread_join(NULL, pool->workers[i].thread);
  (*mag_alloc)(pool->workers, 0, __alignof(mag_worker_t));
  (*mag_alloc)(pool, 0, __alignof(mag_thread_pool_t));
}

/* Submits work payload and awakens all threads */
static void mag_threadpool_kickoff(mag_thread_pool_t *pool, const mag_command_t *cmd, uint32_t num_active_workers, mag_tile_sched_t *tile_sched) {
  pool->num_active_workers = num_active_workers;
  for (uint32_t i=0; i < pool->num_allocated_workers; ++i) { /* Set up payload */
    mag_kernel_payload_t *payload = &pool->workers[i].payload;
    payload->cmd = cmd;
    payload->thread_num = num_active_workers;
    payload->tile_sched = tile_sched;
  }
  mag_phase_fence_kick(&pool->fence, pool->num_allocated_workers);
}

/* Blocks until all threads have completed their work */
static void mag_threadpool_barrier(mag_thread_pool_t *pool) {
  mag_phase_fence_barrier(&pool->fence);
}

static void mag_threadpool_clear_worker_status(mag_thread_pool_t *pool) {
  for (uint32_t i=0; i < pool->num_allocated_workers; ++i) {
    pool->workers[i].stat = MAG_OK;
    memset(&pool->workers[i].err, 0, sizeof(pool->workers[i].err));
  }
}

static mag_status_t mag_threadpool_collect_status(mag_error_t *err, mag_thread_pool_t *pool) {
  for (uint32_t i=0; i < pool->num_active_workers; ++i) {
    mag_worker_t *w = &pool->workers[i];
    if (mag_unlikely(w->stat != MAG_OK)) {
      if (err && err->code == MAG_OK)
        *err = w->err;
      return w->stat;
    }
  }
  return MAG_OK;
}

/* Execute an operator tensor on the CPU */
mag_status_t mag_threadpool_parallel_compute(mag_error_t *err, mag_thread_pool_t *pool, const mag_command_t *cmd, uint32_t num_active_workers) {
  mag_assert2(pool != NULL);
  if (err) memset(err, 0, sizeof(*err));
  mag_threadpool_clear_worker_status(pool);
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_tile_sched_t tile_sched = {0};
  mag_threadpool_kickoff(pool, cmd, num_active_workers, &tile_sched); /* Kick off workers */
  pool->workers[0].stat = mag_worker_exec_and_broadcast(err, pool, pool->kernels, &pool->workers->payload); /* Main thread does work too */
  mag_threadpool_barrier(pool); /* Wait for all workers to finish */
  return mag_threadpool_collect_status(err, pool);
}
