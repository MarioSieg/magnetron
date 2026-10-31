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

#ifndef MAGNETRON_CPU_ASYNC_STREAM_H
#define MAGNETRON_CPU_ASYNC_STREAM_H

#include <core/mag_backend.h>
#include <core/mag_threadlib.h>

#include "mag_cpu_numa.h"
#include "mag_cpu_threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_CPU_CMD_INTRUSIVE_T 6           /* How many in+out tensors to store inline before moving to dyn mem */
#define MAG_CPU_ASYNC_STREAM_DEF_DEPTH 1024
#define MAG_CPU_ASYNC_STREAM_DEF_MB 16      /* Output bytes in flight - beyond cache size it might cost more than the overlap even gains */
#define MAG_CPU_ASYNC_STREAM_RECLAIM_BATCH 16

typedef mag_status_t (mag_cpu_stream_dispatch_fn_t)(mag_error_t *err, void *usr, const mag_command_t *cmd);
typedef void (mag_cpu_stream_seed_fn_t)(void *usr, uint64_t seed);
typedef void (mag_cpu_stream_thread_ctor_fn_t)(void *usr);

typedef enum mag_cpu_cmd_type_t {
  MAG_CPU_CMD_TYPE_OP,       /* Exec op */
  MAG_CPU_CMD_TYPE_SEED      /* Reseed device PRNGs */
} mag_cpu_cmd_type_t;

typedef struct mag_cpu_cmd_t {
  mag_sync_barrier_seq_t seq;
  mag_cpu_cmd_type_t type;
  mag_command_t cmd;
  mag_tensor_t **tensors;
  mag_tensor_t *tensors_intrusive[MAG_CPU_CMD_INTRUSIVE_T];
  uint32_t num_tensors;
  uint32_t cap_tensors;
  mag_op_params_t *params;
  int64_t *aux;
  uint64_t seed;
  size_t out_nb;
} mag_cpu_cmd_t;

extern void mag_cpu_cmd_init(mag_cpu_cmd_t *rec);
extern bool mag_cpu_cmd_reserve_dyn_tensors(mag_cpu_cmd_t *rec, uint32_t extra);
extern bool mag_cpu_cmd_set_params(
  mag_cpu_cmd_t *rec,
  mag_context_t *ctx,
  mag_opcode_t op,
  const mag_op_params_t *params
);
extern mag_status_t mag_cpu_cmd_record(
  mag_error_t *err,
  mag_cpu_cmd_t *rec,
  mag_context_t *ctx,
  const mag_command_t *cmd
);
extern void mag_cpu_cmd_release(mag_cpu_cmd_t *rec, mag_context_t *ctx);

typedef struct mag_cpu_stream_t {
  mag_context_t *ctx;
  mag_cpu_cmd_t *ring_slots;
  uint32_t cap;
  size_t max_inflight_nb;
  mag_cpu_stream_dispatch_fn_t *dispatch;
  mag_cpu_stream_seed_fn_t *seeder;
  mag_cpu_stream_thread_ctor_fn_t *tr_ctor;
  void *usr;
  mag_thread_t *tr;
  bool launch_blocking;
  /* Producer, lock guarded */
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_lock_t lock;
  mag_sync_barrier_seq_t submitted;
  size_t inflight_nb;
  volatile mag_atomic64_t reclaimed;
  /* Producer -> Executor */
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile mag_atomic64_t published;
  volatile mag_atomic32_t exec_parked;
  volatile mag_atomic32_t exec_epoch;
  volatile mag_atomic32_t interrupt;
  /* Executor -> Host */
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile mag_atomic64_t completed;
  volatile mag_atomic32_t waiters;
  volatile mag_atomic32_t done_epoch;
  volatile mag_atomic64_t wake_seq;
  volatile mag_atomic32_t poisoned;
  mag_error_t deferred_err;

  mag_spin_ctrl_t exec_spin;
} mag_cpu_stream_t;

extern mag_status_t mag_cpu_stream_create(
  mag_error_t *err,
  mag_cpu_stream_t **out,
  mag_context_t *ctx,
  uint32_t cap,
  size_t max_inflight_nb,
  mag_cpu_stream_dispatch_fn_t *dispatch,
  mag_cpu_stream_seed_fn_t *seeder,
  mag_cpu_stream_thread_ctor_fn_t *tr_ctor,
  void *usr,
  mag_thread_prio_t prio
);
extern mag_status_t mag_cpu_stream_wait(mag_error_t *err, mag_cpu_stream_t *stream, mag_sync_barrier_seq_t seq);
extern mag_status_t mag_cpu_stream_barrier(mag_error_t *err, mag_cpu_stream_t *stream);
extern mag_status_t mag_cpu_stream_submit(mag_error_t *err, mag_cpu_stream_t *stream, const mag_command_t *cmd, bool inline_ok);
extern mag_status_t mag_cpu_stream_submit_seed(mag_error_t *err, mag_cpu_stream_t *stream, uint64_t seed);
extern bool mag_cpu_stream_query(mag_cpu_stream_t *stream, uint64_t seq);
extern void mag_cpu_stream_shutdown(mag_cpu_stream_t *stream);

#ifdef __cplusplus
}
#endif

#endif
