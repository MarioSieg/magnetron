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

#include "mag_cpu_async_stream.h"

#include <core/mag_context.h>
#include <core/mag_tensor.h>
#include <core/mag_alloc.h>

void mag_cpu_cmd_init(mag_cpu_cmd_t *rec) {
  memset(rec, 0, sizeof(*rec));
  rec->type = MAG_CPU_CMD_TYPE_OP;
  memset(&rec->cmd, 0, sizeof(rec->cmd));
  rec->tensors = rec->tensors_intrusive;
  rec->cap_tensors = MAG_CPU_CMD_INTRUSIVE_T;
}

bool mag_cpu_cmd_reserve_dyn_tensors(mag_cpu_cmd_t *rec, uint32_t extra) {
  if (mag_unlikely(extra > UINT32_MAX-rec->num_tensors)) return false;
  uint32_t len = rec->num_tensors+extra;
  if (len <= rec->cap_tensors) return true;
  uint32_t cap = rec->cap_tensors;
  for (; cap < len; cap <<= 1) {
    if (cap > (UINT32_MAX>>1)) {
      cap = len;
      break;
    }
  }
  mag_tensor_t **realloced;
  /* Transition from intrusive storage to heap */
  if (rec->tensors == rec->tensors_intrusive) {
    realloced = (*mag_try_alloc)(NULL, cap*sizeof(*realloced), 0);
    if (mag_unlikely(!realloced)) return false;
    memcpy(realloced, rec->tensors_intrusive, sizeof(*realloced)*rec->num_tensors);
  } else {
    realloced = (*mag_try_alloc)(rec->tensors, cap*sizeof(*realloced), 0);
    if (mag_unlikely(!realloced)) return false;
  }

  rec->tensors = realloced;
  rec->cap_tensors = cap;
  return true;
}

bool mag_cpu_cmd_set_params(
  mag_cpu_cmd_t *rec,
  mag_context_t *ctx,
  mag_opcode_t op,
  const mag_op_params_t *params
) {
  if (!rec->params) {
    rec->params = mag_slab_alloc(&ctx->cmd_params_slab);
    if (mag_unlikely(!rec->params)) return false;
  }
  *rec->params = *params;
  /* TODO: fix this shit */
  if (op == MAG_OP_REPEAT_INTERLEAVE && params->repeat_interleave.counts) {
    int64_t numel = params->repeat_interleave.count_len;
    rec->aux = (*mag_try_alloc)(NULL, (size_t)numel*sizeof(*rec->aux), 0);
    if (mag_unlikely(!rec->aux)) return false;
    memcpy(rec->aux, params->repeat_interleave.counts, (size_t)numel*sizeof(*rec->aux));
    rec->params->repeat_interleave.counts = rec->aux; /* Horrible design */
  }
  return true;
}

mag_status_t mag_cpu_cmd_record(
  mag_error_t *err,
  mag_cpu_cmd_t *rec,
  mag_context_t *ctx,
  const mag_command_t *cmd
) {
  mag_cpu_cmd_init(rec);
  if (mag_unlikely(!mag_cpu_cmd_reserve_dyn_tensors(rec, cmd->num_in+cmd->num_out)))
    return mag_set_error(err, MAG_ERR_OOM,  "cpu: failed to allocate command tensor list (%u tensors).", cmd->num_in+cmd->num_out);
  for (uint32_t i=0; i < cmd->num_in; ++i) {
    mag_tensor_t *xi = cmd->in[i];
    mag_rc_incref(xi);
    rec->tensors[rec->num_tensors++] = xi;
  }
  for (uint32_t i=0; i < cmd->num_out; ++i) {
    mag_tensor_t *xi = cmd->out[i];
    mag_rc_incref(xi);
    rec->tensors[rec->num_tensors++] = xi;
    rec->out_nb += mag_tensor_numbytes(xi);
  }
  if (cmd->params && mag_unlikely(!mag_cpu_cmd_set_params(rec, ctx, cmd->op, cmd->params))) {
    mag_cpu_cmd_release(rec, ctx);
    return mag_set_error(err, MAG_ERR_OOM, "cpu: failed to allocate command params.");
  }
  rec->cmd = (mag_command_t) {
    .op = cmd->op,
    .in = rec->tensors,
    .num_in = cmd->num_in,
    .out = rec->tensors+cmd->num_in,
    .num_out = cmd->num_out,
    .params = rec->params
  };
  return MAG_OK;
}

void mag_cpu_cmd_release(mag_cpu_cmd_t *rec, mag_context_t *ctx) {
  for (uint32_t i=0; i < rec->num_tensors; ++i)
    mag_rc_decref(rec->tensors[i]);
  if (rec->tensors != rec->tensors_intrusive)
    (*mag_alloc)(rec->tensors, 0, 0);
  if (rec->params) mag_slab_free(&ctx->cmd_params_slab, rec->params);
  if (rec->aux) (*mag_alloc)(rec->aux, 0, 0);
  mag_cpu_cmd_init(rec);
}

static void mag_cpu_cmd_mov(mag_cpu_cmd_t *dst, mag_cpu_cmd_t *src) {
  *dst=*src;
  if (src->tensors == src->tensors_intrusive) dst->tensors = dst->tensors_intrusive;
  mag_cpu_cmd_init(src);
}

static MAG_AINLINE mag_cpu_cmd_t *mag_cpu_stream_ring_slot(mag_cpu_stream_t *stream, mag_sync_barrier_seq_t seq) {
  return stream->ring_slots + (seq-1)%stream->cap;
}
static MAG_AINLINE mag_sync_barrier_seq_t mag_cpu_stream_completed(mag_cpu_stream_t *stream) {
  return (mag_sync_barrier_seq_t)mag_atomic64_load(&stream->completed, MAG_MO_ACQUIRE);
}

static MAG_HOTPROC void mag_stream_exec_tr_entry(void *usr);
mag_status_t mag_cpu_stream_create(
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
) {
  *out = NULL;
  mag_assert2(cap > 0);
  mag_assert2(dispatch && seeder && tr_ctor);
  mag_cpu_stream_t *stream = (*mag_try_alloc)(NULL, sizeof(*stream), __alignof(*stream));
  if (mag_unlikely(!stream))
    return mag_set_error(err, MAG_ERR_OOM, "cpu: failed to allocate command stream.");
  memset(stream, 0, sizeof(*stream));
  stream->ring_slots = (*mag_try_alloc)(NULL, cap*sizeof(*stream->ring_slots), __alignof(*stream->ring_slots));
  if (mag_unlikely(!stream->ring_slots)) {
    (*mag_alloc)(stream, 0, __alignof(*stream));
    return mag_set_error(err, MAG_ERR_OOM, "cpu: failed to allocate command stream ring (%u slots).", cap);
  }
  for (uint32_t i=0; i < cap; ++i)
    mag_cpu_cmd_init(stream->ring_slots+i);
  stream->ctx = ctx;
  stream->cap = cap;
  stream->max_inflight_nb = max_inflight_nb;
  stream->wake_seq = INT64_MAX;
  stream->dispatch = dispatch;
  stream->seeder = seeder;
  stream->tr_ctor = tr_ctor;
  stream->usr = usr;
  stream->lock = MAG_LOCK_INIT;
  mag_spin_ctrl_init(&stream->exec_spin);
  mag_status_t stat = mag_thread_create(
    err,
    &stream->tr,
    &mag_stream_exec_tr_entry,
    prio,
    NULL,
    stream
  );
  if (mag_iserr(stat)) {
    (*mag_alloc)(stream->ring_slots, 0, __alignof(*stream->ring_slots));
    (*mag_alloc)(stream, 0, __alignof(*stream));
    return stat;
  }
  *out = stream;
  return MAG_OK;
}

static void mag_cpu_stream_reclaim(mag_cpu_stream_t *stream) {
  mag_cpu_cmd_t batch[MAG_CPU_ASYNC_STREAM_RECLAIM_BATCH];
  for (;;) {
    if (mag_likely(mag_cpu_stream_completed(stream) <= (uint64_t)mag_atomic64_load(&stream->reclaimed, MAG_MO_RELAXED)))
      return;
    uint32_t n=0;
    mag_lock_acquire(&stream->lock);
    mag_sync_barrier_seq_t done = mag_cpu_stream_completed(stream);
    mag_sync_barrier_seq_t seq = (mag_sync_barrier_seq_t)mag_atomic64_load(&stream->reclaimed, MAG_MO_RELAXED);
    for (; seq < done && n < MAG_CPU_ASYNC_STREAM_RECLAIM_BATCH; ++seq) {
      mag_cpu_cmd_t *rec = mag_cpu_stream_ring_slot(stream, 1+seq);
      stream->inflight_nb -= rec->out_nb;
      mag_cpu_cmd_mov(n+++batch, rec);
    }
    mag_atomic64_store(&stream->reclaimed, (mag_atomic64_t)seq, MAG_MO_RELAXED);
    mag_lock_release(&stream->lock);
    for (uint32_t i=0; i < n; ++i)
      mag_cpu_cmd_release(batch+i, stream->ctx);
  }
}

static void mag_cpu_stream_barrier_until(mag_cpu_stream_t *stream, mag_sync_barrier_seq_t seq) {
  if (mag_cpu_stream_completed(stream) >= seq) return;
  uint64_t t0 = mag_hpc_clock_ns();
  mag_spin_ctrl_t spin_ctrl;
  mag_spin_ctrl_init(&spin_ctrl);
  mag_spin_timer_t spin_timer = mag_spin_timer_start(&spin_ctrl);
  while (mag_cpu_stream_completed(stream) < seq) {
    if (mag_likely(!mag_spin_timer_expired(&spin_timer))) continue;
    mag_atomic32_fetch_add(&stream->waiters, 1, MAG_MO_SEQ_CST);
    mag_atomic64_t want = (mag_atomic64_t)seq;
    mag_atomic64_t cur = mag_atomic64_load(&stream->wake_seq, MAG_MO_SEQ_CST);
    while (cur > want && !mag_atomic64_compare_exchange_weak(&stream->wake_seq, &cur, &want, MAG_MO_SEQ_CST, MAG_MO_SEQ_CST));
    mag_atomic32_t epoch = mag_atomic32_load(&stream->done_epoch, MAG_MO_SEQ_CST);
    if (mag_atomic64_load(&stream->wake_seq, MAG_MO_SEQ_CST) <= want && (uint64_t)mag_atomic64_load(&stream->completed, MAG_MO_SEQ_CST) < seq)
      mag_futex_wait(&stream->done_epoch, epoch);
    mag_atomic32_fetch_sub(&stream->waiters, 1, MAG_MO_SEQ_CST);
  }
  mag_atomic64_fetch_add(&stream->ctx->telemetry.host_waits, 1, MAG_MO_RELAXED);
  mag_atomic64_fetch_add(&stream->ctx->telemetry.host_waits_ns, (mag_atomic64_t)(mag_hpc_clock_ns()-t0), MAG_MO_RELAXED);
}

static mag_status_t mag_cpu_stream_take_deferred_error_locked(mag_error_t *err, mag_cpu_stream_t *stream) {
  if (mag_likely(!mag_atomic32_load(&stream->poisoned, MAG_MO_ACQUIRE)))
    return MAG_OK;
  mag_status_t stat = stream->deferred_err.code;
  if (!err) return stat;
  mag_cpu_stream_barrier_until(stream, stream->submitted);
  if (err->code == MAG_OK) *err = stream->deferred_err;
  memset(&stream->deferred_err, 0, sizeof(stream->deferred_err)); /* Reset error and unpoison */
  mag_atomic32_store(&stream->poisoned, 0, MAG_MO_RELEASE);
  return stat;
}

static void mag_cpu_stream_lock_free_slot(mag_cpu_stream_t *stream, size_t out_nb) {
  bool stalled=false;
  for (;;) {
    mag_cpu_stream_reclaim(stream);
    mag_lock_acquire(&stream->lock);
    uint64_t reclaimed = (uint64_t)mag_atomic64_load(&stream->reclaimed, MAG_MO_RELAXED);
    uint64_t pending = stream->submitted-reclaimed;
    if (mag_likely(pending < stream->cap && (!pending || stream->inflight_nb+out_nb <= stream->max_inflight_nb)))
      return;
    uint64_t need_slots = pending >= stream->cap ? pending - stream->cap*3/4 : 0;
    size_t lim_nb = stream->max_inflight_nb*3/4;
    size_t need_nb = stream->inflight_nb+out_nb > lim_nb ? stream->inflight_nb+out_nb-lim_nb : 0;
    mag_sync_barrier_seq_t target = reclaimed;
    for (size_t freed_nb=0; target < stream->submitted && (target-reclaimed < need_slots || freed_nb < need_nb);)
      freed_nb += mag_cpu_stream_ring_slot(stream, ++target)->out_nb;
    target = mag_vmax(target, 1+reclaimed);
    mag_lock_release(&stream->lock);
    if (!stalled) {
      stalled = true;
      mag_atomic64_fetch_add(&stream->ctx->telemetry.queue_full_stalls, 1, MAG_MO_RELAXED);
    }
    mag_cpu_stream_barrier_until(stream, target);
  }
}

static mag_sync_barrier_seq_t mag_cpu_stream_publish_locked(mag_cpu_stream_t *stream) {
  mag_sync_barrier_seq_t seq = ++stream->submitted;
  mag_atomic64_store(&stream->published, (mag_atomic64_t)seq, MAG_MO_SEQ_CST);
  if (mag_atomic32_load(&stream->exec_parked, MAG_MO_SEQ_CST)) {
    mag_atomic32_fetch_add(&stream->exec_epoch, 1, MAG_MO_SEQ_CST);
    mag_futex_wake1(&stream->exec_epoch);
  }
  return seq;
}

mag_status_t mag_cpu_stream_wait(mag_error_t *err, mag_cpu_stream_t *stream, mag_sync_barrier_seq_t seq) {
  mag_cpu_stream_barrier_until(stream, seq);
  mag_status_t stat = MAG_OK;
  if (mag_unlikely(mag_atomic32_load(&stream->poisoned, MAG_MO_ACQUIRE))) {
    mag_lock_acquire(&stream->lock);
    stat = mag_cpu_stream_take_deferred_error_locked(err, stream);
    mag_lock_release(&stream->lock);
  }
  mag_cpu_stream_reclaim(stream);
  return stat;
}

mag_status_t mag_cpu_stream_barrier(mag_error_t *err, mag_cpu_stream_t *stream) {
  mag_lock_acquire(&stream->lock);
  mag_sync_barrier_seq_t seq = stream->submitted;
  mag_lock_release(&stream->lock);
  return mag_cpu_stream_wait(err, stream, seq);
}

/* Submit command, inline_ok = command is so cheap it should run on the submitting thrad directly (small ops for example) */
mag_status_t mag_cpu_stream_submit(mag_error_t *err, mag_cpu_stream_t *stream, const mag_command_t *cmd, bool inline_ok) {
  size_t out_nb=0;
  for (uint32_t i=0; i < cmd->num_out; ++i)
    out_nb += mag_tensor_numbytes(cmd->out[i]);
  mag_cpu_stream_lock_free_slot(stream, out_nb);
  mag_status_t stat = mag_cpu_stream_take_deferred_error_locked(err, stream);
  if (mag_iserr(stat)) {
    mag_lock_release(&stream->lock);
    mag_cpu_stream_reclaim(stream);
    return stat;
  }
  if (inline_ok && mag_cpu_stream_completed(stream) == stream->submitted) { /* Executor idles */
    stat = (*stream->dispatch)(err, stream->usr, cmd);
    mag_lock_release(&stream->lock);
    return stat;
  }
  mag_sync_barrier_seq_t seq = 1+stream->submitted;
  mag_cpu_cmd_t *slot = mag_cpu_stream_ring_slot(stream, seq);
  stat = mag_cpu_cmd_record(err, slot, stream->ctx, cmd);
  if (mag_unlikely(mag_iserr(stat))) {
    mag_lock_release(&stream->lock);
    return stat;
  }
  slot->seq = seq;
  stream->inflight_nb += slot->out_nb;
  bool host_aliased = false;
  for (uint32_t i=0; i < cmd->num_in; ++i) { /* Stamp hazard tickets for host accessse */
    mag_storage_buffer_t *sto = cmd->in[i]->storage;
    sto->prev_acc_seq = seq;
    host_aliased|=!!(sto->flags&MAG_STORAGE_FLAG_HOST_ALIASED);
  }
  for (uint32_t i=0; i < cmd->num_out; ++i) {
    mag_storage_buffer_t *sto = cmd->out[i]->storage;
    sto->prev_write_seq = sto->prev_acc_seq = seq;
    host_aliased|=!!(sto->flags&MAG_STORAGE_FLAG_HOST_ALIASED);
  }
  mag_cpu_stream_publish_locked(stream);
  mag_lock_release(&stream->lock);
  if (mag_unlikely(host_aliased || stream->launch_blocking)) /* Memory shared with code outside magnetron must observe program order */
    return mag_cpu_stream_wait(err, stream, seq);
  return MAG_OK;
}

mag_status_t mag_cpu_stream_submit_seed(mag_error_t *err, mag_cpu_stream_t *stream, uint64_t seed) {
  (void)err;
  mag_cpu_stream_lock_free_slot(stream, 0);
  mag_cpu_cmd_t *slot = mag_cpu_stream_ring_slot(stream, stream->submitted+1);
  mag_cpu_cmd_init(slot);
  slot->type = MAG_CPU_CMD_TYPE_SEED;
  slot->seed = seed;
  slot->seq = 1+stream->submitted;
  mag_cpu_stream_publish_locked(stream);
  mag_lock_release(&stream->lock);
  return MAG_OK;
}

bool mag_cpu_stream_query(mag_cpu_stream_t *stream, uint64_t seq) {
  return mag_cpu_stream_completed(stream) >= seq && !mag_atomic32_load(&stream->poisoned, MAG_MO_ACQUIRE);
}

void mag_cpu_stream_shutdown(mag_cpu_stream_t *stream) {
  mag_error_t err = {0};
  if (mag_iserr(mag_cpu_stream_barrier(&err, stream)))
    mag_log_error("cpu: deferred error at shutdown: %s", err.message);
  mag_atomic32_store(&stream->interrupt, 1, MAG_MO_SEQ_CST);
  mag_atomic32_fetch_add(&stream->exec_epoch, 1, MAG_MO_SEQ_CST);
  mag_futex_wakeall(&stream->exec_epoch);
  mag_thread_join(NULL, stream->tr);
  mag_cpu_stream_reclaim(stream);
  (*mag_alloc)(stream->ring_slots, 0, __alignof(*stream->ring_slots));
  (*mag_alloc)(stream, 0, __alignof(*stream));
}

/* Executor blocks until command with sequence next is published or stream is interrupted */
static bool mag_cpu_stream_exec_wait(mag_cpu_stream_t *stream, mag_sync_barrier_seq_t next) {
  mag_spin_timer_t timer = mag_spin_timer_start(&stream->exec_spin);
  bool parked = false;
  while ((uint64_t)mag_atomic64_load(&stream->published, MAG_MO_ACQUIRE) < next) {
    if (mag_atomic32_load(&stream->interrupt, MAG_MO_ACQUIRE)) {
      mag_spin_ctrl_update(&stream->exec_spin, &timer, parked);
      return false;
    }
    if (mag_likely(!mag_spin_timer_expired(&timer))) continue;
    parked = true;
    mag_atomic32_store(&stream->exec_parked, 1, MAG_MO_SEQ_CST);
    mag_atomic32_t epoch = mag_atomic32_load(&stream->exec_epoch, MAG_MO_SEQ_CST);
    if ((uint64_t)mag_atomic64_load(&stream->published, MAG_MO_SEQ_CST) < next
      && !mag_atomic32_load(&stream->interrupt, MAG_MO_SEQ_CST))
      mag_futex_wait(&stream->exec_epoch, epoch);
    mag_atomic32_store(&stream->exec_parked, 0, MAG_MO_SEQ_CST);
  }
  mag_spin_ctrl_update(&stream->exec_spin, &timer, parked);
  return true;
}

static void mag_cpu_stream_poison(mag_cpu_stream_t *stream, const mag_cpu_cmd_t *rec, const mag_error_t *cmd_err, mag_status_t stat) {
  mag_error_t *e = &stream->deferred_err;
  *e = *cmd_err;
  if (e->code == MAG_OK) e->code = stat;
  snprintf(
    e->message, sizeof(e->message), "async cpu error in operator '%s' (submitted earlier as command #%" PRIu64 "): %s",
    mag_op_trait(rec->cmd.op)->mnemonic, rec->seq, cmd_err->code != MAG_OK ? cmd_err->message : "kernel failed"
  );
  mag_atomic32_store(&stream->poisoned, 1, MAG_MO_RELEASE);
}

/* Stream thread entry point */
static MAG_HOTPROC void mag_stream_exec_tr_entry(void *usr) {
  mag_cpu_stream_t *stream = usr;
  mag_curr_thread_set_name("mag_cpu_async_stream_executor");
  if (stream->tr_ctor) (*stream->tr_ctor)(stream->usr);
  mag_sync_barrier_seq_t next=1;
  while (mag_likely(mag_cpu_stream_exec_wait(stream, next))) {
    mag_cpu_cmd_t *slot = mag_cpu_stream_ring_slot(stream, next);
    if (slot->type == MAG_CPU_CMD_TYPE_SEED) { /* Seeding works whith poisoned state, no storage is touched */
      (*stream->seeder)(stream->usr, slot->seed);
    } else if (mag_likely(!mag_atomic32_load(&stream->poisoned, MAG_MO_RELAXED))) {
      mag_error_t cmd_err = {0};
      mag_status_t stat = (*stream->dispatch)(&cmd_err, stream->usr, &slot->cmd);
      if (mag_unlikely(mag_iserr(stat)))
        mag_cpu_stream_poison(stream, slot, &cmd_err, stat);
    }
    mag_atomic64_store(&stream->completed, (mag_atomic64_t)next, MAG_MO_SEQ_CST);
    if (mag_atomic32_load(&stream->waiters, MAG_MO_SEQ_CST) && (mag_atomic64_t)next >= mag_atomic64_load(&stream->wake_seq, MAG_MO_SEQ_CST)) {
      mag_atomic64_exchange(&stream->wake_seq, INT64_MAX, MAG_MO_SEQ_CST);
      mag_atomic32_fetch_add(&stream->done_epoch, 1, MAG_MO_SEQ_CST);
      mag_futex_wakeall(&stream->done_epoch);
    }
    ++next;
  }
}
