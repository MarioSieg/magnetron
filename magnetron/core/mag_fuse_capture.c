#include "mag_fuse_capture.h"
#include "mag_context.h"
#include "mag_tensor.h"
#include "mag_backend.h"
#include "mag_operator.h"
#include "mag_op_grads.h"
#include "mag_alloc.h"
#include "mag_rc.h"

#include <string.h>

typedef struct mag_fuse_tape_node_t {
  uint8_t op;
  uint8_t num_in;
  bool records_backward; /* Set when dispatch recorded this node's autodiff inputs. */
  mag_tensor_t *in[3];
  mag_tensor_t *out;
} mag_fuse_tape_node_t;

struct mag_fuse_tape_t {
  mag_fuse_tape_node_t nodes[MAG_FUSE_TAPE_MAX];
  uint32_t len;
  uint32_t depth;     /* Region nesting. Only the outermost exit flushes. */
  uintptr_t owner_thread; /* One context cannot mix chains from concurrent region owners. */
  bool flushing;      /* Guards data-pointer hooks against re-entering while the chain runs. */
  uint64_t chains;
  uint64_t ops_fused;
  uint64_t elided;   /* Results the chain never had to write. */
};

static struct mag_fuse_tape_t *mag_fuse_tape(mag_context_t *ctx) {
  if (mag_likely(ctx->fuse_tape != NULL)) return ctx->fuse_tape;
  struct mag_fuse_tape_t *tape = (*mag_alloc)(NULL, sizeof(*tape), __alignof(struct mag_fuse_tape_t));
  if (mag_unlikely(!tape)) return NULL; /* mag_alloc is contracted never to fail; belt and braces. */
  memset(tape, 0, sizeof(*tape));
  ctx->fuse_tape = tape;
  return tape;
}

/* Clear the pending marks and drop the references the tape was holding. */
static void mag_fuse_tape_release(struct mag_fuse_tape_t *tape) {
  for (uint32_t i=0; i < tape->len; ++i) {
    mag_fuse_tape_node_t *n = tape->nodes+i;
    n->out->meta.flags &= (mag_tensor_flags_t)~MAG_TFLAG_PENDING;
    for (uint8_t j=0; j < n->num_in; ++j) mag_rc_decref(n->in[j]);
    mag_rc_decref(n->out);
  }
  tape->len = 0;
}

void mag_fuse_tape_shutdown(mag_context_t *ctx) {
  if (!ctx->fuse_tape) return;
  /* A context torn down inside a region still has operators on the tape holding references. Drop
     them here, before anything counts what is still alive, or an early exit looks like a leak. */
  mag_fuse_tape_release(ctx->fuse_tape);
  (*mag_alloc)(ctx->fuse_tape, 0, 0);
  ctx->fuse_tape = NULL;
  mag_tls_state.fusing = false;
}

bool mag_fuse_region_active(const mag_context_t *ctx) {
  (void)ctx;
  return mag_tls_state.fusing;
}

mag_status_t mag_fuse_region_begin(mag_error_t *err, mag_context_t *ctx) {
  mag_lock_acquire(&ctx->fuse_state_lock);
  struct mag_fuse_tape_t *tape = mag_fuse_tape(ctx);
  if (mag_unlikely(!tape)) {
    mag_lock_release(&ctx->fuse_state_lock);
    return mag_set_error(err, MAG_ERR_OOM, "fusion: could not allocate the capture tape.");
  }
  uintptr_t thread = mag_thread_id();
  if (tape->depth && tape->owner_thread != thread) {
    mag_lock_release(&ctx->fuse_state_lock);
    return mag_set_error(err, MAG_ERR_OP, "fusion: another thread already owns an active region.");
  }
  tape->owner_thread = thread;
  ++tape->depth;
  mag_tls_state.fusing = true;
  mag_lock_release(&ctx->fuse_state_lock);
  return MAG_OK;
}

mag_status_t mag_fuse_region_end(mag_error_t *err, mag_context_t *ctx) {
  mag_lock_acquire(&ctx->fuse_state_lock);
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (mag_unlikely(!tape || !tape->depth)) {
    mag_lock_release(&ctx->fuse_state_lock);
    return MAG_OK;
  }
  if (tape->owner_thread != mag_thread_id()) {
    mag_lock_release(&ctx->fuse_state_lock);
    return mag_set_error(err, MAG_ERR_OP, "fusion: this region belongs to another thread.");
  }
  if (--tape->depth) {
    mag_lock_release(&ctx->fuse_state_lock);
    return MAG_OK;
  }
  mag_tls_state.fusing = false;
  mag_status_t status = mag_fuse_flush(err, ctx);
  tape->owner_thread = 0;
  mag_lock_release(&ctx->fuse_state_lock);
  return status;
}

void mag_fuse_stats(mag_context_t *ctx, uint64_t *out_chains, uint64_t *out_ops_fused, uint64_t *out_elided) {
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (out_chains) *out_chains = tape ? tape->chains : 0;
  if (out_ops_fused) *out_ops_fused = tape ? tape->ops_fused : 0;
  if (out_elided) *out_elided = tape ? tape->elided : 0;
}

/* Submit the tape one operator at a time, which is what would have happened without a region. */
static mag_status_t mag_fuse_replay_eager(mag_error_t *err, struct mag_fuse_tape_t *tape) {
  for (uint32_t i=0; i < tape->len; ++i) {
    mag_fuse_tape_node_t *n = tape->nodes+i;
    n->out->meta.flags &= (mag_tensor_flags_t)~MAG_TFLAG_PENDING; /* Cleared first: the kernel reads it. */
    mag_device_t *dvc = n->out->meta.device;
    mag_command_t cmd = {
      .op = (mag_opcode_t)n->op,
      .in = n->in,
      .out = &n->out,
      .num_in = n->num_in,
      .num_out = 1,
      .params = NULL
    };
    mag_status_t st = (*dvc->submit)(err, dvc, &cmd);
    if (mag_unlikely(mag_iserr(st))) return st;
  }
  return MAG_OK;
}

static mag_status_t mag_fuse_replay_group(
  mag_error_t *err,
  struct mag_fuse_tape_t *tape,
  const mag_fuse_group_t *group
) {
  for (uint8_t i=0; i < group->len; ++i) {
    mag_fuse_tape_node_t *n = tape->nodes+group->nodes[i];
    n->out->meta.flags &= (mag_tensor_flags_t)~MAG_TFLAG_PENDING;
    mag_device_t *dvc = n->out->meta.device;
    mag_command_t cmd = {
      .op = (mag_opcode_t)n->op,
      .in = n->in,
      .out = &n->out,
      .num_in = n->num_in,
      .num_out = 1,
      .params = NULL
    };
    mag_status_t st = (*dvc->submit)(err, dvc, &cmd);
    if (mag_unlikely(mag_iserr(st))) return st;
  }
  return MAG_OK;
}

/* Is this tensor written by an earlier link, and therefore a value rather than a buffer to load? */
static int32_t mag_fuse_producer(const struct mag_fuse_tape_t *tape, uint32_t upto, const mag_tensor_t *t) {
  for (uint32_t i=0; i < upto; ++i)
    if (tape->nodes[i].out == t) return (int32_t)i;
  return -1;
}

/*
** Can anything outside the chain still read this value once the chain has run?
**
** A chain is captured as it executes, so by the time it runs it is full of results that turned out
** to be wanted only by the next link. Those never need to reach memory at all - they are the traffic
** fusion exists to remove. The ones that do are the results something still holds.
**
** Holding is a reference count question, and the only way to answer it is to account for every
** reference the chain itself caused and see whether any are left over. Two things take them. The
** tape increfs an operator's output when it records it, and increfs each input again for every
** later operator that consumes it. A consumer captured while gradients were recording also has an
** autodiff state that increfs its inputs once per operand position, so an operator that reads the
** same value twice holds it twice.
**
** A leftover reference means somebody outside is holding the tensor and will read it. But a
** reference the chain can account for is not automatically harmless: an autodiff state holds its
** operands so that a backward can use them, and whether that backward reads the memory or only asks
** the shape is a property of the operator, declared by mag_op_backward_ignores_value. A consumer
** whose backward reads the value keeps it alive just as surely as a user variable does.
**
** The accounting fails closed. An unexplained reference makes the count come out high, which reads
** as "still needed" and writes the value back. It could only go wrong by counting a reference that
** does not exist, and only these two places take them.
*/
static bool mag_fuse_value_escapes(const struct mag_fuse_tape_t *tape, uint32_t i) {
  mag_tensor_t *t = tape->nodes[i].out;
  int32_t ours = 1; /* The reference taken when this operator's output was recorded. */
  for (uint32_t j=i+1; j < tape->len; ++j) {
    const mag_fuse_tape_node_t *consumer = tape->nodes+j;
    /* An operator with no backward records nothing, so it holds no autodiff reference either. */
    bool records = consumer->records_backward;
    uint8_t ignores = mag_op_backward_ignores_value((mag_opcode_t)consumer->op);
    for (uint8_t k=0; k < consumer->num_in; ++k) {
      if (consumer->in[k] != t) continue;
      ++ours; /* The tape's own reference to this operand. */
      if (!records) continue;
      ++ours; /* And the autodiff state's. */
      if (!(ignores & (1u<<k)))
        return true; /* This backward will read these bytes, so they have to exist. */
    }
  }
  int32_t rc = (int32_t)mag_atomic32_load(&((mag_rc_control_block_t *)t)->rc_strong, MAG_MO_RELAXED);
  return rc > ours;
}

mag_status_t mag_fuse_flush(mag_error_t *err, mag_context_t *ctx) {
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (!tape || !tape->len || tape->flushing) return MAG_OK;
  tape->flushing = true;
  mag_status_t status = MAG_OK;

  /* The trace is an unfused dependency graph. Nothing here mentions CPU, CUDA, or a compiler. */
  mag_fuse_trace_t trace = {.len = (uint8_t)tape->len};
  for (uint8_t i=0; i < trace.len; ++i) {
    const mag_fuse_tape_node_t *src = tape->nodes+i;
    mag_fuse_trace_node_t *dst = trace.nodes+i;
    dst->op = (mag_opcode_t)src->op;
    dst->num_in = src->num_in;
    dst->dtype = (uint8_t)src->out->meta.dtype;
    dst->numel = src->out->meta.numel;
    dst->device = src->out->meta.device;
    dst->out = src->out;
    dst->observed = mag_fuse_value_escapes(tape, i);
    for (uint8_t j=0; j < src->num_in; ++j) {
      dst->in[j] = src->in[j];
      dst->producer[j] = (int16_t)mag_fuse_producer(tape, i, src->in[j]);
    }
  }
  mag_fuse_plan_t plan;
  if (!mag_fuse_trace_plan(&trace, &plan)) {
    status = mag_fuse_replay_eager(err, tape);
    goto done;
  }

  for (uint8_t step=0; step < plan.num_groups; ++step) {
    uint8_t group_id = plan.order[step];
    const mag_fuse_group_t *group = plan.groups+group_id;
    if (!group->fused) {
      status = mag_fuse_replay_group(err, tape, group);
      if (mag_unlikely(mag_iserr(status))) goto done;
      continue;
    }
    mag_fuse_graph_t graph;
    mag_fuse_graph_init(&graph, (mag_dtype_t)trace.nodes[group->nodes[0]].dtype);
    mag_tensor_t *bufs[MAG_FUSE_MAX_BUF];
    uint8_t num_bufs = 0;
    int32_t node_reg[MAG_FUSE_TAPE_MAX];
    uint32_t elided = 0;

    /* Bind a tensor to a buffer slot, reusing the slot if it is already bound. */
    #define mag_fuse_bind(dst, t) \
      do { \
        int32_t found = -1; \
        for (uint8_t b=0; b < num_bufs; ++b) if (bufs[b] == (t)) { found = b; break; } \
        if (found < 0) { \
          if (num_bufs >= MAG_FUSE_MAX_BUF) goto group_fallback; \
          bufs[num_bufs] = (t); \
          found = num_bufs++; \
        } \
        (dst) = (uint8_t)found; \
      } while (0)

    for (uint8_t local=0; local < group->len; ++local) {
      uint8_t i = group->nodes[local];
      const mag_fuse_trace_node_t *node = trace.nodes+i;
      mag_fuse_operand_t operands[3];
      for (uint8_t j=0; j < node->num_in; ++j) {
        int16_t producer = trace.nodes[i].producer[j];
        if (producer >= 0 && plan.group_of[producer] == group_id) {
          operands[j] = (mag_fuse_operand_t){.kind = MAG_FUSE_REG, .idx = (uint8_t)node_reg[producer]};
          continue;
        }
        uint8_t slot;
        mag_fuse_bind(slot, node->in[j]);
        int32_t reg = node->in[j]->meta.numel == 1
          ? mag_fuse_graph_load_scalar(&graph, slot)
          : mag_fuse_graph_load(&graph, slot);
        if (reg < 0) goto group_fallback;
        operands[j] = (mag_fuse_operand_t){.kind = MAG_FUSE_REG, .idx = (uint8_t)reg};
      }
      int32_t reg = mag_fuse_graph_emit(&graph, node->op, operands, node->num_in);
      if (reg < 0) goto group_fallback;
      node_reg[i] = reg;
    }

    for (uint8_t local=0; local < group->len; ++local) {
      uint8_t i = group->nodes[local];
      if (!plan.store[i]) { ++elided; continue; }
      uint8_t slot;
      mag_fuse_bind(slot, trace.nodes[i].out);
      if (!mag_fuse_graph_store(&graph, slot, node_reg[i])) goto group_fallback;
    }
    mag_fuse_graph_prune(&graph);

    /* The pending marks come off before the kernel asks for its output data pointers. */
    for (uint8_t local=0; local < group->len; ++local)
      tape->nodes[group->nodes[local]].out->meta.flags &= (mag_tensor_flags_t)~MAG_TFLAG_PENDING;

    {
      mag_device_t *dvc = (mag_device_t *)trace.nodes[group->nodes[0]].device;
      mag_op_params_t params = {0};
      params.fused.graph = &graph;
      mag_command_t cmd = {
        .op = MAG_OP_FUSED,
        .in = bufs,
        .out = bufs,
        .num_in = num_bufs,
        .num_out = num_bufs,
        .params = &params
      };
      mag_status_t st = (*dvc->submit)(err, dvc, &cmd);
      if (mag_unlikely(mag_iserr(st))) {
        mag_log_debug("fusion: backend declined a %u-op graph: %s", group->len, err ? err->message : "unknown reason");
        if (err) memset(err, 0, sizeof(*err));
        status = mag_fuse_replay_group(err, tape, group);
        if (mag_unlikely(mag_iserr(status))) goto done;
        continue;
      }
      ++tape->chains;
      tape->ops_fused += group->len;
      tape->elided += elided;
      continue;
    }

group_fallback:
    mag_log_debug("fusion: graph construction fell back to eager replay for %u operators", group->len);
    status = mag_fuse_replay_group(err, tape, group);
    if (mag_unlikely(mag_iserr(status))) goto done;
    #undef mag_fuse_bind
  }

done:
  mag_fuse_tape_release(tape);
  tape->flushing = false;
  return status;
}

/* Can this operand appear in a pointwise trace node at all? */
static bool mag_fuse_operand_ok(const mag_tensor_t *t, const mag_tensor_t *out) {
  if (t->meta.dtype != out->meta.dtype) return false;      /* One dtype per operation. */
  if (t->meta.device != out->meta.device) return false;    /* One device per operation. */
  if (!mag_tensor_is_contiguous(t)) return false;          /* Fused groups index a flat loop. */
  if (t->meta.storage_offset) return false;
  return t->meta.numel == out->meta.numel || t->meta.numel == 1;
}

/* Does this operator touch a value the chain has not produced yet? */
static bool mag_fuse_touches_pending(mag_tensor_t **ts, uint32_t n) {
  for (uint32_t i=0; i < n; ++i)
    if (ts[i] && (ts[i]->meta.flags & MAG_TFLAG_PENDING)) return true;
  return false;
}

/* Storage may be shared by views or by separate tensors borrowing the same buffer. Looking at the
   complete storage range is conservative for views, but preserves execution order for either kind
   of alias without requiring a data-pointer access that could itself flush the tape. */
static bool mag_fuse_storage_overlaps(const mag_tensor_t *a, const mag_tensor_t *b) {
  if (a->meta.device != b->meta.device || !a->storage || !b->storage ||
      !a->storage->size || !b->storage->size) return false;
  uintptr_t pa = a->storage->base, pb = b->storage->base;
  return pa <= pb ? pb-pa < a->storage->size : pa-pb < b->storage->size;
}

bool mag_fuse_tape_reads_storage(const mag_tensor_t *tensor) {
  const struct mag_fuse_tape_t *tape = tensor->ctx->fuse_tape;
  if (!tape || !tape->len || tape->flushing) return false;
  for (uint32_t i=0; i < tape->len; ++i)
    for (uint8_t j=0; j < tape->nodes[i].num_in; ++j)
      if (mag_fuse_storage_overlaps(tensor, tape->nodes[i].in[j])) return true;
  return false;
}

/*
** How many buffer slots the graph will need if the chain is flushed now.
**
** Only values that reach memory need one: the tensors the chain reads from outside itself, and the
** results something still holds. A result consumed by a later link and by nothing else never leaves
** a register, so it costs no slot, which is the reason a long chain fits in the first place.
**
** This is an estimate rather than the decision itself, because the decision depends on reference
** counts that can still change before the chain runs. It only has to be close: if it is low, the
** flush runs out of slots and replays the chain eagerly, which is slower and still right.
*/
static uint32_t mag_fuse_slots_needed(
  const struct mag_fuse_tape_t *tape,
  mag_tensor_t **extra,
  uint32_t num_extra
) {
  uint32_t external = 0;
  for (uint32_t i=0; i < tape->len; ++i) {
    const mag_fuse_tape_node_t *node = tape->nodes+i;
    for (uint8_t j=0; j < node->num_in; ++j) {
      if (mag_fuse_producer(tape, i, node->in[j]) >= 0) continue; /* Produced inside the chain. */
      bool counted = false;
      for (uint32_t k=0; k < i && !counted; ++k)
        for (uint8_t m=0; m < tape->nodes[k].num_in; ++m)
          if (tape->nodes[k].in[m] == node->in[j]) { counted = true; break; }
      if (!counted) ++external;
    }
    /* A result a backward will read has to be written, so it needs a slot of its own. */
    if (i+1 < tape->len && mag_fuse_value_escapes(tape, i)) ++external;
  }
  for (uint32_t i=0; i < num_extra; ++i) ++external;
  return external+1;
}

mag_status_t mag_fuse_capture(
  mag_error_t *err,
  mag_context_t *ctx,
  mag_opcode_t op,
  bool inplace,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out,
  const mag_op_params_t *params,
  bool *captured
) {
  *captured = false;
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (mag_unlikely(!tape || tape->flushing)) return MAG_OK;

  bool deferable =
    !inplace &&
    num_out == 1 && num_in >= 1 && num_in <= 3 &&
    !params && mag_fuse_op_is_deferable(op) &&
    out[0]->meta.numel >= MAG_FUSE_MIN_ELEMS;
  if (deferable)
    for (uint32_t i=0; i < num_in && deferable; ++i)
      deferable = in[i]->meta.device == out[0]->meta.device &&
        (!mag_fuse_op_is_fusible(op) || mag_fuse_operand_ok(in[i], out[0]));
  if (!deferable) {
    /* Only flush when the operator actually reads something the chain still owes. An unrelated
       operator running beside a chain has no reason to cut it short. */
    bool must_flush = mag_fuse_touches_pending(in, num_in) || mag_fuse_touches_pending(out, num_out);
    for (uint32_t i=0; i < num_out && !must_flush; ++i)
      must_flush = mag_fuse_tape_reads_storage(out[i]);
    if (must_flush) return mag_fuse_flush(err, ctx);
    return MAG_OK;
  }

  /* Room for one more link, in instructions and in the buffers the graph can bind. */
  mag_tensor_t *touched[4];
  uint32_t num_touched = 0;
  for (uint32_t i=0; i < num_in; ++i) touched[num_touched++] = in[i];
  touched[num_touched++] = out[0];
  if (tape->len >= MAG_FUSE_TAPE_MAX || mag_fuse_slots_needed(tape, touched, num_touched) > MAG_FUSE_MAX_BUF) {
    mag_status_t st = mag_fuse_flush(err, ctx);
    if (mag_unlikely(mag_iserr(st))) return st;
  }

  mag_fuse_tape_node_t *node = tape->nodes + tape->len;
  node->op = (uint8_t)op;
  node->num_in = (uint8_t)num_in;
  node->records_backward = !mag_tls_state.no_grad && mag_op_trait(op)->backward != NULL;
  for (uint32_t i=0; i < num_in; ++i) {
    node->in[i] = in[i];
    mag_rc_incref(in[i]); /* The chain outlives the caller's own references. */
  }
  node->out = out[0];
  mag_rc_incref(out[0]);
  out[0]->meta.flags |= MAG_TFLAG_PENDING;
  ++tape->len;
  *captured = true;
  return MAG_OK;
}
