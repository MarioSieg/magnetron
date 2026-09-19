#include "mag_fuse_capture.h"
#include "mag_context.h"
#include "mag_tensor.h"
#include "mag_backend.h"
#include "mag_operator.h"
#include "mag_op_grads.h"
#include "mag_alloc.h"
#include "mag_rc.h"

#include <string.h>

#define mag_fuse_arrlen(a) (sizeof(a)/sizeof(*(a)))

typedef struct mag_fuse_tape_node_t {
  uint8_t op;
  uint8_t num_in;
  mag_tensor_t *in[3];
  mag_tensor_t *out;
} mag_fuse_tape_node_t;

struct mag_fuse_tape_t {
  mag_fuse_tape_node_t nodes[MAG_FUSE_TAPE_MAX];
  uint32_t len;
  uint32_t depth;     /* Region nesting. Only the outermost exit flushes. */
  bool flushing;      /* Guards the read hook against re-entering while the chain is being run. */
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
}

bool mag_fuse_region_active(const mag_context_t *ctx) {
  return !!(ctx->flags & MAG_CTX_FLAG_FUSING);
}

void mag_fuse_region_begin(mag_context_t *ctx) {
  struct mag_fuse_tape_t *tape = mag_fuse_tape(ctx);
  if (mag_unlikely(!tape)) return;
  ++tape->depth;
  ctx->flags |= MAG_CTX_FLAG_FUSING;
}

mag_status_t mag_fuse_region_end(mag_error_t *err, mag_context_t *ctx) {
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (mag_unlikely(!tape || !tape->depth)) return MAG_OK;
  if (--tape->depth) return MAG_OK;
  ctx->flags &= (mag_context_flags_t)~MAG_CTX_FLAG_FUSING;
  return mag_fuse_flush(err, ctx);
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

/* Is this tensor written by an earlier link, and therefore a value rather than a buffer to load? */
static int32_t mag_fuse_producer(const struct mag_fuse_tape_t *tape, uint32_t upto, const mag_tensor_t *t) {
  for (uint32_t i=0; i < upto; ++i)
    if (tape->nodes[i].out == t) return (int32_t)i;
  return -1;
}

/*
** How many buffer slots the graph will need if the chain is flushed now.
**
** Only values that reach memory need one: the tensors the chain reads from outside itself, and the
** results something still holds. A result consumed by a later link and by nothing else never leaves
** a register, so it costs no slot - which is the whole reason a long chain fits at all.
**
** With gradients recording nothing is elided, so every result needs a slot and the count is simply
** every distinct tensor the tape touches.
*/
static uint32_t mag_fuse_distinct_tensors(const struct mag_fuse_tape_t *tape, mag_tensor_t **extra, uint32_t num_extra) {
  const mag_tensor_t *seen[MAG_FUSE_TAPE_MAX*4];
  uint32_t n = 0;
  #define mag_fuse_note(t) \
    do { \
      bool found = false; \
      for (uint32_t k=0; k < n; ++k) if (seen[k] == (t)) { found = true; break; } \
      if (!found && n < mag_fuse_arrlen(seen)) seen[n++] = (t); \
    } while (0)
  for (uint32_t i=0; i < tape->len; ++i) {
    const mag_fuse_tape_node_t *node = tape->nodes+i;
    for (uint8_t j=0; j < node->num_in; ++j) mag_fuse_note(node->in[j]);
    mag_fuse_note(node->out);
  }
  for (uint32_t i=0; i < num_extra; ++i) mag_fuse_note(extra[i]);
  #undef mag_fuse_note
  return n;
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
** later operator that consumes it. And while gradients are recording, every consumer's autodiff
** state increfs its inputs too, once per operand position, so an operator that reads the same value
** twice holds it twice.
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
static bool mag_fuse_value_escapes(const mag_context_t *ctx, const struct mag_fuse_tape_t *tape, uint32_t i) {
  mag_tensor_t *t = tape->nodes[i].out;
  bool recording = !!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER);
  int32_t ours = 1; /* The reference taken when this operator's output was recorded. */
  for (uint32_t j=i+1; j < tape->len; ++j) {
    const mag_fuse_tape_node_t *consumer = tape->nodes+j;
    /* An operator with no backward records nothing, so it holds no autodiff reference either. */
    bool records = recording && mag_op_trait((mag_opcode_t)consumer->op)->backward != NULL;
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

  mag_fuse_graph_t graph;
  mag_fuse_graph_init(&graph, tape->nodes[0].out->meta.dtype);

  mag_tensor_t *bufs[MAG_FUSE_MAX_BUF];
  uint8_t num_bufs = 0;
  int32_t node_reg[MAG_FUSE_TAPE_MAX];

  /* Bind a tensor to a buffer slot, reusing the slot if it is already bound. */
  #define mag_fuse_bind(dst, t) \
    do { \
      int32_t found = -1; \
      for (uint8_t b=0; b < num_bufs; ++b) if (bufs[b] == (t)) { found = b; break; } \
      if (found < 0) { \
        if (num_bufs >= MAG_FUSE_MAX_BUF) goto fallback; \
        bufs[num_bufs] = (t); \
        found = num_bufs++; \
      } \
      (dst) = (uint8_t)found; \
    } while (0)

  for (uint32_t i=0; i < tape->len; ++i) {
    mag_fuse_tape_node_t *node = tape->nodes+i;
    mag_fuse_operand_t operands[3];
    for (uint8_t j=0; j < node->num_in; ++j) {
      int32_t producer = mag_fuse_producer(tape, i, node->in[j]);
      if (producer >= 0) { /* Stays in a register; nothing reaches memory for it. */
        operands[j] = (mag_fuse_operand_t){.kind = MAG_FUSE_REG, .idx = (uint8_t)node_reg[producer]};
        continue;
      }
      uint8_t slot;
      mag_fuse_bind(slot, node->in[j]);
      int32_t reg = node->in[j]->meta.numel == 1
        ? mag_fuse_graph_load_scalar(&graph, slot)  /* One element, broadcast over the loop. */
        : mag_fuse_graph_load(&graph, slot);
      if (reg < 0) goto fallback;
      operands[j] = (mag_fuse_operand_t){.kind = MAG_FUSE_REG, .idx = (uint8_t)reg};
    }
    int32_t reg = mag_fuse_graph_emit(&graph, (mag_opcode_t)node->op, operands, node->num_in);
    if (reg < 0) goto fallback;
    node_reg[i] = reg;
  }

  /* Only the results something outside the chain can still read reach memory. */
  uint32_t elided = 0; /* Counted into the context only if the chain actually runs. */
  for (uint32_t i=0; i < tape->len; ++i) {
    if (!mag_fuse_value_escapes(ctx, tape, i)) { ++elided; continue; }
    uint8_t slot;
    mag_fuse_bind(slot, tape->nodes[i].out);
    if (!mag_fuse_graph_store(&graph, slot, node_reg[i])) goto fallback;
  }
  mag_fuse_graph_prune(&graph);
  if (!graph.num_stores) goto done; /* Nothing observable: the chain need not run at all. */

  {
    /* The pending marks come off before submitting, because the kernel is about to write through
       these very tensors and the guard would otherwise recurse into this function. */
    for (uint32_t i=0; i < tape->len; ++i)
      tape->nodes[i].out->meta.flags &= (mag_tensor_flags_t)~MAG_TFLAG_PENDING;

    mag_device_t *dvc = tape->nodes[0].out->meta.device;
    mag_op_params_t params;
    memset(&params, 0, sizeof(params));
    params.fused.graph = &graph;
    mag_command_t cmd = {
      .op = MAG_OP_FUSED,
      .in = bufs,
      .out = bufs,        /* A chain reads and writes the same bound set; the graph says which is which. */
      .num_in = num_bufs,
      .num_out = num_bufs,
      .params = &params
    };
    mag_status_t st = (*dvc->submit)(err, dvc, &cmd);
    if (mag_unlikely(mag_iserr(st))) {
      /* The backend has no lowering for this chain, or could not build one. Nothing has been
         written, so running the operators one at a time produces exactly what was asked for. */
      if (err) memset(err, 0, sizeof(*err)); /* Declining to lower a chain is not an error the caller should see. */
      status = mag_fuse_replay_eager(err, tape);
      goto done;
    }
    ++tape->chains;
    tape->ops_fused += tape->len;
    tape->elided += elided;
    goto done;
  }

fallback:
  status = mag_fuse_replay_eager(err, tape);
done:
  #undef mag_fuse_bind
  mag_fuse_tape_release(tape);
  tape->flushing = false;
  return status;
}

/* Can this operand appear in a chain at all? */
static bool mag_fuse_operand_ok(const mag_tensor_t *t, const mag_tensor_t *out) {
  if (t->meta.dtype != out->meta.dtype) return false;      /* One dtype per chain. */
  if (t->meta.device != out->meta.device) return false;    /* One device per chain. */
  if (!mag_tensor_is_contiguous(t)) return false;          /* The graph indexes a flat loop. */
  if (t->meta.storage_offset) return false;
  return t->meta.numel == out->meta.numel || t->meta.numel == 1;
}

/* Does this operator touch a value the chain has not produced yet? */
static bool mag_fuse_touches_pending(mag_tensor_t **ts, uint32_t n) {
  for (uint32_t i=0; i < n; ++i)
    if (ts[i] && (ts[i]->meta.flags & MAG_TFLAG_PENDING)) return true;
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
  const mag_context_t *ctx,
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
    if (i+1 < tape->len && mag_fuse_value_escapes(ctx, tape, i)) ++external;
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
  bool *captured
) {
  *captured = false;
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (mag_unlikely(!tape || tape->flushing)) return MAG_OK;

  bool fusible =
    !inplace &&
    num_out == 1 && num_in >= 1 && num_in <= 3 &&
    mag_fuse_op_is_fusible(op) &&
    out[0]->meta.numel >= MAG_FUSE_MIN_ELEMS;
  if (fusible)
    for (uint32_t i=0; i < num_in && fusible; ++i)
      fusible = mag_fuse_operand_ok(in[i], out[0]);
  if (fusible && tape->len) { /* Every link in one chain walks the same index space. */
    const mag_tensor_t *head = tape->nodes[0].out;
    fusible = out[0]->meta.numel == head->meta.numel && out[0]->meta.dtype == head->meta.dtype;
  }

  if (!fusible) {
    /* Only flush when the operator actually reads something the chain still owes. An unrelated
       operator running beside a chain has no reason to cut it short. */
    if (mag_fuse_touches_pending(in, num_in) || mag_fuse_touches_pending(out, num_out))
      return mag_fuse_flush(err, ctx);
    return MAG_OK;
  }

  /* Room for one more link, in instructions and in the buffers the graph can bind. */
  mag_tensor_t *touched[4];
  uint32_t num_touched = 0;
  for (uint32_t i=0; i < num_in; ++i) touched[num_touched++] = in[i];
  touched[num_touched++] = out[0];
  if (tape->len >= MAG_FUSE_TAPE_MAX || mag_fuse_slots_needed(ctx, tape, touched, num_touched) > MAG_FUSE_MAX_BUF) {
    mag_status_t st = mag_fuse_flush(err, ctx);
    if (mag_unlikely(mag_iserr(st))) return st;
  }

  mag_fuse_tape_node_t *node = tape->nodes + tape->len;
  node->op = (uint8_t)op;
  node->num_in = (uint8_t)num_in;
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
