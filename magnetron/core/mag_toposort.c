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

#include "mag_toposort.h"
#include "mag_alloc.h"
#include "mag_hashset.h"
#include "mag_autodiff.h"
#include "mag_context.h"

bool mag_topo_set_init(mag_topo_set_t *set, size_t cap) {
  memset(set, 0, sizeof(*set));
  set->cap = cap ? cap : MAG_TOPOSORT_STACK_INIT_CAP;
  set->buf = (*mag_try_alloc)(NULL, sizeof(*set->buf)*set->cap, 0);
  return set->buf != NULL; /* false on OOM. */
}

void mag_topo_set_reset(mag_topo_set_t *set) {
  set->len = 0;
}

void mag_topo_set_free(mag_topo_set_t *set) {
  (*mag_alloc)(set->buf, 0, 0);
  set->len = 0;
  set->cap = 0;
}

static bool mag_topo_set_push(mag_topo_set_t *set, mag_au_state_t *au) {
  if (set->len == set->cap) {
    size_t cap = set->cap<<1;
    mag_au_state_t **realloced = (*mag_try_alloc)(set->buf, cap*sizeof(*set->buf), 0);
    if (mag_unlikely(!realloced)) return false;
    set->buf = realloced;
    set->cap = cap;
  }
  set->buf[set->len++] = au;
  return true;
}

struct mag_topo_stack_record_t {
  mag_au_state_t *node;
  uint32_t next_child_idx;
};

bool mag_topo_stack_init(mag_topo_stack_t *stack, size_t cap) {
  memset(stack, 0, sizeof(*stack));
  stack->cap = cap ? cap : MAG_TOPOSORT_STACK_INIT_CAP;
  stack->top = (*mag_try_alloc)(NULL, sizeof(*stack->top)*stack->cap, 0);
  return stack->top != NULL; /* false on OOM. */
}

void mag_topo_stack_reset(mag_topo_stack_t *stack) {
  stack->len = 0;
}

static bool mag_topo_stack_push(mag_topo_stack_t *stack, mag_au_state_t *au) {
  if (stack->len == stack->cap) {
    size_t cap = stack->cap<<1;
    mag_topo_stack_record_t *realloced = (*mag_try_alloc)(stack->top, cap*sizeof(*stack->top), 0);
    if (mag_unlikely(!realloced)) return false;
    stack->top = realloced;
    stack->cap = cap;
  }
  mag_topo_stack_record_t *rec = stack->top+stack->len++;
  rec->node = au;
  rec->next_child_idx = 0;
  return true;
}

static mag_topo_stack_record_t *mag_topo_stack_peek(mag_topo_stack_t *stack) {
  return stack->top+stack->len-1;
}

static mag_topo_stack_record_t *mag_topo_stack_pop(mag_topo_stack_t *stack) {
  return stack->top+--stack->len;
}

void mag_topo_stack_free(mag_topo_stack_t *stack) {
  (*mag_alloc)(stack->top, 0, 0);
  stack->top = NULL;
  stack->len = 0;
  stack->cap = 0;
}

static void mag_topo_unclaim(mag_au_state_t *au, int64_t epoch) {
  if (!au) return;
  mag_atomic64_t expect = epoch, desire = 0;
  mag_atomic64_compare_exchange_strong(&au->topo_traversal_epoch, &expect, &desire, MAG_MO_ACQ_REL, MAG_MO_RELAXED);
}

void mag_topo_release(const mag_topo_set_t *sorted, int64_t epoch) {
  for (size_t i=0; i < sorted->len; ++i)
    mag_topo_unclaim(sorted->buf[i], epoch);
}

static void mag_topo_release_partial(const mag_topo_stack_t *stack, const mag_topo_set_t *sorted, int64_t epoch) {
  mag_topo_release(sorted, epoch);
  for (size_t i=0; i < stack->len; ++i)
    mag_topo_unclaim(stack->top[i].node, epoch);
}

static int mag_topo_try_claim(mag_au_state_t *au, int64_t epoch) {
  if (mag_atomic64_load(&au->topo_traversal_epoch, MAG_MO_ACQUIRE) == epoch) return 0;
  mag_atomic64_t expect = 0, desire = epoch;
  if (mag_likely(mag_atomic64_compare_exchange_strong(&au->topo_traversal_epoch, &expect, &desire, MAG_MO_ACQ_REL, MAG_MO_RELAXED))) return 1;
  return expect == epoch ? 0 : -1;
}

static MAG_COLDPROC mag_status_t mag_topo_shared_graph_error(mag_error_t *err) {
  return mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: another thread is traversing a graph that shares nodes with this one, concurrent backward is only supported over disjoint graphs.");
}

mag_status_t mag_topo_sort(
  mag_error_t *err,
  mag_au_state_t *root,
  mag_topo_stack_t *tmp_stack,
  mag_topo_set_t *out_sorted,
  int64_t *out_epoch
) {
  mag_topo_stack_reset(tmp_stack);
  mag_topo_set_reset(out_sorted);
  *out_epoch = 0;
  if (mag_unlikely(!root)) return MAG_OK;
  int64_t traversal_epoch = 1+mag_atomic64_fetch_add(&root->ctx->topo_traversal_epoch, 1, MAG_MO_RELAXED);
  mag_status_t status = MAG_OK;
  if (mag_unlikely(mag_topo_try_claim(root, traversal_epoch) < 0)) {
    status = mag_topo_shared_graph_error(err);
    goto cleanup;
  }
  if (mag_unlikely(!mag_topo_stack_push(tmp_stack, root))) {
    status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow traversal stack.");
    goto cleanup;
  }
  while (tmp_stack->len) { /* Iterative DFS */
    mag_topo_stack_record_t *top = mag_topo_stack_peek(tmp_stack);
    mag_au_state_t *au = top->node;
    uint32_t num_children = mag_op_trait(au->op)->in;
    if (num_children == MAG_OP_INOUT_DYN || num_children > au->num_in)
      num_children = au->num_in;
    if (top->next_child_idx >= num_children) { /* All children processed */
      mag_topo_stack_pop(tmp_stack);
      if (mag_unlikely(!mag_topo_set_push(out_sorted, au))) {
        status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow output set.");
        goto cleanup;
      }
      continue;
    }
    uint32_t ci = top->next_child_idx++;
    mag_au_state_t *child = au->in_nodes[ci];
    if (mag_unlikely(!child || !au->in[ci])) continue;
    if (au->in[ci]->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
      int claimed = mag_topo_try_claim(child, traversal_epoch);
      if (mag_unlikely(claimed < 0)) {
        status = mag_topo_shared_graph_error(err);
        goto cleanup;
      }
      if (claimed && mag_unlikely(!mag_topo_stack_push(tmp_stack, child))) {
        status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow traversal stack.");
        goto cleanup;
      }
    }
  }
cleanup:
  if (mag_likely(!mag_iserr(status))) *out_epoch = traversal_epoch;
  else mag_topo_release_partial(tmp_stack, out_sorted, traversal_epoch);
  mag_topo_stack_reset(tmp_stack);
  return status;
}
