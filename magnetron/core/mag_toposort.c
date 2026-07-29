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

static bool mag_topo_set_push(mag_topo_set_t *set, mag_tensor_t *t) {
  if (set->len == set->cap) {
    size_t cap = set->cap<<1;
    mag_tensor_t **realloced = (*mag_try_alloc)(set->buf, cap*sizeof(*set->buf), 0);
    if (mag_unlikely(!realloced)) return false;
    set->buf = realloced;
    set->cap = cap;
  }
  set->buf[set->len++] = t;
  return true;
}

struct mag_topo_stack_record_t {
  mag_tensor_t *tensor;
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

static bool mag_topo_stack_push(mag_topo_stack_t *stack, mag_tensor_t *t) {
  if (stack->len == stack->cap) {
    size_t cap = stack->cap<<1;
    mag_topo_stack_record_t *realloced = (*mag_try_alloc)(stack->top, cap*sizeof(*stack->top), 0);
    if (mag_unlikely(!realloced)) return false;
    stack->top = realloced;
    stack->cap = cap;
  }
  mag_topo_stack_record_t *rec = stack->top+stack->len++;
  rec->tensor = t;
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

mag_status_t mag_topo_sort(
  mag_error_t *err,
  mag_tensor_t *root,
  mag_topo_stack_t *tmp_stack,
  mag_topo_set_t *out_sorted
) {
  mag_topo_stack_reset(tmp_stack);
  mag_topo_set_reset(out_sorted);
  if (mag_unlikely(!(root->meta.flags & MAG_TFLAG_REQUIRES_GRAD))) return MAG_OK;
  uint64_t traversal_epoch = ++root->ctx->topo_traversal_epoch;
  mag_status_t status = MAG_OK;
  if (!root->au_state) {
    if (mag_unlikely(!mag_au_state_lazy_alloc(&root->au_state, root->ctx))) {
      status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to allocate autodiff state.");
      goto cleanup;
    }
    root->au_state->op = MAG_OP_NOP;
  }
  if (mag_unlikely(!mag_topo_stack_push(tmp_stack, root))) {
    status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow traversal stack.");
    goto cleanup;
  }
  while (tmp_stack->len) { /* Iterative DFS */
    mag_topo_stack_record_t *top = mag_topo_stack_peek(tmp_stack);
    mag_tensor_t *top_t = top->tensor;
    if (!top_t->au_state && (top_t->meta.flags & MAG_TFLAG_REQUIRES_GRAD)) {
      if (mag_unlikely(!mag_au_state_lazy_alloc(&top_t->au_state, top_t->ctx))) {
        status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to allocate autodiff state.");
        goto cleanup;
      }
      top_t->au_state->op = MAG_OP_NOP;
    }
    mag_au_state_t *au = top_t->au_state;
    uint32_t num_children = mag_op_trait(au->op)->in;
    if (num_children == MAG_OP_INOUT_DYN)
      num_children = au->num_in;
    if (top->next_child_idx >= num_children) { /* All children processed */
      mag_topo_stack_pop(tmp_stack);
      if (mag_unlikely(!mag_topo_set_push(out_sorted, top_t))) {
        status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow output set.");
        goto cleanup;
      }
      continue;
    }
    mag_tensor_t *child = au->in[top->next_child_idx++];
    if (mag_unlikely(!child || !child->au_state)) continue;
    if ((child->meta.flags & MAG_TFLAG_REQUIRES_GRAD) && child->au_state->topo_traversal_epoch != traversal_epoch) {
      if (mag_unlikely(!mag_topo_stack_push(tmp_stack, child))) {
        status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow traversal stack.");
        goto cleanup;
      }
      child->au_state->topo_traversal_epoch = traversal_epoch;
    }
  }
cleanup:
  mag_topo_stack_reset(tmp_stack);
  return status;
}
