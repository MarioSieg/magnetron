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

void mag_topo_set_init(mag_topo_set_t *ts) {
  ts->data = NULL;
  ts->size = 0;
  ts->capacity = 0;
}

void mag_topo_set_free(mag_topo_set_t *ts) {
  (*mag_alloc)(ts->data, 0, 0);
  ts->size = 0;
  ts->capacity = 0;
}

static bool mag_topo_set_push(mag_topo_set_t *ts, mag_tensor_t *t) {
  if (ts->size == ts->capacity) {
    size_t cap = !ts->capacity ? 16 : ts->capacity<<1;
    mag_tensor_t **grown = (*mag_try_alloc)(ts->data, cap*sizeof(*ts->data), 0);
    if (mag_unlikely(!grown)) return false;
    ts->data = grown;
    ts->capacity = cap;
  }
  ts->data[ts->size++] = t;
  return true;
}

typedef struct mag_topo_stack_record_t {
  mag_tensor_t *tensor;
  uint32_t next_child_idx;
} mag_topo_stack_record_t;

typedef struct mag_topo_stack_t {
  mag_topo_stack_record_t *top;
  size_t len;
  size_t cap;
} mag_topo_stack_t;

static bool mag_topo_stack_init(mag_topo_stack_t *ts, size_t cap) {
  memset(ts, 0, sizeof(*ts));
  ts->cap = cap ? cap : MAG_TOPOSORT_STACK_INIT_CAP;
  ts->top = (*mag_try_alloc)(NULL, sizeof(*ts->top)*ts->cap, 0);
  return ts->top != NULL; /* false on OOM. */
}

static bool mag_topo_stack_push(mag_topo_stack_t *ts, mag_tensor_t *t) {
  if (ts->len == ts->cap) {
    mag_topo_stack_record_t *grown = (*mag_try_alloc)(ts->top, (ts->cap<<1)*sizeof(*ts->top), 0);
    if (mag_unlikely(!grown)) return false;
    ts->top = grown;
    ts->cap <<= 1;
  }
  mag_topo_stack_record_t *rec = ts->top+ts->len++;
  rec->tensor = t;
  rec->next_child_idx = 0;
  return true;
}

static mag_topo_stack_record_t *mag_topo_stack_peek(mag_topo_stack_t *ts) {
  return ts->top+ts->len-1;
}

static mag_topo_stack_record_t *mag_topo_stack_pop(mag_topo_stack_t *ts) {
  return ts->top+--ts->len;
}

static void mag_topo_stack_free(mag_topo_stack_t *ts) {
  (*mag_alloc)(ts->top, 0, 0);
  ts->top = NULL;
  ts->len = 0;
  ts->cap = 0;
}

mag_status_t mag_topo_sort(mag_error_t *err, mag_tensor_t *root, mag_topo_set_t *out_sorted) {
  if (mag_unlikely(!(root->flags & MAG_TFLAG_REQUIRES_GRAD))) return MAG_OK;
  mag_hashset_t visited;
  if (mag_unlikely(!mag_hashset_init(&visited, MAG_TOPOSORT_HASHSET_INIT_CAP)))
    return mag_set_error(err, MAG_ERR_OOM, "toposort: failed to allocate visited set.");
  mag_topo_stack_t stack;
  if (mag_unlikely(!mag_topo_stack_init(&stack, MAG_TOPOSORT_STACK_INIT_CAP))) {
    mag_hashset_free(&visited);
    return mag_set_error(err, MAG_ERR_OOM, "toposort: failed to allocate traversal stack.");
  }
  mag_status_t status = MAG_OK;
  if (!root->au_state) {
    if (mag_unlikely(!mag_au_state_lazy_alloc(&root->au_state, root->ctx))) {
      status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to allocate autodiff state.");
      goto cleanup;
    }
    root->au_state->op = MAG_OP_NOP;
  }
  if (mag_unlikely(!mag_topo_stack_push(&stack, root))) {
    status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow traversal stack.");
    goto cleanup;
  }
  while (stack.len) { /* Iterative DFS */
    mag_topo_stack_record_t *top = mag_topo_stack_peek(&stack);
    mag_tensor_t *top_t = top->tensor;
    if (!top_t->au_state && (top_t->flags & MAG_TFLAG_REQUIRES_GRAD)) {
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
      mag_topo_stack_pop(&stack);
      if (mag_unlikely(!mag_topo_set_push(out_sorted, top_t))) {
        status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow output set.");
        goto cleanup;
      }
      continue;
    }
    mag_tensor_t *child = au->in[top->next_child_idx++];
    if (child && child->flags & MAG_TFLAG_REQUIRES_GRAD && !mag_hashset_contains_key(&visited, child)) {
      if (mag_unlikely(mag_hashset_insert(&visited, child) == MAG_HASHSET_FULL)) {
        status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow visited set.");
        goto cleanup;
      }
      if (mag_unlikely(!mag_topo_stack_push(&stack, child))) {
        status = mag_set_error(err, MAG_ERR_OOM, "toposort: failed to grow traversal stack.");
        goto cleanup;
      }
    }
  }
cleanup:
  mag_topo_stack_free(&stack);
  mag_hashset_free(&visited);
  return status;
}
