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

static void mag_topo_set_push(mag_topo_set_t *ts, mag_tensor_t *t) {
    if (ts->size == ts->capacity) {
        size_t cap = !ts->capacity ? 16 : ts->capacity<<1;
        ts->data = (*mag_alloc)(ts->data, cap*sizeof(*ts->data), 0);
        ts->capacity = cap;
    }
    ts->data[ts->size++] = t;
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

static void mag_topo_stack_init(mag_topo_stack_t *ts, size_t cap) {
    memset(ts, 0, sizeof(*ts));
    ts->cap = cap ? cap : MAG_TOPOSORT_STACK_INIT_CAP;
    ts->top = (*mag_alloc)(NULL, sizeof(*ts->top)*ts->cap, 0);
}

static void mag_topo_stack_push(mag_topo_stack_t *ts, mag_tensor_t *t) {
    if (ts->len == ts->cap)
        ts->top = (*mag_alloc)(ts->top, (ts->cap<<=1)*sizeof(*ts->top), 0);
    mag_topo_stack_record_t *rec = ts->top+ts->len++;
    rec->tensor = t;
    rec->next_child_idx = 0;
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

void mag_topo_sort(mag_tensor_t *root, mag_topo_set_t *out_sorted) {
    if (mag_unlikely(!(root->flags & MAG_TFLAG_REQUIRES_GRAD))) return;
    mag_hashset_t visited;
    mag_hashset_init(&visited, MAG_TOPOSORT_HASHSET_INIT_CAP);
    mag_topo_stack_t stack;
    mag_topo_stack_init(&stack, MAG_TOPOSORT_STACK_INIT_CAP);
    if (!root->au_state) {
        mag_au_state_lazy_alloc(&root->au_state, root->ctx);
        root->au_state->op = MAG_OP_NOP;
    }
    mag_topo_stack_push(&stack, root);
    while (stack.len) { /* Iterative DFS */
        mag_topo_stack_record_t *top = mag_topo_stack_peek(&stack);
        mag_tensor_t *top_t = top->tensor;
        if (!top_t->au_state && (top_t->flags & MAG_TFLAG_REQUIRES_GRAD)) {
            mag_au_state_lazy_alloc(&top_t->au_state, top_t->ctx);
            top_t->au_state->op = MAG_OP_NOP;  // no parents
        }
        mag_au_state_t *au = top_t->au_state;
        uint32_t num_children = mag_op_traits(au->op)->in;
        if (top->next_child_idx >= num_children) { /* All children processed */
            mag_topo_stack_pop(&stack);
            mag_topo_set_push(out_sorted, top_t);
            continue;
        }
        mag_tensor_t *child = au->op_inputs[top->next_child_idx++];
        if (child && child->flags & MAG_TFLAG_REQUIRES_GRAD && !mag_hashset_contains_key(&visited, child)) {
            mag_assert(mag_hashset_insert(&visited, child) != MAG_HASHSET_FULL, "Hashset exhausted");
            mag_topo_stack_push(&stack, child);
        }
    }
    mag_topo_stack_free(&stack);
    mag_hashset_free(&visited);
}
