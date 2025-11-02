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

void mag_tensor_array_init(mag_tensor_array_t *arr) {
    arr->data = NULL;
    arr->size = 0;
    arr->capacity = 0;
}

void mag_tensor_array_free(mag_tensor_array_t *arr) {
    (*mag_alloc)(arr->data, 0, 0);
    arr->size = 0;
    arr->capacity = 0;
}

void mag_tensor_array_push(mag_tensor_array_t *arr, mag_tensor_t *t) {
    if (arr->size == arr->capacity) {
        size_t cap = !arr->capacity ? 16 : arr->capacity<<1;
        arr->data = (*mag_alloc)(arr->data, cap*sizeof(*arr->data), 0);
        arr->capacity = cap;
    }
    arr->data[arr->size++] = t;
}


typedef struct mag_topo_record_t {
    mag_tensor_t *tensor;
    uint32_t next_child_idx;
} mag_topo_record_t;


void mag_toposort(mag_tensor_t *root, mag_tensor_array_t *sorted) {
    size_t sta_len = 0, sta_cap = 0;
    mag_topo_record_t *stack = NULL;

#define mag_sta_push(_t) do { \
if (sta_len == sta_cap) { \
size_t old_cap = sta_cap; \
size_t nc = (old_cap == 0) ? 16 : (old_cap * 2); \
stack = (*mag_alloc)(stack, nc*sizeof(*stack), 0); \
sta_cap = nc; \
} \
stack[sta_len].tensor = (_t); \
stack[sta_len].next_child_idx = 0; \
sta_len++; \
} while(0)
#define mag_sta_pop() (stack[--sta_len])

    if (!(root->flags & MAG_TFLAG_REQUIRES_GRAD)) return;
    mag_hashset_t visited = mag_hashset_init(1024);
    mag_sta_push(root);
    while (sta_len) { /* Iterative DFS */
        mag_topo_record_t *top = &stack[sta_len-1];
        mag_tensor_t *cur_tensor = top->tensor;
        mag_assert(cur_tensor->au_state, "Autodiff state not allocated for tensor that requires gradient");
        if (top->next_child_idx < mag_op_meta_of(cur_tensor->au_state->op)->in) {
            mag_tensor_t *child = cur_tensor->au_state->op_inputs[top->next_child_idx++];
            if (child && (child->flags & MAG_TFLAG_REQUIRES_GRAD)) {
                if (!mag_hashset_contains_key(&visited, child)) {
                    mag_assert(mag_hashset_insert(&visited, child) != MAG_HASHSET_FULL, "Hashset full during toposort");
                    mag_sta_push(child);
                }
            }
        } else {
            (void)mag_sta_pop();
            mag_tensor_array_push(sorted, cur_tensor);
        }
    }

#undef mag_sta_push
#undef mag_sta_pop
    (*mag_alloc)(stack, 0, 0);
    mag_hashset_free(&visited);
}
