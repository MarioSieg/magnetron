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

#include "mag_def.h"
#include "mag_autodiff.h"
#include "mag_toposort.h"
#include "mag_sstream.h"
#include "mag_alloc.h"
#include "mag_hashset.h"

MAG_COLDPROC void mag_tensor_visualize_backprop_graph(mag_tensor_t *tensor, const char *file) {
    mag_topo_set_t post_order;
    mag_topo_set_init(&post_order);
    mag_topo_sort(tensor, &post_order);
    for (size_t i=0, j=post_order.size-1; i < j; ++i, --j) {
        mag_swap(mag_tensor_t *, post_order.data[i], post_order.data[j]);
    }
    mag_sstream_t out;
    mag_sstream_init(&out);
    mag_sstream_append(&out, "digraph backward_graph {\n");
    mag_sstream_append(&out, "    rankdir=TD;\n");
    mag_sstream_append(&out, "    node [shape=record, style=\"rounded,filled\", fontname=\"Helvetica\"];\n");
    for (size_t i=0; i < post_order.size; ++i) {
        mag_tensor_t *node = post_order.data[i];
        if (!node->au_state) continue;
        const mag_op_traits_t *meta = mag_op_traits(node->au_state->op);
        mag_sstream_append(&out, "    \"%p\" [label=\"%s\\nShape: (", node, meta->mnemonic);
        for (int64_t r=0; r < node->coords.rank; ++r) {
            mag_sstream_append(&out, "%zu", (size_t)node->coords.shape[r]);
            if (r < node->coords.rank - 1)
                mag_sstream_append(&out, ", ");
        }
        mag_sstream_append(&out, ")\\nGrad: %s\"];\n", node->au_state->grad ? "set" : "none");
    }
    for (size_t i=0; i < post_order.size; ++i) {
        mag_tensor_t *node = post_order.data[i];
        const mag_op_traits_t *meta = mag_op_traits(node->au_state->op);
        for (uint32_t j = 0; j < meta->in; ++j) {
            mag_tensor_t *input = node->au_state->op_inputs[j];
            if (input) {
                mag_sstream_append(&out, "    \"%p\" -> \"%p\" [label=\"input %u\"];\n", node, input, j);
            }
        }
    }
    mag_sstream_append(&out, "}\n");
    mag_topo_set_free(&post_order);
    mag_sstream_flush(&out, file);
}

typedef struct mag_viz_stack_record_t {
    mag_tensor_t *tensor;
    uint32_t next_child_idx;
} mag_viz_stack_record_t;

typedef struct mag_viz_stack_t {
    mag_viz_stack_record_t *data;
    size_t len;
    size_t cap;
} mag_viz_stack_t;

static void mag_viz_stack_init(mag_viz_stack_t *s, size_t cap) {
    s->len = 0;
    s->cap = cap ? cap : 64;
    s->data = (*mag_alloc)(NULL, s->cap * sizeof(*s->data), 0);
}

static void mag_viz_stack_push(mag_viz_stack_t *s, mag_tensor_t *t) {
    if (s->len == s->cap) {
        s->cap <<= 1;
        s->data = (*mag_alloc)(s->data, s->cap * sizeof(*s->data), 0);
    }
    s->data[s->len++] = (mag_viz_stack_record_t){ .tensor = t, .next_child_idx = 0 };
}

static void mag_viz_stack_pop(mag_viz_stack_t *s) {
    --s->len;
}

static mag_viz_stack_record_t *mag_viz_stack_peek(mag_viz_stack_t *s) {
    return s->data + (s->len - 1);
}

static void mag_viz_stack_free(mag_viz_stack_t *s) {
    (*mag_alloc)(s->data, 0, 0);
    s->data = NULL;
    s->len = 0;
    s->cap = 0;
}

MAG_COLDPROC void mag_tensor_visualize_execution_graph(mag_tensor_t *tensor, const char *file) {
    mag_sstream_t out;
    mag_sstream_init(&out);
    mag_sstream_append(&out, "digraph execution_graph {\n");
    mag_sstream_append(&out, "    rankdir=TD;\n");
    mag_sstream_append(&out, "    node [shape=record, style=\"rounded,filled\", fontname=\"Helvetica\"];\n");
    if (!tensor || !tensor->au_state) {
        mag_sstream_append(&out, "}\n");
        mag_sstream_flush(&out, file);
        return;
    }
    mag_hashset_t visited;
    mag_hashset_init(&visited, 1024);
    mag_viz_stack_t stack;
    mag_viz_stack_init(&stack, 64);
    mag_assert(mag_hashset_insert(&visited, tensor) != MAG_HASHSET_FULL, "Hashset exhausted");
    mag_viz_stack_push(&stack, tensor);
    while (stack.len) {
        mag_viz_stack_record_t *top = mag_viz_stack_peek(&stack);
        mag_tensor_t *node = top->tensor;
        mag_au_state_t *au = node->au_state;
        if (!au) {
            mag_viz_stack_pop(&stack);
            continue;
        }
        if (top->next_child_idx == 0) {
            const mag_op_traits_t *meta = mag_op_traits(au->op);
            mag_sstream_append(&out, "    \"%p\" [label=\"%s\\nShape: (", node, meta->mnemonic);
            for (int64_t r=0; r < node->coords.rank; ++r) {
                mag_sstream_append(&out, "%" PRIi64, node->coords.shape[r]);
                if (r + 1 < node->coords.rank) mag_sstream_append(&out, ", ");
            }
            mag_sstream_append(&out, ")\\nPending: %s\"];\n", (node->flags & MAG_TFLAG_OP_PENDING) ? "yes" : "no");
        }
        if (top->next_child_idx >= au->op_num_inputs) {
            mag_viz_stack_pop(&stack);
            continue;
        }
        mag_tensor_t *input = au->op_inputs[top->next_child_idx++];
        if (!input) continue;
        mag_sstream_append(&out, "    \"%p\" -> \"%p\" [label=\"input %u\"];\n", node, input, top->next_child_idx - 1);
        if (input->au_state && !mag_hashset_contains_key(&visited, input)) {
            mag_assert(mag_hashset_insert(&visited, input) != MAG_HASHSET_FULL, "Hashset exhausted");
            mag_viz_stack_push(&stack, input);
        }
    }
    mag_viz_stack_free(&stack);
    mag_hashset_free(&visited);
    mag_sstream_append(&out, "}\n");
    mag_sstream_flush(&out, file);
}
