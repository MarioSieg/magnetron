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

#include "mag_def.h"
#include "mag_hashset.h"
#include "mag_autodiff.h"
#include "mag_toposort.h"
#include "mag_shape.h"

static MAG_COLDPROC void mag_graphviz_dump(const mag_tensor_t *node, FILE *fp, mag_hashset_t *visited) {
    if (!node->au_state) return;
    if (mag_hashset_contains_key(visited, node)) return;
    mag_hashset_insert(visited, node);
    bool is_input = true;
    for (unsigned i=0; i < MAG_MAX_OP_INPUTS; ++i) {
        if (node->au_state->op_inputs[i] != NULL) {
            is_input = false;
            break;
        }
    }
    const char *fillcolor = is_input ? "palegreen" : "skyblue2";
    char dim_buf[150];
    mag_fmt_shape(&dim_buf, &node->shape, node->rank);
    bool gra = node->flags & MAG_TFLAG_REQUIRES_GRAD;
    fprintf(
        fp,
        "  \"%p\" [label=\"⊕ %s|∇ %s|%s|0x%x\", shape=record, style=\"rounded,filled\", fillcolor=%s];\n",
        (void *)node,
        mag_op_meta_of(node->au_state->op)->mnemonic,\
        gra ? "✓" : "🗙",
        dim_buf,
        node->flags,
        fillcolor
    );
    for (unsigned i=0; i < MAG_MAX_OP_INPUTS; ++i) {
        mag_tensor_t *input = node->au_state->op_inputs[i];
        if (!input) continue;
        char name[128];
        snprintf(name, sizeof(name), " in %u", i);
        fprintf(fp, "  \"%p\" -> \"%p\" [label=\"%s\"];\n", (void *)input, (void *)node, name);
        mag_graphviz_dump(input, fp, visited);
    }
}

MAG_COLDPROC void mag_tensor_export_forward_graph_graphviz(mag_tensor_t *t, const char *file) {
    mag_assert2(t && file && *file);
    FILE *f = mag_fopen(file, "w");
    fprintf(f, "digraph computation_graph {\n");
    fprintf(f, "  rankdir=TD;\n");
    fprintf(f, "  node [fontname=\"Helvetica\", shape=box];\n");
    fprintf(f, "  edge [fontname=\"Helvetica\"];\n");
    mag_hashset_t visited;
    mag_hashset_init(&visited, MAG_TOPOSORT_HASHSET_INIT_CAP);
    mag_graphviz_dump(t, f, &visited);
    mag_hashset_free(&visited);
    fprintf(f, "}\n");
    fclose(f);
}

MAG_COLDPROC void mag_tensor_export_backward_graph_graphviz(mag_tensor_t *t, const char *file) {
    mag_topo_set_t post_order;
    mag_topo_set_init(&post_order);
    mag_topo_sort(t, &post_order);
    for (size_t i=0, j=post_order.size - 1; i < j; ++i, --j) {
        mag_swap(mag_tensor_t *, post_order.data[i], post_order.data[j]);
    }
    FILE *fp = mag_fopen(file, "wt");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing the graphviz output.\n");
        return;
    }
    fprintf(fp, "digraph backward_graph {\n");
    fprintf(fp, "    rankdir=TD;\n");
    fprintf(fp, "    node [shape=record, style=\"rounded,filled\", fontname=\"Helvetica\"];\n");
    for (size_t i=0; i < post_order.size; ++i) {
        mag_tensor_t *node = post_order.data[i];
        if (!node->au_state) continue;
        const mag_opmeta_t *meta = mag_op_meta_of(node->au_state->op);
        fprintf(fp, "    \"%p\" [label=\"%s\\nShape: (", node, meta->mnemonic);
        for (int64_t r=0; r < node->rank; ++r) {
            fprintf(fp, "%zu", (size_t)node->shape[r]);
            if (r < node->rank - 1)
                fprintf(fp, ", ");
        }
        fprintf(fp, ")\\nGrad: %s\"];\n", node->au_state->grad ? "set" : "none");
    }
    for (size_t i=0; i < post_order.size; ++i) {
        mag_tensor_t *node = post_order.data[i];
        const mag_opmeta_t *meta = mag_op_meta_of(node->au_state->op);
        for (uint32_t j = 0; j < meta->in; ++j) {
            mag_tensor_t *input = node->au_state->op_inputs[j];
            if (input) {
                fprintf(fp, "    \"%p\" -> \"%p\" [label=\"input %u\"];\n", node, input, j);
            }
        }
    }
    fprintf(fp, "}\n");
    fclose(fp);
    mag_topo_set_free(&post_order);
}
