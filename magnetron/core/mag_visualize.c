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
#include "mag_context.h"
#include "mag_toposort.h"
#include "mag_sstream.h"

MAG_COLDPROC mag_status_t mag_tensor_visualize_backprop_graph(mag_error_t *err, mag_tensor_t *tensor, const char *file) {
  mag_topo_set_t *post_order = &tensor->ctx->topo_set;
  mag_status_t status = mag_topo_sort(err, tensor, &tensor->ctx->topo_stack, post_order);
  if (mag_unlikely(mag_iserr(status) || !post_order->len))
    return status;
  mag_sstream_t out;
  mag_sstream_init(&out);
  mag_sstream_append(&out, "digraph backward_graph {\n");
  mag_sstream_append(&out, "    rankdir=TD;\n");
  mag_sstream_append(&out, "    node [shape=record, style=\"rounded,filled\", fontname=\"Helvetica\"];\n");
  for (size_t i=post_order->len; i --> 0;) {
    mag_tensor_t *node = post_order->buf[i];
    if (!node->au_state) continue;
    const mag_op_traits_t *meta = mag_op_trait(node->au_state->op);
    mag_sstream_append(&out, "    \"%p\" [label=\"%s\\nShape: (", node, meta->mnemonic);
    for (int64_t r=0; r < node->coords.rank; ++r) {
      mag_sstream_append(&out, "%zu", (size_t)node->coords.shape[r]);
      if (r < node->coords.rank-1)
        mag_sstream_append(&out, ", ");
    }
    mag_sstream_append(&out, ")\\nGrad: %s\"];\n", node->au_state->grad ? "set" : "none");
  }
  for (size_t i=0; i < post_order->len; ++i) {
    mag_tensor_t *node = post_order->buf[i];
    if (!node->au_state) continue;
    const mag_op_traits_t *meta = mag_op_trait(node->au_state->op);
    uint32_t numin = meta->in;
    if (numin == MAG_OP_INOUT_DYN) /* Variadic ops (e.g. cat) carry their real input count on the node. */
      numin = node->au_state->num_in;
    for (uint32_t j=0; j < numin; ++j) {
      mag_tensor_t *input = node->au_state->in[j];
      if (input)
        mag_sstream_append(&out, "    \"%p\" -> \"%p\" [label=\"input %u\"];\n", node, input, j);
    }
  }
  mag_sstream_append(&out, "}\n");
  mag_sstream_flush(&out, file);
  return MAG_OK;
}
