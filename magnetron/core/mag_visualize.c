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
  mag_topo_stack_t topo_stack = {0};
  mag_topo_set_t topo_set = {0};
  mag_topo_set_t *post_order = &topo_set;
  if (mag_unlikely(!mag_topo_set_init(&topo_set, MAG_TOPOSORT_HASHSET_INIT_CAP)))
    return mag_set_error(err, MAG_ERR_OOM, "visualize: failed to allocate traversal set.");
  if (mag_unlikely(!mag_topo_stack_init(&topo_stack, MAG_TOPOSORT_STACK_INIT_CAP))) {
    mag_topo_set_free(&topo_set);
    return mag_set_error(err, MAG_ERR_OOM, "visualize: failed to allocate traversal stack.");
  }
  int64_t topo_epoch = 0;
  mag_status_t status = mag_topo_sort(err, tensor->au_state, &topo_stack, post_order, &topo_epoch);
  mag_topo_stack_free(&topo_stack);
  if (mag_unlikely(mag_iserr(status) || !post_order->len)) {
    if (topo_epoch) mag_topo_release(post_order, topo_epoch);
    mag_topo_set_free(&topo_set);
    return status;
  }
  mag_sstream_t out;
  mag_sstream_init(&out);
  mag_sstream_append(&out, "digraph backward_graph {\n");
  mag_sstream_append(&out, "    rankdir=TD;\n");
  mag_sstream_append(&out, "    node [shape=record, style=\"rounded,filled\", fontname=\"Helvetica\"];\n");
  for (size_t i=post_order->len; i --> 0;) {
    mag_au_state_t *node = post_order->buf[i];
    const mag_op_traits_t *meta = mag_op_trait(node->op);
    mag_sstream_append(&out, "    \"%p\" [label=\"%s\\nShape: (", node, meta->mnemonic);
    mag_tensor_t *owner = node->owner;
    int64_t rank = owner ? owner->meta.coords.rank : 0;
    for (int64_t r=0; r < rank; ++r) {
      mag_sstream_append(&out, "%zu", (size_t)owner->meta.coords.shape[r]);
      if (r < rank-1)
        mag_sstream_append(&out, ", ");
    }
    mag_sstream_append(&out, ")\\nGrad: %s\"];\n", node->grad ? "set" : "none");
  }
  for (size_t i=0; i < post_order->len; ++i) {
    mag_au_state_t *node = post_order->buf[i];
    for (uint32_t j=0; j < node->num_in; ++j) {
      mag_au_state_t *input = node->in_nodes[j];
      if (input)
        mag_sstream_append(&out, "    \"%p\" -> \"%p\" [label=\"input %u\"];\n", node, input, j);
    }
  }
  mag_sstream_append(&out, "}\n");
  mag_sstream_flush(&out, file);
  if (topo_epoch) mag_topo_release(post_order, topo_epoch);
  mag_topo_set_free(&topo_set);
  return MAG_OK;
}
