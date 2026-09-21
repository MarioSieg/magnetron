#include "mag_fuse_trace.h"

#include <string.h>

/* A group depends on the groups of its input nodes. At most 48 groups fit in one mask. */
static bool mag_fuse_group_depends_on(const uint64_t *needs, uint8_t group, uint8_t target) {
  uint64_t todo = needs[group];
  uint64_t seen = 0;
  while (todo) {
    uint8_t next = 0;
    while (!(todo & (UINT64_C(1)<<next))) ++next;
    todo &= todo-1;
    if (next == target) return true;
    if (seen & (UINT64_C(1)<<next)) continue;
    seen |= UINT64_C(1)<<next;
    todo |= needs[next] & ~seen;
  }
  return false;
}

static bool mag_fuse_same_kernel(const mag_fuse_trace_node_t *a, const mag_fuse_trace_node_t *b) {
  return a->device == b->device && a->dtype == b->dtype && a->numel == b->numel;
}

static uint8_t mag_fuse_added_instructions(
  const mag_fuse_trace_node_t *node,
  const mag_fuse_plan_t *plan,
  uint8_t group
) {
  uint8_t count = 1; /* The operator itself. */
  for (uint8_t k=0; k < node->num_in; ++k)
    if (node->producer[k] < 0 || plan->group_of[node->producer[k]] != group)
      ++count; /* An input from outside this group needs a load instruction. */
  return count;
}

bool mag_fuse_trace_plan(const mag_fuse_trace_t *trace, mag_fuse_plan_t *plan) {
  if (!trace || !plan || trace->len > MAG_FUSE_TAPE_MAX) return false;
  memset(plan, 0, sizeof(*plan));
  memset(plan->group_of, MAG_FUSE_NO_GROUP, sizeof(plan->group_of));

  for (uint8_t i=0; i < trace->len; ++i) {
    const mag_fuse_trace_node_t *node = trace->nodes+i;
    if (!mag_fuse_op_is_deferable(node->op) || node->num_in != mag_op_trait(node->op)->in ||
        node->num_in > 3 || !node->device || !node->out || node->numel < 1) return false;
    for (uint8_t k=0; k < node->num_in; ++k) {
      if (!node->in[k] || node->producer[k] < MAG_FUSE_NO_PRODUCER || node->producer[k] >= i) return false;
      if (node->producer[k] >= 0 && trace->nodes[node->producer[k]].out != node->in[k]) return false;
    }
    plan->live[i] = node->observed;
  }

  /* Values needed by a live result are live even when nobody holds their tensors. */
  for (int32_t i=(int32_t)trace->len-1; i >= 0; --i) {
    if (!plan->live[i]) continue;
    const mag_fuse_trace_node_t *node = trace->nodes+i;
    for (uint8_t k=0; k < node->num_in; ++k)
      if (node->producer[k] >= 0) plan->live[node->producer[k]] = true;
  }

  uint64_t needs[MAG_FUSE_TAPE_MAX] = {0};
  uint8_t first[MAG_FUSE_TAPE_MAX] = {0};
  uint16_t instructions[MAG_FUSE_TAPE_MAX] = {0};
  for (uint8_t i=0; i < trace->len; ++i) {
    if (!plan->live[i]) continue;
    const mag_fuse_trace_node_t *node = trace->nodes+i;
    bool fusible = mag_fuse_op_is_fusible(node->op);
    uint8_t group = MAG_FUSE_NO_GROUP;
    for (uint8_t g=0; fusible && g < plan->num_groups; ++g) {
      if (!plan->groups[g].fused) continue;
      if (!mag_fuse_same_kernel(node, trace->nodes+first[g])) continue;
      if (instructions[g] + mag_fuse_added_instructions(node, plan, g) > MAG_FUSE_MAX_INS) continue;
      bool cycle = false;
      for (uint8_t k=0; k < node->num_in; ++k) {
        int16_t producer = node->producer[k];
        if (producer < 0) continue;
        uint8_t input_group = plan->group_of[producer];
        if (input_group != g && mag_fuse_group_depends_on(needs, input_group, g)) {
          cycle = true;
          break;
        }
      }
      if (!cycle) { group = g; break; }
    }
    if (group == MAG_FUSE_NO_GROUP) {
      group = plan->num_groups++;
      first[group] = i;
      plan->groups[group].fused = fusible;
    }
    plan->group_of[i] = group;
    mag_fuse_group_t *dst = plan->groups+group;
    dst->nodes[dst->len++] = i;
    if (fusible) instructions[group] += mag_fuse_added_instructions(node, plan, group);
    for (uint8_t k=0; k < node->num_in; ++k) {
      int16_t producer = node->producer[k];
      if (producer < 0) continue;
      uint8_t input_group = plan->group_of[producer];
      if (input_group != group) needs[group] |= UINT64_C(1)<<input_group;
    }
  }

  /* An edge crossing kernels has to reach storage, even if no caller keeps that tensor. */
  for (uint8_t i=0; i < trace->len; ++i) {
    if (!plan->live[i]) continue;
    const mag_fuse_trace_node_t *node = trace->nodes+i;
    plan->store[i] = node->observed || !mag_fuse_op_is_fusible(node->op);
    for (uint8_t k=0; k < node->num_in; ++k) {
      int16_t producer = node->producer[k];
      if (producer >= 0 && plan->group_of[producer] != plan->group_of[i])
        plan->store[producer] = true;
    }
  }

  /* Stable topological schedule. Pure independent kernels retain their first-seen order. */
  uint64_t done = 0;
  for (uint8_t s=0; s < plan->num_groups; ++s) {
    uint8_t ready = MAG_FUSE_NO_GROUP;
    for (uint8_t g=0; g < plan->num_groups; ++g)
      if (!(done & (UINT64_C(1)<<g)) && !(needs[g] & ~done)) { ready = g; break; }
    if (ready == MAG_FUSE_NO_GROUP) return false;
    plan->order[s] = ready;
    done |= UINT64_C(1)<<ready;
  }
  return true;
}
