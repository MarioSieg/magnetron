#include <prelude.hpp>

#include <core/mag_fuse_trace.h>
#include <core/mag_tensor.h>

namespace {
    int device_a, device_b;
    mag_tensor_t outputs[16] {}, leaves[3] {};

    auto node(const mag_fuse_trace_t &trace, uint8_t id, mag_opcode_t op, const void *device, int64_t numel, bool observed,
              int16_t a = MAG_FUSE_NO_PRODUCER, int16_t b = MAG_FUSE_NO_PRODUCER) -> mag_fuse_trace_node_t {
        mag_fuse_trace_node_t n {};
        n.op = op;
        n.num_in = static_cast<uint8_t>(mag_op_trait(op)->in);
        n.dtype = MAG_DTYPE_FLOAT32;
        n.numel = numel;
        n.device = device;
        n.out = outputs+id;
        n.in[0] = a >= 0 ? trace.nodes[a].out : leaves;
        n.in[1] = b >= 0 ? trace.nodes[b].out : leaves+1;
        n.in[2] = leaves+2;
        n.producer[0] = a;
        n.producer[1] = b;
        n.producer[2] = MAG_FUSE_NO_PRODUCER;
        n.observed = observed;
        return n;
    }
}

TEST(fuse_trace, groups_compatible_operations_across_independent_work) {
    mag_fuse_trace_t trace {};
    trace.len = 4;
    trace.nodes[0] = node(trace, 0, MAG_OP_MUL, &device_a, 16384, false);
    trace.nodes[1] = node(trace, 1, MAG_OP_MUL, &device_a, 32768, false);
    trace.nodes[2] = node(trace, 2, MAG_OP_ADD, &device_a, 16384, true, 0);
    trace.nodes[3] = node(trace, 3, MAG_OP_ADD, &device_a, 32768, true, 1);

    mag_fuse_plan_t plan {};
    ASSERT_TRUE(mag_fuse_trace_plan(&trace, &plan));
    ASSERT_EQ(plan.num_groups, 2);
    EXPECT_EQ(plan.groups[0].len, 2);
    EXPECT_EQ(plan.groups[0].nodes[0], 0);
    EXPECT_EQ(plan.groups[0].nodes[1], 2);
    EXPECT_EQ(plan.groups[1].len, 2);
    EXPECT_EQ(plan.groups[1].nodes[0], 1);
    EXPECT_EQ(plan.groups[1].nodes[1], 3);
    EXPECT_FALSE(plan.store[0]);
    EXPECT_FALSE(plan.store[1]);
    EXPECT_TRUE(plan.store[2]);
    EXPECT_TRUE(plan.store[3]);
}

TEST(fuse_trace, splits_a_group_when_merging_would_cycle) {
    mag_fuse_trace_t trace {};
    trace.len = 3;
    trace.nodes[0] = node(trace, 0, MAG_OP_SQR, &device_a, 16384, false);
    trace.nodes[1] = node(trace, 1, MAG_OP_SQR, &device_b, 16384, false, 0);
    trace.nodes[2] = node(trace, 2, MAG_OP_SQR, &device_a, 16384, true, 1);

    mag_fuse_plan_t plan {};
    ASSERT_TRUE(mag_fuse_trace_plan(&trace, &plan));
    ASSERT_EQ(plan.num_groups, 3);
    EXPECT_EQ(plan.order[0], plan.group_of[0]);
    EXPECT_EQ(plan.order[1], plan.group_of[1]);
    EXPECT_EQ(plan.order[2], plan.group_of[2]);
    EXPECT_TRUE(plan.store[0]);
    EXPECT_TRUE(plan.store[1]);
    EXPECT_TRUE(plan.store[2]);
}

TEST(fuse_trace, schedules_uncompiled_pointwise_ops_between_fused_groups) {
    mag_fuse_trace_t trace {};
    trace.len = 3;
    trace.nodes[0] = node(trace, 0, MAG_OP_MUL, &device_a, 16384, false);
    trace.nodes[1] = node(trace, 1, MAG_OP_TANH, &device_a, 16384, false, 0);
    trace.nodes[2] = node(trace, 2, MAG_OP_ADD, &device_a, 16384, true, 1);

    mag_fuse_plan_t plan {};
    ASSERT_TRUE(mag_fuse_trace_plan(&trace, &plan));
    ASSERT_EQ(plan.num_groups, 3);
    EXPECT_TRUE(plan.groups[plan.order[0]].fused);
    EXPECT_FALSE(plan.groups[plan.order[1]].fused);
    EXPECT_TRUE(plan.groups[plan.order[2]].fused);
    EXPECT_TRUE(plan.store[0]);
    EXPECT_TRUE(plan.store[1]);
    EXPECT_TRUE(plan.store[2]);
}

TEST(fuse_trace, removes_unobserved_branches_before_partitioning) {
    mag_fuse_trace_t trace {};
    trace.len = 3;
    trace.nodes[0] = node(trace, 0, MAG_OP_MUL, &device_a, 16384, false);
    trace.nodes[1] = node(trace, 1, MAG_OP_NEG, &device_a, 16384, false);
    trace.nodes[2] = node(trace, 2, MAG_OP_ADD, &device_a, 16384, true, 0);

    mag_fuse_plan_t plan {};
    ASSERT_TRUE(mag_fuse_trace_plan(&trace, &plan));
    ASSERT_EQ(plan.num_groups, 1);
    EXPECT_TRUE(plan.live[0]);
    EXPECT_FALSE(plan.live[1]);
    EXPECT_TRUE(plan.live[2]);
    EXPECT_EQ(plan.group_of[1], MAG_FUSE_NO_GROUP);
    EXPECT_EQ(plan.groups[0].len, 2);
}

TEST(fuse_trace, rejects_forward_dependencies) {
    mag_fuse_trace_t trace {};
    trace.len = 2;
    trace.nodes[0] = node(trace, 0, MAG_OP_SQR, &device_a, 16384, false, 1);
    trace.nodes[1] = node(trace, 1, MAG_OP_SQR, &device_a, 16384, true);
    trace.nodes[0].in[0] = trace.nodes[1].out;
    mag_fuse_plan_t plan {};
    EXPECT_FALSE(mag_fuse_trace_plan(&trace, &plan));
}
