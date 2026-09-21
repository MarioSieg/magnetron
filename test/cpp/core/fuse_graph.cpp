#include <prelude.hpp>

#include <core/mag_fuse_graph.h>

using namespace magnetron;

static mag_fuse_operand_t reg(int32_t r) { return mag_fuse_operand_t{MAG_FUSE_REG, static_cast<uint8_t>(r)}; }
static mag_fuse_operand_t imm(uint8_t i) { return mag_fuse_operand_t{MAG_FUSE_IMM, i}; }

TEST(fuse_graph, fusibility_comes_from_the_operator_table) {
    /* Exact arithmetic fuses. */
    EXPECT_TRUE(mag_fuse_op_is_fusible(MAG_OP_ADD));
    EXPECT_TRUE(mag_fuse_op_is_fusible(MAG_OP_MUL));
    EXPECT_TRUE(mag_fuse_op_is_fusible(MAG_OP_CLAMP));
    EXPECT_TRUE(mag_fuse_op_is_fusible(MAG_OP_SQRT));
    EXPECT_TRUE(mag_fuse_op_is_fusible(MAG_OP_RELU));

    /* Transcendentals do not: a lowering would have to reproduce the vector kernel's approximation
       rather than call libm, and anything less than bit-identity is not worth fusing for. */
    EXPECT_FALSE(mag_fuse_op_is_fusible(MAG_OP_TANH));
    EXPECT_FALSE(mag_fuse_op_is_fusible(MAG_OP_EXP));
    EXPECT_FALSE(mag_fuse_op_is_fusible(MAG_OP_GELU));
    EXPECT_FALSE(mag_fuse_op_is_fusible(MAG_OP_RCP));

    /* Neither do operators that are not elementwise at all. */
    EXPECT_FALSE(mag_fuse_op_is_fusible(MAG_OP_SUM));
    EXPECT_FALSE(mag_fuse_op_is_fusible(MAG_OP_MATMUL));
    EXPECT_FALSE(mag_fuse_op_is_fusible(MAG_OP_SOFTMAX));
    EXPECT_FALSE(mag_fuse_op_is_fusible(MAG_OP_MINIMA)); /* The reduction, not the binary MIN. */
    EXPECT_TRUE(mag_fuse_op_is_fusible(MAG_OP_MIN));     /* The binary one does fuse. */

    EXPECT_FALSE(mag_fuse_op_is_fusible(MAG_OP__NUM));
}

TEST(fuse_graph, registers_are_single_assignment_and_ordered) {
    mag_fuse_graph_t g;
    mag_fuse_graph_init(&g, MAG_DTYPE_FLOAT32);

    int32_t a = mag_fuse_graph_load(&g, 0);
    int32_t b = mag_fuse_graph_load(&g, 1);
    ASSERT_EQ(a, 0);
    ASSERT_EQ(b, 1);

    mag_fuse_operand_t add[2] = {reg(a), reg(b)};
    int32_t sum = mag_fuse_graph_emit(&g, MAG_OP_ADD, add, 2);
    ASSERT_EQ(sum, 2);

    /* Each instruction writes the register named by its own index. */
    for (uint32_t i=0; i < g.num_ins; ++i) EXPECT_EQ(g.ins[i].dst, i);
    EXPECT_EQ(g.num_bufs, 2);
}

TEST(fuse_graph, rejects_what_it_cannot_represent) {
    mag_fuse_graph_t g;
    mag_fuse_graph_init(&g, MAG_DTYPE_FLOAT32);
    int32_t a = mag_fuse_graph_load(&g, 0);

    /* Wrong arity for the operator, taken from the operator table rather than a private list. */
    mag_fuse_operand_t one[1] = {reg(a)};
    EXPECT_LT(mag_fuse_graph_emit(&g, MAG_OP_ADD, one, 1), 0);
    mag_fuse_operand_t three[3] = {reg(a), reg(a), reg(a)};
    EXPECT_LT(mag_fuse_graph_emit(&g, MAG_OP_ADD, three, 3), 0);

    /* An operator that ends a chain. */
    EXPECT_LT(mag_fuse_graph_emit(&g, MAG_OP_TANH, one, 1), 0);

    /* A register that does not exist yet: operands may only name earlier values. */
    mag_fuse_operand_t fwd[2] = {reg(a), reg(99)};
    EXPECT_LT(mag_fuse_graph_emit(&g, MAG_OP_ADD, fwd, 2), 0);

    /* A store of a register that was never produced. */
    EXPECT_FALSE(mag_fuse_graph_store(&g, 0, 99));
    EXPECT_FALSE(mag_fuse_graph_store(&g, 0, -1));
}

TEST(fuse_graph, overflow_is_reported_not_fatal) {
    mag_fuse_graph_t g;
    mag_fuse_graph_init(&g, MAG_DTYPE_FLOAT32);
    int32_t last = mag_fuse_graph_load(&g, 0);
    ASSERT_GE(last, 0);

    /* Keep squaring until the chain will not take another instruction. It must decline rather than
       abort, because the caller only finds out a chain is too long by running into the limit. */
    int32_t r = 0;
    for (uint32_t i=0; i < MAG_FUSE_MAX_INS+8; ++i) {
        mag_fuse_operand_t in[1] = {reg(last)};
        r = mag_fuse_graph_emit(&g, MAG_OP_SQR, in, 1);
        if (r < 0) break;
        last = r;
    }
    EXPECT_LT(r, 0);
    EXPECT_EQ(g.num_ins, MAG_FUSE_MAX_INS);
}

TEST(fuse_graph, pruning_drops_what_no_store_can_reach) {
    mag_fuse_graph_t g;
    mag_fuse_graph_init(&g, MAG_DTYPE_FLOAT32);

    int32_t x = mag_fuse_graph_load(&g, 0);
    int32_t y = mag_fuse_graph_load(&g, 1);

    mag_fuse_operand_t mul[2] = {reg(x), reg(y)};
    int32_t wanted = mag_fuse_graph_emit(&g, MAG_OP_MUL, mul, 2);

    /* A value nothing will store and nothing else reads. */
    mag_fuse_operand_t sub[2] = {reg(x), reg(y)};
    int32_t dead = mag_fuse_graph_emit(&g, MAG_OP_SUB, sub, 2);
    ASSERT_GE(dead, 0);

    ASSERT_TRUE(mag_fuse_graph_store(&g, 2, wanted));
    uint32_t before = g.num_ins;
    EXPECT_EQ(mag_fuse_graph_prune(&g), 1);
    EXPECT_EQ(g.num_ins, before-1);

    /* The surviving store still names the value it named before, under its new number. */
    ASSERT_EQ(g.num_stores, 1);
    EXPECT_EQ(g.ins[g.stores[0].reg].op, MAG_OP_MUL);
    for (uint32_t i=0; i < g.num_ins; ++i) EXPECT_EQ(g.ins[i].dst, i);
}

TEST(fuse_graph, pruning_keeps_a_value_two_stores_deep) {
    mag_fuse_graph_t g;
    mag_fuse_graph_init(&g, MAG_DTYPE_FLOAT32);
    int32_t x = mag_fuse_graph_load(&g, 0);
    mag_fuse_operand_t sq[1] = {reg(x)};
    int32_t a = mag_fuse_graph_emit(&g, MAG_OP_SQR, sq, 1);
    mag_fuse_operand_t sq2[1] = {reg(a)};
    int32_t b = mag_fuse_graph_emit(&g, MAG_OP_SQR, sq2, 1);
    ASSERT_TRUE(mag_fuse_graph_store(&g, 1, b));

    /* Nothing is dead: b needs a, a needs x. */
    EXPECT_EQ(mag_fuse_graph_prune(&g), 0);
    EXPECT_EQ(g.num_ins, 3);
}

TEST(fuse_graph, a_chain_with_no_stores_prunes_to_nothing) {
    mag_fuse_graph_t g;
    mag_fuse_graph_init(&g, MAG_DTYPE_FLOAT32);
    int32_t x = mag_fuse_graph_load(&g, 0);
    mag_fuse_operand_t sq[1] = {reg(x)};
    ASSERT_GE(mag_fuse_graph_emit(&g, MAG_OP_SQR, sq, 1), 0);

    EXPECT_EQ(mag_fuse_graph_prune(&g), 2);
    EXPECT_EQ(g.num_ins, 0);
}

TEST(fuse_graph, hash_covers_structure_but_not_immediate_values) {
    /* Two chains with the same shape must be the same kernel to a backend, so that changing a
       learning rate is a cache hit rather than a recompile. The immediate is referenced by slot and
       never appears in the graph, so this is really a statement about what the graph holds. */
    auto build = [](uint8_t buf) {
        mag_fuse_graph_t g;
        mag_fuse_graph_init(&g, MAG_DTYPE_FLOAT32);
        int32_t x = mag_fuse_graph_load(&g, 0);
        int32_t s = mag_fuse_graph_imm(&g);
        mag_fuse_operand_t mul[2] = {reg(x), imm(static_cast<uint8_t>(s))};
        int32_t r = mag_fuse_graph_emit(&g, MAG_OP_MUL, mul, 2);
        mag_fuse_graph_store(&g, buf, r);
        return g;
    };
    mag_fuse_graph_t a = build(1), b = build(1);
    EXPECT_EQ(mag_fuse_graph_hash(&a), mag_fuse_graph_hash(&b));

    /* A different store target is a different program. */
    mag_fuse_graph_t c = build(2);
    EXPECT_NE(mag_fuse_graph_hash(&a), mag_fuse_graph_hash(&c));

    /* So is a different operator. */
    mag_fuse_graph_t d;
    mag_fuse_graph_init(&d, MAG_DTYPE_FLOAT32);
    int32_t x = mag_fuse_graph_load(&d, 0);
    int32_t s = mag_fuse_graph_imm(&d);
    mag_fuse_operand_t add[2] = {reg(x), imm(static_cast<uint8_t>(s))};
    mag_fuse_graph_store(&d, 1, mag_fuse_graph_emit(&d, MAG_OP_ADD, add, 2));
    EXPECT_NE(mag_fuse_graph_hash(&a), mag_fuse_graph_hash(&d));

    /* And so is the dtype, since a lowering picks its storage and rounding from it. */
    mag_fuse_graph_t e = build(1);
    e.dtype = MAG_DTYPE_FLOAT16;
    EXPECT_NE(mag_fuse_graph_hash(&a), mag_fuse_graph_hash(&e));
}

TEST(fuse_graph, pruning_does_not_change_the_hash_of_an_already_tight_chain) {
    /* Pruning zeroes what it removes, so a chain that arrives with dead code and one that never had
       any must look identical to a backend's cache once both are pruned. */
    mag_fuse_graph_t tight;
    mag_fuse_graph_init(&tight, MAG_DTYPE_FLOAT32);
    int32_t x0 = mag_fuse_graph_load(&tight, 0);
    mag_fuse_operand_t s0[1] = {reg(x0)};
    mag_fuse_graph_store(&tight, 1, mag_fuse_graph_emit(&tight, MAG_OP_SQR, s0, 1));

    mag_fuse_graph_t padded;
    mag_fuse_graph_init(&padded, MAG_DTYPE_FLOAT32);
    int32_t x1 = mag_fuse_graph_load(&padded, 0);
    mag_fuse_operand_t s1[1] = {reg(x1)};
    int32_t keep = mag_fuse_graph_emit(&padded, MAG_OP_SQR, s1, 1);
    mag_fuse_operand_t dead[1] = {reg(x1)};
    mag_fuse_graph_emit(&padded, MAG_OP_NEG, dead, 1); /* Nothing reads this. */
    mag_fuse_graph_store(&padded, 1, keep);

    mag_fuse_graph_prune(&tight);
    mag_fuse_graph_prune(&padded);
    EXPECT_EQ(mag_fuse_graph_hash(&tight), mag_fuse_graph_hash(&padded));
}
