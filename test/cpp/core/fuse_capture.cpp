#include <prelude.hpp>

#include <core/mag_fuse_capture.h>

using namespace magnetron;
using namespace magnetron::test;

namespace {
    /* Anything smaller is left to the eager path, so a capture test on it would test nothing. */
    constexpr int64_t kN = MAG_FUSE_MIN_ELEMS;

    /* Fill a tensor with something whose exact bits are easy to reason about. */
    auto ramp(context &ctx, int64_t n, float scale) -> tensor {
        tensor t {ctx, dtype::float32, n};
        std::vector<float> v (static_cast<size_t>(n));
        for (int64_t i=0; i < n; ++i) v[static_cast<size_t>(i)] = scale*static_cast<float>(i+1);
        t.copy_(v);
        return t;
    }

    struct region final {   /* mag_fuse_region_end runs the chain, so it must not be skipped. */
        explicit region(context &ctx) : m_ctx{&*ctx} { mag_fuse_region_begin(m_ctx); }
        ~region() { mag_error_t err {}; mag_fuse_region_end(&err, m_ctx); }
    private:
        mag_context_t *m_ctx;
    };

    auto chain_stats(context &ctx) -> std::pair<uint64_t, uint64_t> {
        uint64_t chains = 0, ops = 0;
        mag_fuse_stats(&*ctx, &chains, &ops);
        return {chains, ops};
    }
}

TEST(fuse_capture, a_region_does_not_change_the_answer) {
    context ctx {};
    tensor a = ramp(ctx, kN, 1.0f);
    tensor b = ramp(ctx, kN, 0.5f);

    std::vector<float> eager = a.mul(b).add(a).to_vector<float>();

    tensor fused_out {ctx, dtype::float32, 1};
    {
        region r {ctx};
        fused_out = a.mul(b).add(a);
    }
    /* Bit-identical, not merely close. A chain that rounds differently from the eager path is a
       chain nobody can safely turn on. */
    EXPECT_EQ(fused_out.to_vector<float>(), eager);
}

TEST(fuse_capture, reading_a_value_inside_a_region_runs_the_chain_first) {
    context ctx {};
    tensor a = ramp(ctx, kN, 1.0f);
    std::vector<float> expect = a.mul(a).to_vector<float>();

    region r {ctx};
    tensor sq = a.mul(a);
    /* The operator was recorded, not run, so this tensor owes a value. */
    EXPECT_TRUE((*sq).meta.flags & MAG_TFLAG_PENDING);

    /* Asking for the numbers has to settle the debt, and the guard that does it sits in the only
       path that reaches a tensor's bytes. */
    std::vector<float> got = sq.to_vector<float>();
    EXPECT_EQ(got, expect);
    EXPECT_FALSE((*sq).meta.flags & MAG_TFLAG_PENDING);
}

TEST(fuse_capture, an_operator_that_cannot_join_ends_the_chain) {
    context ctx {};
    tensor a = ramp(ctx, kN, 1.0f);
    std::vector<float> expect = a.mul(a).tanh().add(a).to_vector<float>();

    tensor out {ctx, dtype::float32, 1};
    {
        region r {ctx};
        out = a.mul(a).tanh().add(a);  /* tanh is not exactly defined, so it splits this in two. */
    }
    EXPECT_EQ(out.to_vector<float>(), expect);

    /* tanh splits this into two chains - mul before it, add after - and both of them ran as
       chains rather than one operator at a time. */
    auto [chains, ops] = chain_stats(ctx);
    EXPECT_EQ(chains, 2u);
    EXPECT_EQ(ops, 2u);
}

TEST(fuse_capture, nesting_only_runs_the_chain_on_the_outermost_exit) {
    context ctx {};
    tensor a = ramp(ctx, kN, 1.0f);
    std::vector<float> expect = a.add(a).mul(a).to_vector<float>();

    tensor out {ctx, dtype::float32, 1};
    {
        region outer {ctx};
        tensor mid {ctx, dtype::float32, 1};
        {
            region inner {ctx};
            mid = a.add(a);
        }
        /* Leaving the inner region must not have run anything: a helper that opens a region, called
           from code that already did, should not chop the chain in half. */
        EXPECT_TRUE((*mid).meta.flags & MAG_TFLAG_PENDING);
        out = mid.mul(a);
    }
    EXPECT_EQ(out.to_vector<float>(), expect);
}

TEST(fuse_capture, a_chain_longer_than_the_tape_still_computes) {
    context ctx {};
    tensor a = ramp(ctx, kN, 0.125f);

    tensor eager = a;
    for (int i=0; i < MAG_FUSE_TAPE_MAX+8; ++i) eager = eager.add(a);
    std::vector<float> expect = eager.to_vector<float>();

    tensor out {ctx, dtype::float32, 1};
    {
        region r {ctx};
        tensor acc = a;
        for (int i=0; i < MAG_FUSE_TAPE_MAX+8; ++i) acc = acc.add(a);
        out = acc;
    }
    EXPECT_EQ(out.to_vector<float>(), expect);
}

TEST(fuse_capture, a_single_element_tensor_is_left_alone) {
    context ctx {};
    tensor a = ramp(ctx, 1, 3.0f);
    tensor out {ctx, dtype::float32, 1};
    {
        region r {ctx};
        out = a.mul(a);
        /* Not worth a kernel, so it runs where it stands and owes nothing. */
        EXPECT_FALSE((*out).meta.flags & MAG_TFLAG_PENDING);
    }
    EXPECT_FLOAT_EQ(out.to_vector<float>()[0], 9.0f);
}

TEST(fuse_capture, nothing_is_captured_outside_a_region) {
    context ctx {};
    tensor a = ramp(ctx, kN, 1.0f);
    tensor out = a.mul(a);
    EXPECT_FALSE((*out).meta.flags & MAG_TFLAG_PENDING);
}
