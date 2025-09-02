// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;

TEST(prng, automatic_seeding) {
    std::vector<float> a, b;
    {
        context ctx {compute_device::cpu};
        tensor ta {ctx, dtype::e8m23, 8192, 8192};
        ta.fill_rand_uniform(-1.0f, 1.0f);
        a = ta.to_vector<float>();
    }
    {
        context ctx {compute_device::cpu};
        tensor tb {ctx, dtype::e8m23, 8192, 8192};
        tb.fill_rand_uniform(-1.0f, 1.0f);
        b = tb.to_vector<float>();
    }

    ASSERT_EQ(a.size(), b.size());

    ASSERT_NE(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));

    std::size_t matches = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) ++matches;
    }
    EXPECT_LE(matches, 30u) << "Too many exact matches - suspicious seeding";
}

TEST(prng, manual_seeding) {
    std::random_device rd;
    std::mt19937_64 eng {rd()};
    std::uniform_int_distribution<std::uint64_t> distr {};
    std::uint64_t seed = distr(eng);
    std::vector<float> a {}, b {};
    {
        context ctx {compute_device::cpu};
        ctx.manual_seed(seed);
        tensor ta {ctx, dtype::e8m23, 8192, 8192};
        ta.fill_rand_uniform(-1.0f, 1.0f);
        a = ta.to_vector<float>();
    }

    {
        context ctx {compute_device::cpu};
        ctx.manual_seed(seed);
        tensor tb {ctx, dtype::e8m23, 8192, 8192};
        tb.fill_rand_uniform(-1.0f, 1.0f);
        b = tb.to_vector<float>();
    }

    ASSERT_EQ(a.size(), b.size());

    for (std::size_t i=0; i < a.size(); i++)
        ASSERT_FLOAT_EQ(a[i], b[i]) << "i=" << i;
}