// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace magnetron::test;

static constexpr std::int64_t lim {4};

TEST(cpu_tensor_indexing, subscript_flattened_e8m23) {
    auto ctx = context{device_type::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        tensor t {ctx, dtype::e8m23, shape};
        t.fill_rand_uniform(-1.0f, 1.0f);
        std::vector<float> data {t.to_vector<float>()};
        ASSERT_EQ(t.numel(), data.size());
        for (std::size_t i {0}; i < data.size(); ++i) {
            ASSERT_FLOAT_EQ(t(i), data[i]);
        }
    });
}

TEST(cpu_tensor_indexing, subscript_flattened_e5m10) {
    auto ctx = context{device_type::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        tensor t {ctx, dtype::e5m10, shape};
        t.fill_rand_uniform(-1.0f, 1.0f);
        std::vector<float> data {t.to_vector<float>()};
        ASSERT_EQ(t.numel(), data.size());
        for (std::size_t i {0}; i < data.size(); ++i) {
            ASSERT_FLOAT_EQ(t(i), data[i]);
        }
    });
}
