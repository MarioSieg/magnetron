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

#include <prelude.hpp>

using namespace magnetron;
using namespace magnetron::test;

#if 0
TEST(cpu_tensor_indexing, subscript_flattened_e8m23) {
    auto ctx = context{};
    for_all_test_shapes([&](std::span<const int64_t> shape) {
        tensor t {ctx, dtype::e8m23, shape};
        t.fill_rand_uniform(-1.0f, 1.0f);
        std::vector<float> data {t.to_vector<float>()};
        ASSERT_EQ(t.numel(), data.size());
        for (size_t i {0}; i < data.size(); ++i) {
            ASSERT_FLOAT_EQ(t.to_vector<float>()[i], data[i]);
        }
    });
}

TEST(cpu_tensor_indexing, subscript_flattened_e5m10) {
    auto ctx = context{};
    for_all_test_shapes([&](std::span<const int64_t> shape) {
        tensor t {ctx, dtype::e5m10, shape};
        t.fill_rand_uniform(-1.0f, 1.0f);
        std::vector<float> data {t.to_vector<float>()};
        ASSERT_EQ(t.numel(), data.size());
        for (size_t i {0}; i < data.size(); ++i) {
            ASSERT_FLOAT_EQ(t.to_vector<float>()[i], data[i]);
        }
    });
}
#endif
