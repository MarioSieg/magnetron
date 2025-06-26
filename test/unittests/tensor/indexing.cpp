// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace magnetron::test;

static constexpr std::int64_t lim {4};

TEST(cpu_tensor_indexing, subscript_flattened_e8m23) {
    auto ctx = context{compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        tensor t {ctx, dtype::e8m23, shape};
        t.fill_rand_uniform_float(-1.0f, 1.0f);
        std::vector<mag_E8M23> data {t.to_float_vector()};
        ASSERT_EQ(t.numel(), data.size());
        for (std::size_t i {0}; i < data.size(); ++i) {
            ASSERT_FLOAT_EQ(t(i), data[i]);
        }
    });
}

TEST(cpu_tensor_indexing, subscript_flattened_e5m10) {
    auto ctx = context{compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        tensor t {ctx, dtype::e5m10, shape};
        t.fill_rand_uniform_float(-1.0f, 1.0f);
        std::vector<mag_E8M23> data {t.to_float_vector()};
        ASSERT_EQ(t.numel(), data.size());
        for (std::size_t i {0}; i < data.size(); ++i) {
            ASSERT_FLOAT_EQ(t(i), data[i]);
        }
    });
}

TEST(cpu_tensor_indexing, view_slice_positive_step) {
    constexpr std::array<std::int64_t, 3> shape = {8, 3, 4};
    auto ctx = context{compute_device::cpu};
    tensor base {ctx, dtype::e8m23, shape};
    tensor view = base.view_slice(0, 2, 3, 1);
    ASSERT_EQ(view.rank(), 3);
    ASSERT_EQ(view.shape()[0], 3);
    ASSERT_EQ(view.shape()[1], 3);
    ASSERT_EQ(view.shape()[2], 4);
    ASSERT_TRUE(view.is_view());
    ASSERT_EQ(view.strides()[0], base.strides()[0]);
    auto base_addr = std::bit_cast<std::uintptr_t>(base.data_ptr());
    auto view_addr = std::bit_cast<std::uintptr_t>(view.data_ptr());
    std::uintptr_t expected = base_addr + 2*base.strides()[0] * sizeof(e8m23_t);
    ASSERT_EQ(view_addr, expected);
}

TEST(cpu_tensor_indexing, view_slice_chain_accumulates_offset) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 10, 2};
    tensor v1 = base.view_slice(0, 2, 6, 1); // rows 2..7
    tensor v2 = v1.view_slice(0, 3, 2, 1); // rows 5..6 of base
    const auto expect = std::bit_cast<std::uintptr_t>(base.data_ptr()) +
                        5*base.strides()[0]*sizeof(e8m23_t);
    ASSERT_EQ(std::bit_cast<std::uintptr_t>(v2.data_ptr()), expect);
    ASSERT_TRUE(v2.is_view());
}

TEST(cpu_tensor_indexing, flattened_write_uses_offset) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 4, 3}; // (rows, cols)
    tensor v = base.view_slice(0, 1, 2, 1); // rows 1 & 2
    v(0, 42.0f); // first elem of view
    ASSERT_FLOAT_EQ(base(1*3 + 0), 42.0f);
}

TEST(cpu_tensor_indexing, storage_alias_consistency) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 5};
    tensor v1 = base.view_slice(0,1,3,1);
    tensor v2 = base.view_slice(0,2,2,1);
    ASSERT_EQ(base.storage_base_ptr(), v1.storage_base_ptr());
    ASSERT_EQ(v1.storage_base_ptr(),  v2.storage_base_ptr());
}

TEST(cpu_tensor_indexing, storage_offset_matches_pointer) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 8};
    tensor v = base.view_slice(0, 3, 2, 1);          // starts at elem 3
    const auto logical = std::bit_cast<std::uintptr_t>(v.data_ptr());
    const auto expect = std::bit_cast<std::uintptr_t>(base.storage_base_ptr()) + v.view_offset();
    ASSERT_EQ(logical, expect);
}
