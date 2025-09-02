// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace magnetron::test;

TEST(views, view) {
    constexpr std::array<std::int64_t, 3> shape = {8, 3, 4};
    auto ctx = context{compute_device::cpu};
    tensor base {ctx, dtype::e8m23, shape};
    tensor view = base.view();
    ASSERT_EQ(view.rank(), 3);
    ASSERT_EQ(view.shape()[0], 8);
    ASSERT_EQ(view.shape()[1], 3);
    ASSERT_EQ(view.shape()[2], 4);
    ASSERT_TRUE(view.is_view());
    ASSERT_EQ(view.strides()[0], base.strides()[0]);
    auto base_addr = std::bit_cast<std::uintptr_t>(base.data_ptr());
    auto view_addr = std::bit_cast<std::uintptr_t>(view.data_ptr());
    ASSERT_EQ(view_addr, base_addr);
}

TEST(views, view_of_view) {
    constexpr std::array<std::int64_t, 3> shape = {8, 3, 4};
    auto ctx = context{compute_device::cpu};
    tensor base {ctx, dtype::e8m23, shape};
    tensor view1 = base.view();
    tensor view2 = view1.view();
    ASSERT_EQ(view2.rank(), 3);
    ASSERT_EQ(view2.shape()[0], 8);
    ASSERT_EQ(view2.shape()[1], 3);
    ASSERT_EQ(view2.shape()[2], 4);
    ASSERT_TRUE(view2.is_view());
    ASSERT_EQ(view2.strides()[0], base.strides()[0]);
    auto base_addr = std::bit_cast<std::uintptr_t>(base.data_ptr());
    auto view_addr = std::bit_cast<std::uintptr_t>(view2.data_ptr());
    ASSERT_EQ(view_addr, base_addr);
}

TEST(views, view_slice_positive_step) {
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

TEST(views, view_of_view_slice) {
    constexpr std::array<std::int64_t, 3> shape = {8, 3, 4};
    auto ctx = context{compute_device::cpu};
    tensor base {ctx, dtype::e8m23, shape};
    tensor view1 = base.view_slice(0, 2, 3, 1);
    tensor view2 = view1.view({9, 4}); // view of view
    ASSERT_EQ(view2.rank(), 2);
    ASSERT_EQ(view2.shape()[0], 9);
    ASSERT_EQ(view2.shape()[1], 4);
    ASSERT_TRUE(view2.is_view());
}

TEST(views, view_slice_chain_accumulates_offset) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 10, 2};
    tensor v1 = base.view_slice(0, 2, 6, 1); // rows 2..7
    tensor v2 = v1.view_slice(0, 3, 2, 1); // rows 5..6 of base
    const auto expect = std::bit_cast<std::uintptr_t>(base.data_ptr()) +
                        5*base.strides()[0]*sizeof(e8m23_t);
    ASSERT_EQ(std::bit_cast<std::uintptr_t>(v2.data_ptr()), expect);
    ASSERT_TRUE(v2.is_view());
}

TEST(views, flattened_write_uses_offset) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 4, 3}; // (rows, cols)
    tensor v = base.view_slice(0, 1, 2, 1); // rows 1 & 2
    v(0, 42.0f); // first elem of view
    ASSERT_FLOAT_EQ(base(1*3 + 0), 42.0f);
}

TEST(views, storage_alias_consistency) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 5};
    tensor v1 = base.view_slice(0,1,3,1);
    tensor v2 = base.view_slice(0,2,2,1);
    ASSERT_EQ(base.storage_base_ptr(), v1.storage_base_ptr());
    ASSERT_EQ(v1.storage_base_ptr(),  v2.storage_base_ptr());
}

TEST(views, tail_identity) {
    context ctx{compute_device::cpu};
    tensor t{ctx, dtype::e8m23, 2, 3};
    tensor v1 = t.view();                 // contiguous alias
    tensor v2 = t.view_slice(1, 0, 2, 2); // strided rows 0,2
    for (auto* p : {&t, &v1, &v2}) {
        for (auto i = p->rank(); i < MAG_MAX_DIMS; ++i) {
            ASSERT_EQ(p->shape()[i],   1);
            ASSERT_EQ(p->strides()[i], 1);
        }
    }
}

TEST(views, view_keeps_strides) {
    context ctx{compute_device::cpu};
    tensor base  {ctx, dtype::e8m23, 4, 4};
    tensor slice = base.view_slice(1, 0, 2, 2);   // stride {8,1}
    tensor alias = slice.view();                  // same logical shape
    ASSERT_EQ(alias.strides()[0], slice.strides()[0]);
    ASSERT_EQ(alias.strides()[1], slice.strides()[1]);
}

TEST(views, reshape_requires_contiguous) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 4, 4};
    tensor slice = base.view_slice(1, 0, 2, 2);   // non-contiguous
    auto view = slice.view({4, 2});;
}

TEST(views, reshape_requires_contiguous_wrong) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 4, 4};
    tensor slice = base.view_slice(1, 0, 2, 2);   // non-contiguous
    ASSERT_DEATH({
        auto view = slice.view({8, 2});;
    }, "");
}

TEST(views, offset_accumulation) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 10, 2};      // row-major
    tensor v1 = base.view_slice(0, 2, 6, 1);    // rows 2..7
    tensor v2 = v1.view_slice(0, 3, 2, 1);      // rows 5..6

    auto expect = reinterpret_cast<std::uintptr_t>(base.data_ptr()) +
                  5 * base.strides()[0] * sizeof(e8m23_t);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(v2.data_ptr()), expect);
}

TEST(views, to_float_vector_copies_view) {
    context ctx{compute_device::cpu};
    tensor base{ctx, dtype::e8m23, 8, 3, 4};
    base.fill_rand_uniform_float(-1.f, 1.f);
    tensor slice = base.view_slice(0,0,4,2);
    auto ref = base.to_float_vector();
    auto got = slice.to_float_vector();
    for (int64_t i = 0; i < slice.numel(); ++i) {
        int64_t row = i / (3*4);
        int64_t col = i % (3*4);
        ASSERT_FLOAT_EQ(got[i], ref[row*2*3*4 + col]);
    }
}

TEST(views, inplace_bumps_version_and_detaches) {
    context ctx{compute_device::cpu};
    tensor x{ctx, dtype::e8m23, 2, 2};
    x.requires_grad(true);
    tensor v = x.view();
    tensor y = v.abs();
    ctx.stop_grad_recorder();
    x += tensor{mag_tensor_full_like(&*x, 1.f)};
    ctx.start_grad_recorder();
    tensor loss = y.sum();
    loss.backward();
    ASSERT_TRUE(x.grad()->is_contiguous());
}
