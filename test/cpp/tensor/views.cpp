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

TEST(views, view) {
    std::vector<int64_t> shape = {8, 3, 4};
    auto ctx = context{};
    tensor base {ctx, dtype::float32, shape};
    tensor view = base.view(base.shape());
    ASSERT_EQ(view.rank(), 3);
    ASSERT_EQ(view.shape()[0], 8);
    ASSERT_EQ(view.shape()[1], 3);
    ASSERT_EQ(view.shape()[2], 4);
    ASSERT_TRUE(view.is_view());
    ASSERT_EQ(view.strides()[0], base.strides()[0]);
    auto base_addr = reinterpret_cast<std::uintptr_t>(base.data_ptr());
    auto view_addr = reinterpret_cast<std::uintptr_t>(view.data_ptr());
    ASSERT_EQ(view_addr, base_addr);
}

TEST(views, view_of_view) {
    std::vector<int64_t> shape = {8, 3, 4};
    auto ctx = context{};
    tensor base {ctx, dtype::float32, shape};
    tensor view1 = base.view(base.shape());
    tensor view2 = view1.view(view1.shape());
    ASSERT_EQ(view2.rank(), 3);
    ASSERT_EQ(view2.shape()[0], 8);
    ASSERT_EQ(view2.shape()[1], 3);
    ASSERT_EQ(view2.shape()[2], 4);
    ASSERT_TRUE(view2.is_view());
    ASSERT_EQ(view2.strides()[0], base.strides()[0]);
    auto base_addr = reinterpret_cast<std::uintptr_t>(base.data_ptr());
    auto view_addr = reinterpret_cast<std::uintptr_t>(view2.data_ptr());
    ASSERT_EQ(view_addr, base_addr);
}

TEST(views, view_slice_positive_step) {
    std::vector<int64_t> shape = {8, 3, 4};
    auto ctx = context{};
    tensor base {ctx, dtype::float32, shape};
    tensor view = base.view_slice(0, 2, 3, 1);
    ASSERT_EQ(view.rank(), 3);
    ASSERT_EQ(view.shape()[0], 3);
    ASSERT_EQ(view.shape()[1], 3);
    ASSERT_EQ(view.shape()[2], 4);
    ASSERT_TRUE(view.is_view());
    ASSERT_EQ(view.strides()[0], base.strides()[0]);
    auto base_addr = reinterpret_cast<std::uintptr_t>(base.data_ptr());
    auto view_addr = reinterpret_cast<std::uintptr_t>(view.data_ptr());
    std::uintptr_t expected = base_addr + 2*base.strides()[0] * sizeof(float);
    ASSERT_EQ(view_addr, expected);
}

TEST(views, view_of_view_slice) {
    std::vector<int64_t> shape = {8, 3, 4};
    auto ctx = context{};
    tensor base {ctx, dtype::float32, shape};
    tensor view1 = base.view_slice(0, 2, 3, 1);
    tensor view2 = view1.view({9, 4}); // view of view
    ASSERT_EQ(view2.rank(), 2);
    ASSERT_EQ(view2.shape()[0], 9);
    ASSERT_EQ(view2.shape()[1], 4);
    ASSERT_TRUE(view2.is_view());
}

TEST(views, view_slice_chain_accumulates_offset) {
    context ctx{};
    tensor base{ctx, dtype::float32, 10, 2};
    tensor v1 = base.view_slice(0, 2, 6, 1); // rows 2..7
    tensor v2 = v1.view_slice(0, 3, 2, 1); // rows 5..6 of base
    const auto expect = reinterpret_cast<std::uintptr_t>(base.data_ptr()) + 5*base.strides()[0]*sizeof(float);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(v2.data_ptr()), expect);
    ASSERT_TRUE(v2.is_view());
}

TEST(views, flattened_write_uses_offset) {
    context ctx{};
    tensor base{ctx, dtype::float32, 4, 3}; // (rows, cols)
    tensor v = base.view_slice(0, 1, 2, 1); // rows 1 & 2
    //v(0, 42.0f); // first elem of view TODO
    //ASSERT_FLOAT_EQ(base(1*3 + 0), 42.0f);
}

TEST(views, storage_alias_consistency) {
    context ctx{};
    tensor base{ctx, dtype::float32, 5};
    tensor v1 = base.view_slice(0,1,3,1);
    tensor v2 = base.view_slice(0,2,2,1);
    ASSERT_EQ(base.storage_base_ptr(), v1.storage_base_ptr());
    ASSERT_EQ(v1.storage_base_ptr(),  v2.storage_base_ptr());
}

TEST(views, tail_identity) {
    context ctx{};
    tensor t{ctx, dtype::float32, 2, 3};
    tensor v1 = t.view(t.shape());                 // contiguous alias
    tensor v2 = t.view_slice(1, 0, 2, 2); // strided rows 0,2
    for (auto* p : {&t, &v1, &v2}) {
        for (auto i = p->rank(); i < MAG_MAX_DIMS; ++i) {
            ASSERT_EQ(mag_tensor_shape_ptr(&**p)[i], 1); // Use C ptr to not access vector out of bounds because shape() closes until 0..rank elements
            ASSERT_EQ(mag_tensor_strides_ptr(&**p)[i], 1);
        }
    }
}

TEST(views, view_keeps_strides) {
    context ctx{};
    tensor base  {ctx, dtype::float32, 4, 4};
    tensor slice = base.view_slice(1, 0, 2, 2);   // stride {8,1}
    tensor alias = slice.view(slice.shape());                  // same logical shape
    ASSERT_EQ(alias.strides()[0], slice.strides()[0]);
    ASSERT_EQ(alias.strides()[1], slice.strides()[1]);
}

TEST(views, reshape_requires_contiguous) {
    context ctx{};
    tensor base{ctx, dtype::float32, 4, 4};
    tensor slice = base.view_slice(1, 0, 2, 2);   // non-contiguous
    auto view = slice.view({4, 2});;
}

TEST(views, reshape_requires_contiguous_wrong) {
    context ctx{};
    tensor base{ctx, dtype::float32, 4, 4};
    tensor slice = base.view_slice(1, 0, 2, 2);   // non-contiguous
    ASSERT_DEATH({
        auto view = slice.view({8, 2});;
    }, "");
}

TEST(views, offset_accumulation) {
    context ctx{};
    tensor base{ctx, dtype::float32, 10, 2};      // row-major
    tensor v1 = base.view_slice(0, 2, 6, 1);    // rows 2..7
    tensor v2 = v1.view_slice(0, 3, 2, 1);      // rows 5..6

    auto expect = reinterpret_cast<std::uintptr_t>(base.data_ptr()) +
                  5 * base.strides()[0] * sizeof(float);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(v2.data_ptr()), expect);
}

TEST(views, to_float_vector_copies_view) {
    context ctx{};
    tensor base{ctx, dtype::float32, 8, 3, 4};
    base.uniform_(-1.f, 1.f);
    tensor slice = base.view_slice(0,0,4,2);
    auto ref = base.to_vector<float>();
    auto got = slice.to_vector<float>();
    for (int64_t i = 0; i < slice.numel(); ++i) {
        int64_t row = i / (3*4);
        int64_t col = i % (3*4);
        ASSERT_FLOAT_EQ(got[i], ref[row*2*3*4 + col]);
    }
}

TEST(views, inplace_bumps_version_and_detaches) {
    context ctx{};
    tensor x{ctx, dtype::float32, 2, 2};
    x.requires_grad(true);
    tensor v = x.view(x.shape());
    tensor y = v.abs();
    ctx.stop_grad_recorder();
    mag_tensor_t *vv;
    handle_error(mag_full_like(&vv, &*x, mag_scalar_float(1.0)));
    x += tensor{vv};
    ctx.start_grad_recorder();
    tensor loss = y.sum();
    loss.backward();
    ASSERT_TRUE(x.grad()->is_contiguous());
}

TEST(views, view_no_axes) {
    auto ctx = context{};
    auto base = tensor{ctx, dtype::float32, 2, 2, 3, 1};
    auto v = base.view(base.shape());
    ASSERT_FALSE(base.is_view());
    ASSERT_TRUE(v.is_view());
    ASSERT_EQ(base.storage_base_ptr(), v.storage_base_ptr());
}
