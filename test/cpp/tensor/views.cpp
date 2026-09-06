/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include <cstring>

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
    handle_error(mag_full_like(nullptr, &vv, &*x, mag_scalar_from_float64(1.0)));
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

TEST(views, reinterpret_view_widens_and_narrows) {
    auto ctx = context{};
    tensor base {ctx, dtype::u8, std::vector<int64_t>{16}};
    auto *bytes = static_cast<uint8_t *>(base.data_ptr());
    for (int i=0; i < 16; ++i) bytes[i] = static_cast<uint8_t>(i);

    // 16 bytes read as 4 floats, and back again: same storage, same address, no conversion.
    mag_tensor_t *out = nullptr;
    mag_error_t err {};
    std::vector<int64_t> as_f32 = {4};
    ASSERT_EQ(mag_reinterpret_view(&err, &out, &*base, MAG_DTYPE_FLOAT32, as_f32.data(), 1), MAG_OK) << err.message;
    tensor f32 {out};
    ASSERT_EQ(f32.dtype(), dtype::float32);
    ASSERT_EQ(f32.numel(), 4);
    ASSERT_TRUE(f32.is_view());
    ASSERT_EQ(f32.data_ptr(), base.data_ptr());
    ASSERT_EQ(std::memcmp(f32.data_ptr(), bytes, 16), 0);

    std::vector<int64_t> back = {16};
    ASSERT_EQ(mag_reinterpret_view(&err, &out, &*f32, MAG_DTYPE_UINT8, back.data(), 1), MAG_OK) << err.message;
    tensor u8 {out};
    ASSERT_EQ(u8.numel(), 16);
    ASSERT_EQ(std::memcmp(u8.data_ptr(), bytes, 16), 0);
}

TEST(views, reinterpret_view_reshapes_and_infers) {
    auto ctx = context{};
    tensor base {ctx, dtype::u8, std::vector<int64_t>{48}};
    mag_tensor_t *out = nullptr;
    mag_error_t err {};
    std::vector<int64_t> shape = {3, -1};   // -1 is resolved against the reinterpreted count, not the base's
    ASSERT_EQ(mag_reinterpret_view(&err, &out, &*base, MAG_DTYPE_FLOAT32, shape.data(), 2), MAG_OK) << err.message;
    tensor f32 {out};
    ASSERT_EQ(f32.rank(), 2);
    ASSERT_EQ(f32.shape()[0], 3);
    ASSERT_EQ(f32.shape()[1], 4);
    ASSERT_EQ(f32.numel(), 12);
}

TEST(views, reinterpret_view_same_dtype_is_a_view) {
    auto ctx = context{};
    tensor base {ctx, dtype::float32, std::vector<int64_t>{6}};
    mag_tensor_t *out = nullptr;
    mag_error_t err {};
    std::vector<int64_t> shape = {2, 3};
    ASSERT_EQ(mag_reinterpret_view(&err, &out, &*base, MAG_DTYPE_FLOAT32, shape.data(), 2), MAG_OK) << err.message;
    tensor v {out};
    ASSERT_EQ(v.rank(), 2);
    ASSERT_EQ(v.data_ptr(), base.data_ptr());
}

TEST(views, reinterpret_view_tracks_the_offset_of_a_slice) {
    auto ctx = context{};
    tensor base {ctx, dtype::u8, std::vector<int64_t>{64}};
    auto *bytes = static_cast<uint8_t *>(base.data_ptr());
    for (int i=0; i < 64; ++i) bytes[i] = static_cast<uint8_t>(i);

    // The path a snapshot takes: narrow to a byte range, then read that range as another dtype.
    mag_tensor_t *sliced = nullptr;
    mag_error_t err {};
    ASSERT_EQ(mag_view_slice(&err, &sliced, &*base, 0, 32, 16, 1), MAG_OK) << err.message;
    tensor slice {sliced};
    mag_tensor_t *out = nullptr;
    std::vector<int64_t> shape = {4};
    ASSERT_EQ(mag_reinterpret_view(&err, &out, &*slice, MAG_DTYPE_FLOAT32, shape.data(), 1), MAG_OK) << err.message;
    tensor f32 {out};
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(f32.data_ptr()), reinterpret_cast<std::uintptr_t>(base.data_ptr())+32);
    ASSERT_EQ(std::memcmp(f32.data_ptr(), bytes+32, 16), 0);
}

TEST(views, reinterpret_view_rejects_invalid_reinterpretations) {
    auto ctx = context{};
    mag_tensor_t *out = nullptr;
    mag_error_t err {};

    { // A byte count that is not a whole number of target elements.
        tensor base {ctx, dtype::u8, std::vector<int64_t>{6}};
        std::vector<int64_t> shape = {1};
        ASSERT_NE(mag_reinterpret_view(&err, &out, &*base, MAG_DTYPE_FLOAT32, shape.data(), 1), MAG_OK);
    }
    { // An element offset that is fine for uint8 but does not divide for float32.
        tensor base {ctx, dtype::u8, std::vector<int64_t>{64}};
        mag_tensor_t *sliced = nullptr;
        ASSERT_EQ(mag_view_slice(&err, &sliced, &*base, 0, 2, 16, 1), MAG_OK) << err.message;
        tensor slice {sliced};
        std::vector<int64_t> shape = {4};
        ASSERT_NE(mag_reinterpret_view(&err, &out, &*slice, MAG_DTYPE_FLOAT32, shape.data(), 1), MAG_OK);
    }
    { // A transposed base: the innermost step is not one element, so the bytes that would be
      // merged are not neighbours.
        tensor base {ctx, dtype::u8, std::vector<int64_t>{8, 8}};
        mag_tensor_t *tr = nullptr;
        ASSERT_EQ(mag_transpose(&err, &tr, &*base, 0, 1), MAG_OK) << err.message;
        tensor transposed {tr};
        std::vector<int64_t> shape = {16};
        ASSERT_NE(mag_reinterpret_view(&err, &out, &*transposed, MAG_DTYPE_FLOAT32, shape.data(), 1), MAG_OK);
    }
    { // An outer stride whose byte length is not a multiple of the target element size.
        tensor base {ctx, dtype::u8, std::vector<int64_t>{4, 10}};
        mag_tensor_t *sl = nullptr;
        ASSERT_EQ(mag_view_slice(&err, &sl, &*base, 1, 0, 8, 1), MAG_OK) << err.message;
        tensor rows {sl};
        std::vector<int64_t> shape = {4, 2};
        ASSERT_NE(mag_reinterpret_view(&err, &out, &*rows, MAG_DTYPE_FLOAT32, shape.data(), 2), MAG_OK);
    }
    { // A shape whose element count does not match the reinterpreted one.
        tensor base {ctx, dtype::u8, std::vector<int64_t>{16}};
        std::vector<int64_t> shape = {5};
        ASSERT_NE(mag_reinterpret_view(&err, &out, &*base, MAG_DTYPE_FLOAT32, shape.data(), 1), MAG_OK);
    }
    { // Autograd: a bit pattern has no derivative to record.
        tensor base {ctx, dtype::float32, std::vector<int64_t>{8}};
        base.requires_grad(true);
        std::vector<int64_t> shape = {4};
        ASSERT_NE(mag_reinterpret_view(&err, &out, &*base, MAG_DTYPE_INT64, shape.data(), 1), MAG_OK);
    }
    { // An invalid dtype id.
        tensor base {ctx, dtype::u8, std::vector<int64_t>{16}};
        std::vector<int64_t> shape = {16};
        ASSERT_NE(mag_reinterpret_view(&err, &out, &*base, static_cast<mag_dtype_t>(MAG_DTYPE__NUM), shape.data(), 1), MAG_OK);
    }
}

TEST(views, reinterpret_view_accepts_a_strided_base) {
    // Rows of a larger matrix: strided between rows, unit-stride within one. Reinterpreting that
    // is well defined and copies nothing, which is the layout a quantized weight arrives in.
    auto ctx = context{};
    tensor base {ctx, dtype::u8, std::vector<int64_t>{4, 16}};
    auto *bytes = static_cast<uint8_t *>(base.data_ptr());
    for (int i=0; i < 64; ++i) bytes[i] = static_cast<uint8_t>(i);

    mag_tensor_t *sl = nullptr;
    mag_error_t err {};
    ASSERT_EQ(mag_view_slice(&err, &sl, &*base, 1, 0, 8, 1), MAG_OK) << err.message;  // u8[4,8] strides (16,1)
    tensor rows {sl};
    ASSERT_FALSE(rows.is_contiguous());
    ASSERT_EQ(rows.strides()[0], 16);

    mag_tensor_t *out = nullptr;
    std::vector<int64_t> shape = {4, 2};
    ASSERT_EQ(mag_reinterpret_view(&err, &out, &*rows, MAG_DTYPE_FLOAT32, shape.data(), 2), MAG_OK) << err.message;
    tensor f32 {out};
    ASSERT_EQ(f32.shape()[0], 4);
    ASSERT_EQ(f32.shape()[1], 2);
    ASSERT_EQ(f32.strides()[0], 4) << "16 uint8 apart is 4 float32 apart";
    ASSERT_EQ(f32.strides()[1], 1);
    ASSERT_EQ(f32.data_ptr(), base.data_ptr()) << "no copy";

    // Row r of the result must be the 8 bytes at 16*r, read as two floats.
    for (int r=0; r < 4; ++r) {
        const auto *row = static_cast<const float *>(f32.data_ptr()) + 4*r;
        ASSERT_EQ(std::memcmp(row, bytes + 16*r, 8), 0) << "row " << r;
    }
}
