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
using namespace test;

class binary_operators : public TestWithParam<device_kind> {};

static void test_binary_float_operator(
    device_kind dev,
    bool broadcast,
    bool inplace,
    mag_e8m23_t eps,
    dtype ty,
    std::function<tensor (tensor, tensor)>&& a,
    std::function<mag_e8m23_t (mag_e8m23_t, mag_e8m23_t)>&& b,
    mag_e8m23_t min = -10.0,
    mag_e8m23_t max = 10.0
) {
    auto& ctx = get_cached_context(dev);
    ctx.stop_grad_recorder();
    for_all_test_shapes([&](std::span<const std::int64_t> shape) {
        tensor t_a {ctx, ty, shape};
        t_a.fill_rand_uniform(min, max);
        tensor t_b {t_a.clone()};
        data_accessor<mag_e8m23_t> d_a {t_a};
        data_accessor<mag_e8m23_t> d_b {t_b};
        tensor t_r {std::invoke(a, t_a, t_b)};
        if (inplace) {
            ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
        } else {
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        }
        data_accessor<mag_e8m23_t> d_r {t_r};
        ASSERT_EQ(d_a.size(), d_b.size());
        ASSERT_EQ(d_a.size(), d_r.size());
        ASSERT_EQ(t_a.dtype(), t_b.dtype());
        ASSERT_EQ(t_a.dtype(), t_r.dtype());
        for (std::int64_t i = 0; i < d_r.size(); ++i) {
            ASSERT_NEAR(std::invoke(b, d_a[i], d_b[i]), d_r[i], eps) << d_a[i] << " op " << d_b[i] << " = " << d_r[i];
        }
    });
}

static void test_binary_int_operator(
    device_kind dev,
    bool broadcast,
    bool inplace,
    dtype ty,
    int32_t min,
    int32_t max,
    std::function<tensor (tensor, tensor)>&& a,
    std::function<int32_t (int32_t, int32_t)>&& b
) {
    auto& ctx = get_cached_context(dev);
    ctx.stop_grad_recorder();
    for_all_test_shapes([&](std::span<const std::int64_t> shape) {
        tensor t_a {ctx, ty, shape};
        std::uniform_int_distribution<int32_t> fill_val {min, max};
        t_a.fill(fill_val(gen));
        tensor t_b {t_a.clone()};
        std::vector<int32_t> d_a {t_a.to_vector<int32_t>()};
        std::vector<int32_t> d_b {t_b.to_vector<int32_t>()};
        tensor t_r {std::invoke(a, t_a, t_b)};
        if (inplace) {
            ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
        } else {
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        }
        std::vector<int32_t> d_r {t_r.to_vector<int32_t>()};
        ASSERT_EQ(d_a.size(), d_b.size());
        ASSERT_EQ(d_a.size(), d_r.size());
        ASSERT_EQ(t_a.dtype(), t_b.dtype());
        ASSERT_EQ(t_a.dtype(), t_r.dtype());
        for (std::int64_t i = 0; i < d_r.size(); ++i) {
            ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " op " << d_b[i] << " = " << d_r[i];
        }
    });
}

static void test_binary_boolean_operator(
    device_kind dev,
    bool broadcast,
    bool inplace,
    std::function<tensor (tensor, tensor)>&& a,
    std::function<bool (bool, bool)>&& b
) {
    auto& ctx = get_cached_context(dev);
    ctx.stop_grad_recorder();
    for_all_test_shapes([&](std::span<const std::int64_t> shape) {
        tensor t_a {ctx, dtype::boolean, shape};
        t_a.fill(true);
        tensor t_b {t_a.clone()};
        t_b.fill(false);
        std::vector<bool> d_a {t_a.to_vector<bool>()};
        std::vector<bool> d_b {t_b.to_vector<bool>()};
        tensor t_r {std::invoke(a, t_a, t_b)};
        if (inplace) {
            ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
        } else {
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        }
        std::vector<bool> d_r {t_r.to_vector<bool>()};
        ASSERT_EQ(d_a.size(), d_b.size());
        ASSERT_EQ(d_a.size(), d_r.size());
        ASSERT_EQ(t_a.dtype(), t_b.dtype());
        ASSERT_EQ(t_a.dtype(), t_r.dtype());
        for (std::int64_t i = 0; i < d_r.size(); ++i) {
            ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " op " << d_b[i] << " = " << d_r[i];
        }
    });
}

static void test_binary_float_compare(
    device_kind dev,
    bool broadcast,
    dtype ty,
    std::function<tensor (tensor, tensor)>&& a,
    std::function<mag_e8m23_t (mag_e8m23_t, mag_e8m23_t)>&& b,
    mag_e8m23_t min = -10.0,
    mag_e8m23_t max = 10.0
) {
    auto& ctx = get_cached_context(dev);
    ctx.stop_grad_recorder();
    for_all_test_shapes([&](std::span<const std::int64_t> shape){
        tensor t_a{ctx, ty, shape};
        t_a.fill_rand_uniform(min, max);
        tensor t_b{t_a.clone()};
        std::vector<mag_e8m23_t> d_a {t_a.to_vector<mag_e8m23_t>()};
        std::vector<mag_e8m23_t> d_b {t_b.to_vector<mag_e8m23_t>()};
        tensor t_r {std::invoke(a, t_a, t_b)};
        ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        ASSERT_EQ(t_r.dtype(), dtype::boolean);
        ASSERT_EQ(d_a.size(), d_b.size());
        ASSERT_EQ(d_a.size(), t_r.numel());
        std::vector<bool> d_r {t_r.to_vector<bool>()};
        for (std::int64_t i = 0; i < d_r.size(); ++i)
            ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " ? " << d_b[i] << " = " << d_r[i];
    });
}

static void test_binary_int_compare(
    device_kind dev,
    bool broadcast,
    dtype ty,
    std::function<tensor (tensor, tensor)>&& a,
    std::function<int32_t (int32_t, int32_t)>&& b,
    int32_t min = -10,
    int32_t max = 10
) {
    auto& ctx = get_cached_context(dev);
    ctx.stop_grad_recorder();
    std::uniform_int_distribution<int32_t> dist{min, max};
    for_all_test_shapes([&](std::span<const std::int64_t> shape){
        tensor t_a{ctx, ty, shape};  t_a.fill(dist(gen));
        tensor t_b{t_a.clone()};
        std::vector<int32_t> d_a{t_a.to_vector<int32_t>()};
        std::vector<int32_t> d_b{t_b.to_vector<int32_t>()};
        tensor t_r{std::invoke(a, t_a, t_b)};
        ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        ASSERT_EQ(t_r.dtype(), dtype::boolean);
        ASSERT_EQ(d_a.size(), d_b.size());
        ASSERT_EQ(d_a.size(), t_r.numel());
        std::vector<bool> d_r{t_r.to_vector<bool>()};
        for (std::int64_t i = 0; i < d_r.size(); ++i)
            ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " ? " << d_b[i] << " = " << d_r[i];
    });
}

static void test_binary_bool_compare(
    device_kind dev,
    bool broadcast,
    std::function<tensor (tensor, tensor)>&& a,
    std::function<bool (bool, bool)>&& b
) {
    auto& ctx = get_cached_context(dev);
    ctx.stop_grad_recorder();
    for_all_test_shapes([&](std::span<const std::int64_t> shape){
        tensor t_a{ctx, dtype::boolean, shape};  t_a.fill(true);
        tensor t_b{t_a.clone()};
        t_b.fill(false);
        std::vector<bool> d_a{t_a.to_vector<bool>()};
        std::vector<bool> d_b{t_b.to_vector<bool>()};
        tensor t_r{std::invoke(a, t_a, t_b)};
        ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        ASSERT_EQ(t_r.dtype(), dtype::boolean);
        ASSERT_EQ(d_a.size(), d_b.size());
        ASSERT_EQ(d_a.size(), t_r.numel());
        std::vector<bool> d_r{t_r.to_vector<bool>()};
        for (std::int64_t i = 0; i < d_r.size(); ++i)
            ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " ? " << d_b[i] << " = " << d_r[i];
    });
}

#define impl_binary_operator_float_test_group(name, op, data_type) \
    TEST_P(binary_operators, name##_same_shape_##data_type) { \
        test_binary_float_operator(GetParam(), false, false, dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](mag_e8m23_t a, mag_e8m23_t b) -> mag_e8m23_t { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_broadcast_##data_type) { \
        test_binary_float_operator(GetParam(), true, false, dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](mag_e8m23_t a, mag_e8m23_t b) -> mag_e8m23_t { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_inplace_same_shape_##data_type) { \
        test_binary_float_operator(GetParam(), false, true, dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](mag_e8m23_t a, mag_e8m23_t b) -> mag_e8m23_t { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_inplace_broadcast_##data_type) { \
        test_binary_float_operator(GetParam(), true, true, dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](mag_e8m23_t a, mag_e8m23_t b) -> mag_e8m23_t { return a op b; } \
        ); \
    }

#define impl_binary_operator_bool_test_group(name, op) \
    TEST_P(binary_operators, name##_same_shape_bool) { \
        test_binary_boolean_operator(GetParam(), false, false, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_broadcast_bool) { \
        test_binary_boolean_operator(GetParam(), true, false, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_inplace_same_shape_bool) { \
        test_binary_boolean_operator(GetParam(), false, true, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_inplace_broadcast_bool) { \
        test_binary_boolean_operator(GetParam(), true, true, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    }

#define impl_binary_operator_int_test_group(name, op, data_type, min, max) \
    TEST_P(binary_operators, name##_same_shape_##data_type) { \
        test_binary_int_operator(GetParam(), false, false, dtype::data_type, (min), (max), \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](int32_t a, int32_t b) -> int32_t { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_broadcast_##data_type) { \
        test_binary_int_operator(GetParam(), true, false, dtype::data_type, (min), (max), \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](int32_t a, int32_t b) -> int32_t { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_inplace_same_shape_##data_type) { \
        test_binary_int_operator(GetParam(), false, true, dtype::data_type, (min), (max), \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](int32_t a, int32_t b) -> int32_t { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_inplace_broadcast_##data_type) { \
        test_binary_int_operator(GetParam(), true, true, dtype::data_type, (min), (max), \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](int32_t a, int32_t b) -> int32_t { return a op b; } \
        ); \
    }

#define impl_binary_operator_cmp_float_test_group(name, op, data_type) \
    TEST_P(binary_operators, name##_same_shape_##data_type) { \
        test_binary_float_compare(GetParam(), false, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](mag_e8m23_t a, mag_e8m23_t b) -> bool { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_broadcast_##data_type) { \
        test_binary_float_compare(GetParam(), true, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](mag_e8m23_t a, mag_e8m23_t b) -> bool { return a op b; } \
        ); \
    }

#define impl_binary_operator_cmp_int_test_group(name, op, data_type) \
    TEST_P(binary_operators, name##_same_shape_##data_type) { \
        test_binary_int_compare(GetParam(), false, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](int32_t a, int32_t b) -> bool { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_broadcast_##data_type) { \
        test_binary_int_compare(GetParam(), true, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](int32_t a, int32_t b) -> bool { return a op b; } \
        ); \
    }

#define impl_binary_operator_cmp_bool_test_group(name, op) \
    TEST_P(binary_operators, name##_same_shape_bool) { \
        test_binary_bool_compare(GetParam(), false, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_broadcast_bool) { \
        test_binary_bool_compare(GetParam(), true, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    }

impl_binary_operator_float_test_group(add, +, e8m23)
impl_binary_operator_float_test_group(add, +, e5m10)

impl_binary_operator_float_test_group(sub, -, e8m23)
impl_binary_operator_float_test_group(sub, -, e5m10)

impl_binary_operator_float_test_group(mul, *, e8m23)
impl_binary_operator_float_test_group(mul, *, e5m10)

impl_binary_operator_float_test_group(div, /, e8m23)
impl_binary_operator_float_test_group(div, /, e5m10)

impl_binary_operator_bool_test_group(and, &)
impl_binary_operator_bool_test_group(or, |)
impl_binary_operator_bool_test_group(xor, ^)

impl_binary_operator_int_test_group(add, +, i32, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max())
impl_binary_operator_int_test_group(sub, -, i32, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max())
impl_binary_operator_int_test_group(mul, *, i32, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max())
impl_binary_operator_int_test_group(div, /, i32, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max())
impl_binary_operator_int_test_group(and, &, i32, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max())
impl_binary_operator_int_test_group(or, |, i32, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max())
impl_binary_operator_int_test_group(xor, ^, i32, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max())
impl_binary_operator_int_test_group(shl, <<, i32, 0, 32)
impl_binary_operator_int_test_group(shr, >>, i32, 0, 32)

impl_binary_operator_cmp_float_test_group(eq, ==, e8m23)
impl_binary_operator_cmp_float_test_group(eq, ==, e5m10)
impl_binary_operator_cmp_int_test_group(eq, ==, i32)
impl_binary_operator_cmp_bool_test_group(eq, ==)

impl_binary_operator_cmp_float_test_group(ne, !=, e8m23)
impl_binary_operator_cmp_float_test_group(ne, !=, e5m10)
impl_binary_operator_cmp_int_test_group(ne, !=, i32)
impl_binary_operator_cmp_bool_test_group(ne, !=)

impl_binary_operator_cmp_float_test_group(le, <=, e8m23)
impl_binary_operator_cmp_float_test_group(le, <=, e5m10)
impl_binary_operator_cmp_int_test_group(le, <=, i32)

impl_binary_operator_cmp_float_test_group(ge, >=, e8m23)
impl_binary_operator_cmp_float_test_group(ge, >=, e5m10)
impl_binary_operator_cmp_int_test_group(ge, >=, i32)

impl_binary_operator_cmp_float_test_group(lt, <, e8m23)
impl_binary_operator_cmp_float_test_group(lt, <, e5m10)
impl_binary_operator_cmp_int_test_group(lt, <, i32)

impl_binary_operator_cmp_float_test_group(gt, >, e8m23)
impl_binary_operator_cmp_float_test_group(gt, >, e5m10)
impl_binary_operator_cmp_int_test_group(gt, >, i32)

#undef impl_binary_operator_float_test_group
#undef impl_binary_operator_bool_test_group
#undef impl_binary_operator_int_test_group
#undef impl_binary_operator_cmp_float_test_group

INSTANTIATE_TEST_SUITE_P(
    binary_operators_multi_backend,
    binary_operators,
    ValuesIn(get_supported_test_backends()),
    get_gtest_backend_name
);
