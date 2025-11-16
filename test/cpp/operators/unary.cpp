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

static constexpr std::int64_t lim {3};
static constexpr std::int64_t broadcast_lim {lim-1};

class unary_operators : public TestWithParam<device_kind> {};

static auto test_unary_operator(
    std::int64_t lim,
    bool broadcast,
    bool inplace,
    bool subview,
    mag_e8m23_t eps,
    dtype ty,
    std::function<tensor (tensor)>&& a,
    std::function<mag_e8m23_t (mag_e8m23_t)>&& b,
    mag_e8m23_t min = 0.0,
    mag_e8m23_t max = 2.0
) -> void {
    auto ctx = context{};
    for_all_shape_perms(lim, broadcast ? 2 : 1, [&](std::span<const std::int64_t> shape) {
        tensor base{ctx, ty, shape};
        base.fill_rand_uniform(min, max);
        tensor t_a = subview ? make_random_view(base) : base;
        if (subview)
            ASSERT_TRUE(t_a.is_view());
        std::vector<mag_e8m23_t> d_a{t_a.to_vector<mag_e8m23_t>()};
        tensor t_r = std::invoke(a, t_a);
        if (inplace)
            ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
        else
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        if (inplace)
            ASSERT_EQ(t_a.storage_base_ptr(), t_r.storage_base_ptr());
        std::vector<mag_e8m23_t> d_r{t_r.to_vector<mag_e8m23_t>()};
        ASSERT_EQ(d_a.size(), d_r.size());
        for (std::size_t i = 0; i < d_r.size(); ++i)
            ASSERT_NEAR(std::invoke(b, d_a[i]), d_r[i], eps);
    });
}

#define impl_unary_operator_test_group(eps, name, data_type, lambda) \
    TEST_P(unary_operators, name##_same_shape_##data_type) { \
        test_unary_operator(lim, false, false, false, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](mag_e8m23_t a) -> mag_e8m23_t { return lambda(a); } \
        ); \
    } \
    TEST_P(unary_operators, name##_broadcast_##data_type) { \
        test_unary_operator(broadcast_lim, true, false, false, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](mag_e8m23_t a) -> mag_e8m23_t { return lambda(a); } \
        ); \
    } \
    TEST_P(unary_operators, name##_inplace_same_shape_##data_type) { \
        test_unary_operator(lim, false, true, false, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](mag_e8m23_t a) -> mag_e8m23_t { return lambda(a); } \
        ); \
    } \
    TEST_P(unary_operators, name##_inplace_broadcast_##data_type) { \
        test_unary_operator(broadcast_lim, true, true, false, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](mag_e8m23_t a) -> mag_e8m23_t { return lambda(a); } \
        ); \
    } \
    TEST_P(unary_operators, name##_view_same_shape_##data_type) { \
        test_unary_operator(lim, false, false, true, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](mag_e8m23_t a) -> mag_e8m23_t { return lambda(a); } \
        ); \
    } \
    TEST_P(unary_operators, name##_view_broadcast_##data_type) { \
        test_unary_operator(broadcast_lim, true, false, true, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](mag_e8m23_t a) -> mag_e8m23_t { return lambda(a); } \
        ); \
    } \
    TEST_P(unary_operators, name##_view_inplace_same_shape_##data_type) { \
        test_unary_operator(lim, false, true, true, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](mag_e8m23_t a) -> mag_e8m23_t { return lambda(a); } \
        ); \
    } \
    TEST_P(unary_operators, name##_view_inplace_broadcast_##data_type) { \
        test_unary_operator(broadcast_lim, true, true, true, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](mag_e8m23_t a) -> mag_e8m23_t { return lambda(a); } \
        ); \
    }

impl_unary_operator_test_group(0.f, abs, e8m23, [](auto x) { return std::abs(x); })
impl_unary_operator_test_group(0.f, abs, e5m10, [](auto x) { return std::abs(x); })
impl_unary_operator_test_group(0.f, sgn, e8m23, [](auto x) { return std::copysign(1.0f, x); })
impl_unary_operator_test_group(0.f, sgn, e5m10, [](auto x) { return std::copysign(1.0f, x); })
impl_unary_operator_test_group(0.f, neg, e8m23, [](auto x) { return -x; })
impl_unary_operator_test_group(0.f, neg, e5m10, [](auto x) { return -x; })
impl_unary_operator_test_group(0.f, log, e8m23, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(0.f, log, e5m10, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(0.f, sqr, e8m23, [](auto x) { return x*x; })
impl_unary_operator_test_group(0.f, sqr, e5m10, [](auto x) { return x*x; })
impl_unary_operator_test_group(0.f, sqrt, e8m23, [](auto x) { return std::sqrt(x); })
impl_unary_operator_test_group(0.f, sqrt, e5m10, [](auto x) { return std::sqrt(x); })
impl_unary_operator_test_group(0.f, sin, e8m23, [](auto x) { return std::sin(x); })
impl_unary_operator_test_group(0.f, sin, e5m10, [](auto x) { return std::sin(x); })
impl_unary_operator_test_group(0.f, cos, e8m23, [](auto x) { return std::cos(x); })
impl_unary_operator_test_group(0.f, cos, e5m10, [](auto x) { return std::cos(x); })
impl_unary_operator_test_group(0.f, step, e8m23, [](auto x) { return x > 0.0f ? 1.0f : 0.0f; })
impl_unary_operator_test_group(0.f, step, e5m10, [](auto x) { return x > 0.0f ? 1.0f : 0.0f; })
impl_unary_operator_test_group(0.f, exp, e8m23, [](auto x) { return std::exp(x); })
impl_unary_operator_test_group(0.f, exp, e5m10, [](auto x) { return std::exp(x); })
impl_unary_operator_test_group(0.f, floor, e8m23, [](auto x) { return std::floor(x); })
impl_unary_operator_test_group(0.f, floor, e5m10, [](auto x) { return std::floor(x); })
impl_unary_operator_test_group(0.f, ceil, e8m23, [](auto x) { return std::ceil(x); })
impl_unary_operator_test_group(0.f, ceil, e5m10, [](auto x) { return std::ceil(x); })
impl_unary_operator_test_group(0.f, round, e8m23, [](auto x) { return std::rint(x); })
impl_unary_operator_test_group(0.f, round, e5m10, [](auto x) { return std::rint(x); })
impl_unary_operator_test_group(0.f, sigmoid, e8m23, [](auto x) { return 1.0f / (1.0f + std::exp(-(x))); })
impl_unary_operator_test_group(0.f, sigmoid, e5m10, [](auto x) { return 1.0f / (1.0f + std::exp(-(x))); })
impl_unary_operator_test_group(0.f, hard_sigmoid, e8m23, [](auto x) { return std::min(1.0f, std::max(0.0f, (x + 3.0f)/6.0f)); })
impl_unary_operator_test_group(0.f, hard_sigmoid, e5m10, [](auto x) { return std::min(1.0f, std::max(0.0f, (x + 3.0f)/6.0f)); })
impl_unary_operator_test_group(0.f, silu, e8m23, [](auto x) { return x * (1.0f / (1.0f + std::exp(-(x)))); })
impl_unary_operator_test_group(0.f, silu, e5m10, [](auto x) { return x * (1.0f / (1.0f + std::exp(-(x)))); })
impl_unary_operator_test_group(0.f, tanh, e8m23, [](auto x) { return std::tanh(x); })
impl_unary_operator_test_group(0.f, tanh, e5m10, [](auto x) { return std::tanh(x); })
impl_unary_operator_test_group(0.f, relu, e8m23, [](auto x) { return std::max(0.0f, x); })
impl_unary_operator_test_group(0.f, relu, e5m10, [](auto x) { return std::max(0.0f, x); })
impl_unary_operator_test_group(0.f, gelu, e8m23, [](auto x) { return .5f*x*(1.f + std::erf(x*(1.0f / std::sqrt(2.0f)))); })
impl_unary_operator_test_group(0.f, gelu, e5m10, [](auto x) { return .5f*x*(1.f + std::erf(x*(1.0f / std::sqrt(2.0f)))); })
impl_unary_operator_test_group(1e-3f, gelu_approx, e8m23, [](auto x) { return .5f*x*(1.f+std::tanh((1.f/std::sqrt(2.f))*(x+MAG_GELU_COEFF*std::pow(x, 3.f)))); })
impl_unary_operator_test_group(0.f, gelu_approx, e5m10, [](auto x) { return .5f*x*(1.f+std::tanh((1.f/std::sqrt(2.f))*(x+MAG_GELU_COEFF*std::pow(x, 3.f)))); })

INSTANTIATE_TEST_SUITE_P(
    unary_operators_multi_backend,
    unary_operators,
    ValuesIn(get_supported_test_backends()),
    get_gtest_backend_name
);
