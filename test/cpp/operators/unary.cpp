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

class unary_operators : public TestWithParam<device_kind> {};

template <typename T>
static auto test_unary_operator(
    device_kind dev,
    bool inplace,
    bool subview,
    float eps,
    dtype ty,
    std::function<tensor (tensor)>&& a,
    std::function<T (T)>&& b,
    T min = static_cast<T>(0.0),
    T max = static_cast<T>(2.0)
) -> void {
    auto& ctx = get_cached_context(dev);
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        tensor base{ctx, ty, shape};
        base.uniform_(static_cast<float>(min), static_cast<float>(max));
        tensor t_a = subview ? make_random_view(base) : base;
        if (subview)
            ASSERT_TRUE(t_a.is_view());
        std::vector<T> d_a {t_a.to_vector<T>()};
        tensor t_r = std::invoke(a, t_a);
        if (inplace)
            ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
        else
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        if (inplace)
            ASSERT_EQ(t_a.storage_base_ptr(), t_r.storage_base_ptr());
        std::vector<T> d_r {t_r.to_vector<T>()};
        ASSERT_EQ(d_a.size(), d_r.size());
        for (size_t i=0; i < d_r.size(); ++i) {
            auto x = d_a[i];
            auto y = d_r[i];
            if (std::isnan(x) || std::isnan(y)) [[unlikely]] continue;
            ASSERT_NEAR(std::invoke(b, x), y, eps);
        }
    });
}

#define impl_unary_operator_test_group(eps, name, data_type, T, lambda) \
    TEST_P(unary_operators, name##_same_shape_##data_type) { \
        test_unary_operator<T>(GetParam(), false, false, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](T a) -> T { return static_cast<T>(lambda(a)); } \
        ); \
    } \
    TEST_P(unary_operators, name##_inplace_same_shape_##data_type) { \
        test_unary_operator<T>(GetParam(), true, false, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](T a) -> T { return static_cast<T>(lambda(a)); } \
        ); \
    } \
    TEST_P(unary_operators, name##_view_same_shape_##data_type) { \
        test_unary_operator<T>(GetParam(), false, true, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](T a) -> T { return static_cast<T>(lambda(a)); } \
        ); \
    } \
    TEST_P(unary_operators, name##_view_inplace_same_shape_##data_type) { \
        test_unary_operator<T>(GetParam(), true, true, eps != 0.0f ? eps : dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](T a) -> T { return static_cast<T>(lambda(a)); } \
        ); \
    } \

impl_unary_operator_test_group(0.f, abs, float32, float, [](auto x) { return std::abs(x); })
impl_unary_operator_test_group(0.f, abs, float16, float16, [](auto x) { return std::abs(x); })
impl_unary_operator_test_group(0.f, sgn, float32, float, [](auto x) { return std::copysign(1.0f, x); })
impl_unary_operator_test_group(0.f, sgn, float16, float16, [](auto x) { return std::copysign(1.0f, x); })
impl_unary_operator_test_group(0.f, neg, float32, float, [](auto x) { return -x; })
impl_unary_operator_test_group(0.f, neg, float16, float16, [](auto x) { return -x; })
impl_unary_operator_test_group(0.f, log, float32, float, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(0.f, log, float16, float16, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(0.f, log10, float32, float, [](auto x) { return std::log10(x); })
impl_unary_operator_test_group(0.f, log10, float16, float16, [](auto x) { return std::log10(x); })
impl_unary_operator_test_group(0.f, log1p, float32, float, [](auto x) { return std::log1p(x); })
impl_unary_operator_test_group(0.f, log1p, float16, float16, [](auto x) { return std::log1p(x); })
impl_unary_operator_test_group(0.f, log2, float32, float, [](auto x) { return std::log2(x); })
impl_unary_operator_test_group(0.f, log2, float16, float16, [](auto x) { return std::log2(x); })
impl_unary_operator_test_group(0.f, sqr, float32, float, [](auto x) { return x*x; })
impl_unary_operator_test_group(0.f, sqr, float16, float16, [](auto x) { return x*x; })
impl_unary_operator_test_group(0.f, rcp, float32, float, [](auto x) { return 1.f/x; })
impl_unary_operator_test_group(0.f, rcp, float16, float16, [](auto x) { return 1.f/x; })
impl_unary_operator_test_group(0.f, sqrt, float32, float, [](auto x) { return std::sqrt(x); })
impl_unary_operator_test_group(0.f, sqrt, float16, float16, [](auto x) { return std::sqrt(x); })
impl_unary_operator_test_group(0.f, rsqrt, float32, float, [](auto x) { return 1.f/std::sqrt(x); })
impl_unary_operator_test_group(0.f, rsqrt, float16, float16, [](auto x) { return 1.f/std::sqrt(x); })
impl_unary_operator_test_group(0.f, sin, float32, float, [](auto x) { return std::sin(x); })
impl_unary_operator_test_group(0.f, sin, float16, float16, [](auto x) { return std::sin(x); })
impl_unary_operator_test_group(0.f, cos, float32, float, [](auto x) { return std::cos(x); })
impl_unary_operator_test_group(0.f, cos, float16, float16, [](auto x) { return std::cos(x); })
impl_unary_operator_test_group(0.f, tan, float32, float, [](auto x) { return std::tan(x); })
impl_unary_operator_test_group(0.f, tan, float16, float16, [](auto x) { return std::tan(x); })
impl_unary_operator_test_group(0.f, sinh, float32, float, [](auto x) { return std::sinh(x); })
impl_unary_operator_test_group(0.f, sinh, float16, float16, [](auto x) { return std::sinh(x); })
impl_unary_operator_test_group(0.f, cosh, float32, float, [](auto x) { return std::cosh(x); })
impl_unary_operator_test_group(0.f, cosh, float16, float16, [](auto x) { return std::cosh(x); })
impl_unary_operator_test_group(0.f, tanh, float32, float, [](auto x) { return std::tanh(x); })
impl_unary_operator_test_group(0.f, tanh, float16, float16, [](auto x) { return std::tanh(x); })
impl_unary_operator_test_group(0.f, asin, float32, float, [](auto x) { return std::asin(x); })
impl_unary_operator_test_group(0.f, asin, float16, float16, [](auto x) { return std::asin(x); })
impl_unary_operator_test_group(0.f, acos, float32, float, [](auto x) { return std::acos(x); })
impl_unary_operator_test_group(0.f, acos, float16, float16, [](auto x) { return std::acos(x); })
impl_unary_operator_test_group(0.f, atan, float32, float, [](auto x) { return std::atan(x); })
impl_unary_operator_test_group(0.f, atan, float16, float16, [](auto x) { return std::atan(x); })
impl_unary_operator_test_group(0.f, asinh, float32, float, [](auto x) { return std::asinh(x); })
impl_unary_operator_test_group(0.f, asinh, float16, float16, [](auto x) { return std::asinh(x); })
impl_unary_operator_test_group(0.f, acosh, float32, float, [](auto x) { return std::acosh(x); })
impl_unary_operator_test_group(0.f, acosh, float16, float16, [](auto x) { return std::acosh(x); })
impl_unary_operator_test_group(0.f, atanh, float32, float, [](auto x) { return std::atanh(x); })
impl_unary_operator_test_group(0.f, atanh, float16, float16, [](auto x) { return std::atanh(x); })
impl_unary_operator_test_group(0.f, step, float32, float, [](auto x) { return x > 0.0f ? 1.0f : 0.0f; })
impl_unary_operator_test_group(0.f, step, float16, float16, [](auto x) { return x > 0.0f ? 1.0f : 0.0f; })
impl_unary_operator_test_group(0.f, erf, float32, float, [](auto x) { return std::erf(x); })
impl_unary_operator_test_group(0.f, erf, float16, float16, [](auto x) { return std::erf(x); })
impl_unary_operator_test_group(0.f, erfc, float32, float, [](auto x) { return std::erfc(x); })
impl_unary_operator_test_group(0.f, erfc, float16, float16, [](auto x) { return std::erfc(x); })
impl_unary_operator_test_group(0.f, exp, float32, float, [](auto x) { return std::exp(x); })
impl_unary_operator_test_group(0.f, exp, float16, float16, [](auto x) { return std::exp(x); })
impl_unary_operator_test_group(0.f, exp2, float32, float, [](auto x) { return std::exp2(x); })
impl_unary_operator_test_group(0.f, exp2, float16, float16, [](auto x) { return std::exp2(x); })
impl_unary_operator_test_group(0.f, expm1, float32, float, [](auto x) { return std::expm1(x); })
impl_unary_operator_test_group(0.f, expm1, float16, float16, [](auto x) { return std::expm1(x); })
impl_unary_operator_test_group(0.f, floor, float32, float, [](auto x) { return std::floor(x); })
impl_unary_operator_test_group(0.f, floor, float16, float16, [](auto x) { return std::floor(x); })
impl_unary_operator_test_group(0.f, ceil, float32, float, [](auto x) { return std::ceil(x); })
impl_unary_operator_test_group(0.f, ceil, float16, float16, [](auto x) { return std::ceil(x); })
impl_unary_operator_test_group(0.f, round, float32, float, [](auto x) { return std::nearbyint(x); })
impl_unary_operator_test_group(0.f, round, float16, float16, [](auto x) { return std::nearbyint(x); })
impl_unary_operator_test_group(0.f, trunc, float32, float, [](auto x) { return std::trunc(x); })
impl_unary_operator_test_group(0.f, trunc, float16, float16, [](auto x) { return std::trunc(x); })
impl_unary_operator_test_group(0.f, sigmoid, float32, float, [](auto x) { return 1.0f / (1.0f + std::exp(-(x))); })
impl_unary_operator_test_group(0.f, sigmoid, float16, float16, [](auto x) { return 1.0f / (1.0f + std::exp(-(x))); })
impl_unary_operator_test_group(0.f, hard_sigmoid, float32, float, [](auto x) { return std::min(1.0f, std::max(0.0f, (x + 3.0f)/6.0f)); })
impl_unary_operator_test_group(0.f, hard_sigmoid, float16, float16, [](auto x) { return std::min(1.0f, std::max(0.0f, (x + 3.0f)/6.0f)); })
impl_unary_operator_test_group(0.f, silu, float32, float, [](auto x) { return x * (1.0f / (1.0f + std::exp(-(x)))); })
impl_unary_operator_test_group(0.f, silu, float16, float16, [](auto x) { return x * (1.0f / (1.0f + std::exp(-(x)))); })
impl_unary_operator_test_group(0.f, relu, float32, float, [](auto x) { return std::max(0.0f, x); })
impl_unary_operator_test_group(0.f, relu, float16, float16, [](auto x) { return std::max(0.0f, static_cast<float>(x)); })
impl_unary_operator_test_group(0.f, gelu, float32, float, [](auto x) { return .5f*x*(1.f + std::erf(x*(1.0f / std::sqrt(2.0f)))); })
impl_unary_operator_test_group(0.f, gelu, float16, float16, [](auto x) { return .5f*x*(1.f + std::erf(x*(1.0f / std::sqrt(2.0f)))); })
impl_unary_operator_test_group(1e-3f, gelu_approx, float32, float, [](auto x) { return .5f*x*(1.f+std::tanh((1.f/std::sqrt(2.f))*(x+MAG_GELU_COEFF*std::pow(x, 3.f)))); })
impl_unary_operator_test_group(0.f, gelu_approx, float16, float16, [](auto x) { return .5f*x*(1.f+std::tanh((1.f/std::sqrt(2.f))*(x+MAG_GELU_COEFF*std::pow(x, 3.f)))); })

INSTANTIATE_TEST_SUITE_P(
    unary_operators_multi_backend,
    unary_operators,
    ValuesIn(get_supported_test_backends()),
    get_gtest_backend_name
);
