// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

static constexpr std::int64_t lim {4};
static constexpr std::int64_t broadcast_lim {lim-1};

#define impl_unary_operator_test_group(eps, name, data_type, lambda) \
    TEST(cpu_tensor_unary_ops, name##_same_shape_##data_type) { \
        test::test_unary_operator<false, false, false>(lim, eps != 0.f ? eps : test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_broadcast_##data_type) { \
        test::test_unary_operator<true, false, false>(broadcast_lim, eps != 0.f ? eps : test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_inplace_same_shape_##data_type) { \
        test::test_unary_operator<false, true, false>(lim, eps != 0.f ? eps : test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_inplace_broadcast_##data_type) { \
        test::test_unary_operator<true, true, false>(broadcast_lim, eps != 0.f ? eps : test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_view_same_shape_##data_type) { \
        test::test_unary_operator<false, false, true>(lim, eps != 0.f ? eps : test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_view_broadcast_##data_type) { \
        test::test_unary_operator<true, false, true>(broadcast_lim, eps != 0.f ? eps : test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_view_inplace_same_shape_##data_type) { \
        test::test_unary_operator<false, true, true>(lim, eps != 0.f ? eps : test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_view_inplace_broadcast_##data_type) { \
        test::test_unary_operator<true, true, true>(broadcast_lim, eps != 0.f ? eps : test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
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

