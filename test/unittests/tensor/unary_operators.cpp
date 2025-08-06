// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

static constexpr std::int64_t lim {4};
static constexpr std::int64_t broadcast_lim {lim-1};

#define impl_unary_operator_test_group(name, data_type, lambda) \
    TEST(cpu_tensor_unary_ops, name##_same_shape_##data_type) { \
        test::test_unary_operator<false, false, false>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_broadcast_##data_type) { \
        test::test_unary_operator<true, false, false>(broadcast_lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_inplace_same_shape_##data_type) { \
        test::test_unary_operator<false, true, false>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_inplace_broadcast_##data_type) { \
        test::test_unary_operator<true, true, false>(broadcast_lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_view_same_shape_##data_type) { \
        test::test_unary_operator<false, false, true>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_view_broadcast_##data_type) { \
        test::test_unary_operator<true, false, true>(broadcast_lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_view_inplace_same_shape_##data_type) { \
        test::test_unary_operator<false, true, true>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    } \
    TEST(cpu_tensor_unary_ops, name##_view_inplace_broadcast_##data_type) { \
        test::test_unary_operator<true, true, true>(broadcast_lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a) -> tensor { return a.name##_(); }, \
            [](float a) -> float { return lambda(a); } \
        ); \
    }

impl_unary_operator_test_group(abs, e8m23, [](auto x) { return std::abs(x); })
impl_unary_operator_test_group(abs, e5m10, [](auto x) { return std::abs(x); })
impl_unary_operator_test_group(sgn, e8m23, [](auto x) { return std::copysign(1.0f, x); })
impl_unary_operator_test_group(sgn, e5m10, [](auto x) { return std::copysign(1.0f, x); })
impl_unary_operator_test_group(neg, e8m23, [](auto x) { return -x; })
impl_unary_operator_test_group(neg, e5m10, [](auto x) { return -x; })
impl_unary_operator_test_group(log, e8m23, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(log, e5m10, [](auto x) { return std::log(x); })
impl_unary_operator_test_group(sqr, e8m23, [](auto x) { return x*x; })
impl_unary_operator_test_group(sqr, e5m10, [](auto x) { return x*x; })
impl_unary_operator_test_group(sqrt, e8m23, [](auto x) { return std::sqrt(x); })
impl_unary_operator_test_group(sqrt, e5m10, [](auto x) { return std::sqrt(x); })
impl_unary_operator_test_group(sin, e8m23, [](auto x) { return std::sin(x); })
impl_unary_operator_test_group(sin, e5m10, [](auto x) { return std::sin(x); })
impl_unary_operator_test_group(cos, e8m23, [](auto x) { return std::cos(x); })
impl_unary_operator_test_group(cos, e5m10, [](auto x) { return std::cos(x); })
impl_unary_operator_test_group(step, e8m23, [](auto x) { return x > 0.0f ? 1.0f : 0.0f; })
impl_unary_operator_test_group(step, e5m10, [](auto x) { return x > 0.0f ? 1.0f : 0.0f; })
impl_unary_operator_test_group(exp, e8m23, [](auto x) { return std::exp(x); })
impl_unary_operator_test_group(exp, e5m10, [](auto x) { return std::exp(x); })
impl_unary_operator_test_group(floor, e8m23, [](auto x) { return std::floor(x); })
impl_unary_operator_test_group(floor, e5m10, [](auto x) { return std::floor(x); })
impl_unary_operator_test_group(ceil, e8m23, [](auto x) { return std::ceil(x); })
impl_unary_operator_test_group(ceil, e5m10, [](auto x) { return std::ceil(x); })
impl_unary_operator_test_group(round, e8m23, [](auto x) { return std::rint(x); })
impl_unary_operator_test_group(round, e5m10, [](auto x) { return std::rint(x); })
impl_unary_operator_test_group(sigmoid, e8m23, [](auto x) { return 1.0f / (1.0f + std::exp(-(x))); })
impl_unary_operator_test_group(sigmoid, e5m10, [](auto x) { return 1.0f / (1.0f + std::exp(-(x))); })
impl_unary_operator_test_group(hard_sigmoid, e8m23, [](auto x) { return std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f)); })
impl_unary_operator_test_group(hard_sigmoid, e5m10, [](auto x) { return std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f)); })  // TODO
impl_unary_operator_test_group(silu, e8m23, [](auto x) { return x * (1.0f / (1.0f + std::exp(-(x)))); })
impl_unary_operator_test_group(silu, e5m10, [](auto x) { return x * (1.0f / (1.0f + std::exp(-(x)))); })
impl_unary_operator_test_group(tanh, e8m23, [](auto x) { return std::tanh(x); })
impl_unary_operator_test_group(tanh, e5m10, [](auto x) { return std::tanh(x); })
impl_unary_operator_test_group(relu, e8m23, [](auto x) { return std::max(0.0f, x); })
impl_unary_operator_test_group(relu, e5m10, [](auto x) { return std::max(0.0f, x); })
impl_unary_operator_test_group(gelu, e8m23, [](auto x) { return .5f * x * (1.f + std::erf(x * (1.0f / std::sqrt(2.0f)))); })
impl_unary_operator_test_group(gelu, e5m10, [](auto x) { return .5f * x * (1.f + std::erf(x * (1.0f / std::sqrt(2.0f)))); })
impl_unary_operator_test_group(gelu_approx, e8m23, [](auto x) { return .5f * x * (1.f + std::erf(x * (1.0f / std::sqrt(2.0f)))); })
impl_unary_operator_test_group(gelu_approx, e5m10, [](auto x) { return .5f * x * (1.f + std::erf(x * (1.0f / std::sqrt(2.0f)))); })

