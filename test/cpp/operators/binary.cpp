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

template <typename T>
[[nodiscard]] static constexpr std::pair<T, T> get_sample_interval() noexcept {
    if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, half_float::half>)
        return {static_cast<T>(-10.0f), static_cast<T>(10.0f)};
    else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>)
        return {std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};
    else return {0, 1};
}

template <typename T>
static void test_binary_operator(
    device_kind dev,
    bool inplace,
    dtype ty,
    std::function<tensor (tensor, tensor)>&& a,
    std::function<T (T, T)>&& b,
    std::pair<T, T> random_interval = get_sample_interval<T>()
) {
    auto& ctx = get_cached_context(dev);
    ctx.stop_grad_recorder();
    for_all_test_shapes([&](std::span<const int64_t> shape) {
        tensor t_a {ctx, ty, shape};
        if constexpr (std::is_same_v<T, bool>)
            t_a.fill(std::bernoulli_distribution{}(gen));
        else if constexpr (std::is_integral_v<T>)
             t_a.fill(std::uniform_int_distribution<T>{random_interval.first, random_interval.second}(gen));
        else
             t_a.fill(std::uniform_real_distribution<float>{random_interval.first, random_interval.second}(gen));
        tensor t_b {t_a.clone()};
        const data_accessor<T> d_a {t_a};
        const data_accessor<T> d_b {t_b};
        tensor t_r {std::invoke(a, t_a, t_b)};
        if (inplace) {
            ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
        } else {
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        }
        const data_accessor<T> d_r {t_r};
        ASSERT_EQ(d_a.size(), d_b.size());
        ASSERT_EQ(d_a.size(), d_r.size());
        ASSERT_EQ(t_a.dtype(), t_b.dtype());
        ASSERT_EQ(t_a.dtype(), t_r.dtype());
        for (int64_t i=0; i < d_r.size(); ++i) {
            ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " op " << d_b[i] << " = " << d_r[i];
        }
    });
}

template <typename T>
static void test_binary_cmp(
    device_kind dev,
    dtype ty,
    std::function<tensor (tensor, tensor)>&& a,
    std::function<bool (T, T)>&& b,
    std::pair<T, T> random_interval = get_sample_interval<T>()
) {
    auto& ctx = get_cached_context(dev);
    ctx.stop_grad_recorder();
    for_all_test_shapes([&](std::span<const int64_t> shape){
        tensor t_a{ctx, ty, shape};
        if constexpr (std::is_same_v<T, bool>)
            t_a.fill(std::bernoulli_distribution{}(gen));
        else if constexpr (std::is_integral_v<T>)
            t_a.fill(std::uniform_int_distribution<T>{random_interval.first, random_interval.second}(gen));
        else
           t_a.fill(std::uniform_real_distribution<float>{random_interval.first, random_interval.second}(gen));
        tensor t_b{t_a.clone()};
        const data_accessor<T> d_a {t_a};
        const data_accessor<T> d_b {t_b};
        tensor t_r{std::invoke(a, t_a, t_b)};
        ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        ASSERT_EQ(t_r.dtype(), dtype::boolean);
        ASSERT_EQ(d_a.size(), d_b.size());
        ASSERT_EQ(d_a.size(), t_r.numel());
        const data_accessor<T> d_r {t_r};
        for (int64_t i = 0; i < d_r.size(); ++i)
            ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " ? " << d_b[i] << " = " << d_r[i];
    });
}

#define impl_binary_operator_test_group(name, op, data_type, T) \
    TEST_P(binary_operators, name##_same_shape_##data_type) { \
        test_binary_operator<T>(GetParam(), false, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](T a, T b) -> T { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_inplace_same_shape_##data_type) { \
        test_binary_operator<T>(GetParam(), true, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](T a, T b) -> T { return a op b; } \
        ); \
    }

#define impl_binary_operator_cmp_test_group(name, op, data_type, T) \
    TEST_P(binary_operators, name##_same_shape_##data_type) { \
        test_binary_cmp<T>(GetParam(), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](T a, T b) -> bool { return a op b; } \
        ); \
    } \
    TEST_P(binary_operators, name##_broadcast_##data_type) { \
        test_binary_cmp<T>(GetParam(), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](T a, T b) -> bool { return a op b; } \
        ); \
    }

impl_binary_operator_test_group(add, +, e8m23, float)
impl_binary_operator_test_group(add, +, e5m10, float16)
impl_binary_operator_test_group(add, +, u8, uint8_t)
impl_binary_operator_test_group(add, +, i8, int8_t)
impl_binary_operator_test_group(add, +, u16, uint16_t)
impl_binary_operator_test_group(add, +, i16, int16_t)
impl_binary_operator_test_group(add, +, u32, uint32_t)
impl_binary_operator_test_group(add, +, i32, int32_t)
impl_binary_operator_test_group(add, +, u64, uint64_t)
impl_binary_operator_test_group(add, +, i64, int64_t)


impl_binary_operator_test_group(sub, -, e8m23, float)
impl_binary_operator_test_group(sub, -, e5m10, float16)
impl_binary_operator_test_group(sub, -, u8, uint8_t)
impl_binary_operator_test_group(sub, -, i8, int8_t)
impl_binary_operator_test_group(sub, -, u16, uint16_t)
impl_binary_operator_test_group(sub, -, i16, int16_t)
impl_binary_operator_test_group(sub, -, u32, uint32_t)
impl_binary_operator_test_group(sub, -, i32, int32_t)
impl_binary_operator_test_group(sub, -, u64, uint64_t)
impl_binary_operator_test_group(sub, -, i64, int64_t)

impl_binary_operator_test_group(mul, *, e8m23, float)
impl_binary_operator_test_group(mul, *, e5m10, float16)
impl_binary_operator_test_group(mul, *, u8, uint8_t)
impl_binary_operator_test_group(mul, *, i8, int8_t)
impl_binary_operator_test_group(mul, *, u16, uint16_t)
impl_binary_operator_test_group(mul, *, i16, int16_t)
impl_binary_operator_test_group(mul, *, u32, uint32_t)
impl_binary_operator_test_group(mul, *, i32, int32_t)
impl_binary_operator_test_group(mul, *, u64, uint64_t)
impl_binary_operator_test_group(mul, *, i64, int64_t)

impl_binary_operator_test_group(div, /, e8m23, float)
impl_binary_operator_test_group(div, /, e5m10, float16)
impl_binary_operator_test_group(div, /, u8, uint8_t)
impl_binary_operator_test_group(div, /, i8, int8_t)
impl_binary_operator_test_group(div, /, u16, uint16_t)
impl_binary_operator_test_group(div, /, i16, int16_t)
impl_binary_operator_test_group(div, /, u32, uint32_t)
impl_binary_operator_test_group(div, /, i32, int32_t)
impl_binary_operator_test_group(div, /, u64, uint64_t)
impl_binary_operator_test_group(div, /, i64, int64_t)

impl_binary_operator_test_group(and, &, boolean, bool)
impl_binary_operator_test_group(and, &, u8, uint8_t)
impl_binary_operator_test_group(and, &, i8, int8_t)
impl_binary_operator_test_group(and, &, u16, uint16_t)
impl_binary_operator_test_group(and, &, i16, int16_t)
impl_binary_operator_test_group(and, &, u32, uint32_t)
impl_binary_operator_test_group(and, &, i32, int32_t)
impl_binary_operator_test_group(and, &, u64, uint64_t)
impl_binary_operator_test_group(or, |, boolean, bool)
impl_binary_operator_test_group(or, |, u8, uint8_t)
impl_binary_operator_test_group(or, |, i8, int8_t)
impl_binary_operator_test_group(or, |, u16, uint16_t)
impl_binary_operator_test_group(or, |, i16, int16_t)
impl_binary_operator_test_group(or, |, u32, uint32_t)
impl_binary_operator_test_group(or, |, i32, int32_t)
impl_binary_operator_test_group(or, |, u64, uint64_t)
impl_binary_operator_test_group(xor, ^, boolean, bool)
impl_binary_operator_test_group(xor, ^, u8, uint8_t)
impl_binary_operator_test_group(xor, ^, i8, int8_t)
impl_binary_operator_test_group(xor, ^, u16, uint16_t)
impl_binary_operator_test_group(xor, ^, i16, int16_t)
impl_binary_operator_test_group(xor, ^, u32, uint32_t)
impl_binary_operator_test_group(xor, ^, i32, int32_t)
impl_binary_operator_test_group(xor, ^, u64, uint64_t)

impl_binary_operator_test_group(shl, <<, u8, uint8_t)
impl_binary_operator_test_group(shl, <<, i8, int8_t)
impl_binary_operator_test_group(shl, <<, u16, uint16_t)
impl_binary_operator_test_group(shl, <<, i16, int16_t)
impl_binary_operator_test_group(shl, <<, u32, uint32_t)
impl_binary_operator_test_group(shl, <<, i32, int32_t)
impl_binary_operator_test_group(shl, <<, u64, uint64_t)

impl_binary_operator_test_group(shr, >>, u8, uint8_t)
impl_binary_operator_test_group(shr, >>, i8, int8_t)
impl_binary_operator_test_group(shr, >>, u16, uint16_t)
impl_binary_operator_test_group(shr, >>, i16, int16_t)
impl_binary_operator_test_group(shr, >>, u32, uint32_t)
impl_binary_operator_test_group(shr, >>, i32, int32_t)
impl_binary_operator_test_group(shr, >>, u64, uint64_t)

impl_binary_operator_cmp_test_group(eq, ==, e8m23, float)
impl_binary_operator_cmp_test_group(eq, ==, e5m10, float16)
impl_binary_operator_cmp_test_group(eq, ==, boolean, bool)
impl_binary_operator_cmp_test_group(eq, ==, u8, uint8_t)
impl_binary_operator_cmp_test_group(eq, ==, i8, int8_t)
impl_binary_operator_cmp_test_group(eq, ==, u16, uint16_t)
impl_binary_operator_cmp_test_group(eq, ==, i16, int16_t)
impl_binary_operator_cmp_test_group(eq, ==, u32, uint32_t)
impl_binary_operator_cmp_test_group(eq, ==, i32, int32_t)
impl_binary_operator_cmp_test_group(eq, ==, u64, uint64_t)

impl_binary_operator_cmp_test_group(ne, !=, e8m23, float)
impl_binary_operator_cmp_test_group(ne, !=, e5m10, float16)
impl_binary_operator_cmp_test_group(ne, !=, boolean, bool)
impl_binary_operator_cmp_test_group(ne, !=, u8, uint8_t)
impl_binary_operator_cmp_test_group(ne, !=, i8, int8_t)
impl_binary_operator_cmp_test_group(ne, !=, u16, uint16_t)
impl_binary_operator_cmp_test_group(ne, !=, i16, int16_t)
impl_binary_operator_cmp_test_group(ne, !=, u32, uint32_t)
impl_binary_operator_cmp_test_group(ne, !=, i32, int32_t)
impl_binary_operator_cmp_test_group(ne, !=, u64, uint64_t)

impl_binary_operator_cmp_test_group(lt, <, e8m23, float)
impl_binary_operator_cmp_test_group(lt, <, e5m10, float16)
impl_binary_operator_cmp_test_group(lt, <, u8, uint8_t)
impl_binary_operator_cmp_test_group(lt, <, i8, int8_t)
impl_binary_operator_cmp_test_group(lt, <, u16, uint16_t)
impl_binary_operator_cmp_test_group(lt, <, i16, int16_t)
impl_binary_operator_cmp_test_group(lt, <, u32, uint32_t)
impl_binary_operator_cmp_test_group(lt, <, i32, int32_t)
impl_binary_operator_cmp_test_group(lt, <, u64, uint64_t)

impl_binary_operator_cmp_test_group(gt, >, e8m23, float)
impl_binary_operator_cmp_test_group(gt, >, e5m10, float16)
impl_binary_operator_cmp_test_group(gt, >, u8, uint8_t)
impl_binary_operator_cmp_test_group(gt, >, i8, int8_t)
impl_binary_operator_cmp_test_group(gt, >, u16, uint16_t)
impl_binary_operator_cmp_test_group(gt, >, i16, int16_t)
impl_binary_operator_cmp_test_group(gt, >, u32, uint32_t)
impl_binary_operator_cmp_test_group(gt, >, i32, int32_t)
impl_binary_operator_cmp_test_group(gt, >, u64, uint64_t)

impl_binary_operator_cmp_test_group(le, <=, e8m23, float)
impl_binary_operator_cmp_test_group(le, <=, e5m10, float16)
impl_binary_operator_cmp_test_group(le, <=, u8, uint8_t)
impl_binary_operator_cmp_test_group(le, <=, i8, int8_t)
impl_binary_operator_cmp_test_group(le, <=, u16, uint16_t)
impl_binary_operator_cmp_test_group(le, <=, i16, int16_t)
impl_binary_operator_cmp_test_group(le, <=, u32, uint32_t)
impl_binary_operator_cmp_test_group(le, <=, i32, int32_t)
impl_binary_operator_cmp_test_group(le, <=, u64, uint64_t)

impl_binary_operator_cmp_test_group(ge, >=, e8m23, float)
impl_binary_operator_cmp_test_group(ge, >=, e5m10, float16)
impl_binary_operator_cmp_test_group(ge, >=, u8, uint8_t)
impl_binary_operator_cmp_test_group(ge, >=, i8, int8_t)
impl_binary_operator_cmp_test_group(ge, >=, u16, uint16_t)
impl_binary_operator_cmp_test_group(ge, >=, i16, int16_t)
impl_binary_operator_cmp_test_group(ge, >=, u32, uint32_t)
impl_binary_operator_cmp_test_group(ge, >=, i32, int32_t)
impl_binary_operator_cmp_test_group(ge, >=, u64, uint64_t)

INSTANTIATE_TEST_SUITE_P(
    binary_operators_multi_backend,
    binary_operators,
    ValuesIn(get_supported_test_backends()),
    get_gtest_backend_name
);
