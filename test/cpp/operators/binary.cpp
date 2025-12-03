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
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        tensor t_a {ctx, ty, shape};
        if constexpr (std::is_same_v<T, bool>)
            t_a.fill_(std::bernoulli_distribution{}(gen));
        else if constexpr (std::is_integral_v<T>)
             t_a.fill_(std::uniform_int_distribution<T>{random_interval.first, random_interval.second}(gen));
        else
             t_a.fill_(std::uniform_real_distribution<float>{random_interval.first, random_interval.second}(gen));
        tensor t_b {t_a.clone()};
        std::vector<T> d_a {t_a.to_vector<T>()}; // must be cloned here as inplace op modified buffer
        std::vector<T> d_b {t_b.to_vector<T>()};
        tensor t_r {std::invoke(a, t_a, t_b)};
        if (inplace) {
            ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
        } else {
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        }
        std::vector<T> d_r {t_r.to_vector<T>()};
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
    for_all_test_shapes([&](const std::vector<int64_t>& shape){
        tensor t_a{ctx, ty, shape};
        if constexpr (std::is_same_v<T, bool>)
            t_a.fill_(std::bernoulli_distribution{}(gen));
        else if constexpr (std::is_integral_v<T>)
            t_a.fill_(std::uniform_int_distribution<T>{random_interval.first, random_interval.second}(gen));
        else
           t_a.fill_(std::uniform_real_distribution<float>{random_interval.first, random_interval.second}(gen));
        tensor t_b{t_a.clone()};
        std::vector<T> d_a {t_a.to_vector<T>()}; // must be cloned here as inplace op modifies buffer
        std::vector<T> d_b {t_b.to_vector<T>()};
        tensor t_r{std::invoke(a, t_a, t_b)};
        ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
        ASSERT_EQ(t_r.dtype(), dtype::boolean);
        ASSERT_EQ(d_a.size(), d_b.size());
        ASSERT_EQ(d_a.size(), t_r.numel());
        std::vector<bool> d_r {t_r.to_vector<bool>()};
        for (int64_t i = 0; i < d_r.size(); ++i)
            ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " ? " << d_b[i] << " = " << d_r[i];
    });
}

#define impl_binary_operator_test_group(name, op, opb, data_type, T) \
    TEST_P(binary_operators, name##_same_shape_##data_type) { \
        test_binary_operator<T>(GetParam(), false, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](T a, T b) -> T { return a op (b opb); } \
        ); \
    } \
    TEST_P(binary_operators, name##_inplace_same_shape_##data_type) { \
        test_binary_operator<T>(GetParam(), true, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](T a, T b) -> T { return a op (b opb); } \
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

#define NO_OP_B

impl_binary_operator_test_group(add, +, NO_OP_B, float32, float)
impl_binary_operator_test_group(add, +, NO_OP_B, float16, float16)
impl_binary_operator_test_group(add, +, NO_OP_B, u8, uint8_t)
impl_binary_operator_test_group(add, +, NO_OP_B, i8, int8_t)
impl_binary_operator_test_group(add, +, NO_OP_B, u16, uint16_t)
impl_binary_operator_test_group(add, +, NO_OP_B, i16, int16_t)
impl_binary_operator_test_group(add, +, NO_OP_B, u32, uint32_t)
impl_binary_operator_test_group(add, +, NO_OP_B, i32, int32_t)
impl_binary_operator_test_group(add, +, NO_OP_B, u64, uint64_t)
impl_binary_operator_test_group(add, +, NO_OP_B, i64, int64_t)


impl_binary_operator_test_group(sub, -, NO_OP_B, float32, float)
impl_binary_operator_test_group(sub, -, NO_OP_B, float16, float16)
impl_binary_operator_test_group(sub, -, NO_OP_B, u8, uint8_t)
impl_binary_operator_test_group(sub, -, NO_OP_B, i8, int8_t)
impl_binary_operator_test_group(sub, -, NO_OP_B, u16, uint16_t)
impl_binary_operator_test_group(sub, -, NO_OP_B, i16, int16_t)
impl_binary_operator_test_group(sub, -, NO_OP_B, u32, uint32_t)
impl_binary_operator_test_group(sub, -, NO_OP_B, i32, int32_t)
impl_binary_operator_test_group(sub, -, NO_OP_B, u64, uint64_t)
impl_binary_operator_test_group(sub, -, NO_OP_B, i64, int64_t)

impl_binary_operator_test_group(mul, *, NO_OP_B, float32, float)
impl_binary_operator_test_group(mul, *, NO_OP_B, float16, float16)
impl_binary_operator_test_group(mul, *, NO_OP_B, u8, uint8_t)
impl_binary_operator_test_group(mul, *, NO_OP_B, i8, int8_t)
impl_binary_operator_test_group(mul, *, NO_OP_B, u16, uint16_t)
impl_binary_operator_test_group(mul, *, NO_OP_B, i16, int16_t)
impl_binary_operator_test_group(mul, *, NO_OP_B, u32, uint32_t)
impl_binary_operator_test_group(mul, *, NO_OP_B, i32, int32_t)
impl_binary_operator_test_group(mul, *, NO_OP_B, u64, uint64_t)
impl_binary_operator_test_group(mul, *, NO_OP_B, i64, int64_t)

impl_binary_operator_test_group(div, /, NO_OP_B, float32, float)
impl_binary_operator_test_group(div, /, NO_OP_B, float16, float16)
impl_binary_operator_test_group(div, /, NO_OP_B, u8, uint8_t)
impl_binary_operator_test_group(div, /, NO_OP_B, i8, int8_t)
impl_binary_operator_test_group(div, /, NO_OP_B, u16, uint16_t)
impl_binary_operator_test_group(div, /, NO_OP_B, i16, int16_t)
impl_binary_operator_test_group(div, /, NO_OP_B, u32, uint32_t)
impl_binary_operator_test_group(div, /, NO_OP_B, i32, int32_t)
impl_binary_operator_test_group(div, /, NO_OP_B, u64, uint64_t)
impl_binary_operator_test_group(div, /, NO_OP_B, i64, int64_t)

impl_binary_operator_test_group(and, &, NO_OP_B, boolean, bool)
impl_binary_operator_test_group(and, &, NO_OP_B, u8, uint8_t)
impl_binary_operator_test_group(and, &, NO_OP_B, i8, int8_t)
impl_binary_operator_test_group(and, &, NO_OP_B, u16, uint16_t)
impl_binary_operator_test_group(and, &, NO_OP_B, i16, int16_t)
impl_binary_operator_test_group(and, &, NO_OP_B, u32, uint32_t)
impl_binary_operator_test_group(and, &, NO_OP_B, i32, int32_t)
impl_binary_operator_test_group(and, &, NO_OP_B, u64, uint64_t)
impl_binary_operator_test_group(and, &, NO_OP_B, i64, int64_t)

impl_binary_operator_test_group(or, |, NO_OP_B, boolean, bool)
impl_binary_operator_test_group(or, |, NO_OP_B, u8, uint8_t)
impl_binary_operator_test_group(or, |, NO_OP_B, i8, int8_t)
impl_binary_operator_test_group(or, |, NO_OP_B, u16, uint16_t)
impl_binary_operator_test_group(or, |, NO_OP_B, i16, int16_t)
impl_binary_operator_test_group(or, |, NO_OP_B, u32, uint32_t)
impl_binary_operator_test_group(or, |, NO_OP_B, i32, int32_t)
impl_binary_operator_test_group(or, |, NO_OP_B, u64, uint64_t)
impl_binary_operator_test_group(or, |, NO_OP_B, i64, int64_t)

impl_binary_operator_test_group(xor, ^, NO_OP_B, boolean, bool)
impl_binary_operator_test_group(xor, ^, NO_OP_B, u8, uint8_t)
impl_binary_operator_test_group(xor, ^, NO_OP_B, i8, int8_t)
impl_binary_operator_test_group(xor, ^, NO_OP_B, u16, uint16_t)
impl_binary_operator_test_group(xor, ^, NO_OP_B, i16, int16_t)
impl_binary_operator_test_group(xor, ^, NO_OP_B, u32, uint32_t)
impl_binary_operator_test_group(xor, ^, NO_OP_B, i32, int32_t)
impl_binary_operator_test_group(xor, ^, NO_OP_B, u64, uint64_t)
impl_binary_operator_test_group(xor, ^, NO_OP_B, i64, int64_t)

impl_binary_operator_test_group(shl, <<, &7, u8, uint8_t)
impl_binary_operator_test_group(shl, <<, &7, i8, int8_t)
impl_binary_operator_test_group(shl, <<, &15, u16, uint16_t)
impl_binary_operator_test_group(shl, <<, &15, i16, int16_t)
impl_binary_operator_test_group(shl, <<, &31, u32, uint32_t)
impl_binary_operator_test_group(shl, <<, &31, i32, int32_t)
impl_binary_operator_test_group(shl, <<, &63, u64, uint64_t)
impl_binary_operator_test_group(shl, <<, &63, i64, int64_t)

impl_binary_operator_test_group(shr, >>, &7, u8, uint8_t)
impl_binary_operator_test_group(shr, >>, &7, i8, int8_t)
impl_binary_operator_test_group(shr, >>, &15, u16, uint16_t)
impl_binary_operator_test_group(shr, >>, &15, i16, int16_t)
impl_binary_operator_test_group(shr, >>, &31, u32, uint32_t)
impl_binary_operator_test_group(shr, >>, &31, i32, int32_t)
impl_binary_operator_test_group(shr, >>, &63, u64, uint64_t)
impl_binary_operator_test_group(shr, >>, &63, i64, int64_t)

impl_binary_operator_cmp_test_group(eq, ==, float32, float)
impl_binary_operator_cmp_test_group(eq, ==, float16, float16)
impl_binary_operator_cmp_test_group(eq, ==, boolean, bool)
impl_binary_operator_cmp_test_group(eq, ==, u8, uint8_t)
impl_binary_operator_cmp_test_group(eq, ==, i8, int8_t)
impl_binary_operator_cmp_test_group(eq, ==, u16, uint16_t)
impl_binary_operator_cmp_test_group(eq, ==, i16, int16_t)
impl_binary_operator_cmp_test_group(eq, ==, u32, uint32_t)
impl_binary_operator_cmp_test_group(eq, ==, i32, int32_t)
impl_binary_operator_cmp_test_group(eq, ==, u64, uint64_t)
impl_binary_operator_cmp_test_group(eq, ==, i64, int64_t)

impl_binary_operator_cmp_test_group(ne, !=, float32, float)
impl_binary_operator_cmp_test_group(ne, !=, float16, float16)
impl_binary_operator_cmp_test_group(ne, !=, boolean, bool)
impl_binary_operator_cmp_test_group(ne, !=, u8, uint8_t)
impl_binary_operator_cmp_test_group(ne, !=, i8, int8_t)
impl_binary_operator_cmp_test_group(ne, !=, u16, uint16_t)
impl_binary_operator_cmp_test_group(ne, !=, i16, int16_t)
impl_binary_operator_cmp_test_group(ne, !=, u32, uint32_t)
impl_binary_operator_cmp_test_group(ne, !=, i32, int32_t)
impl_binary_operator_cmp_test_group(ne, !=, u64, uint64_t)
impl_binary_operator_cmp_test_group(ne, !=, i64, int64_t)

impl_binary_operator_cmp_test_group(lt, <, float32, float)
impl_binary_operator_cmp_test_group(lt, <, float16, float16)
impl_binary_operator_cmp_test_group(lt, <, u8, uint8_t)
impl_binary_operator_cmp_test_group(lt, <, i8, int8_t)
impl_binary_operator_cmp_test_group(lt, <, u16, uint16_t)
impl_binary_operator_cmp_test_group(lt, <, i16, int16_t)
impl_binary_operator_cmp_test_group(lt, <, u32, uint32_t)
impl_binary_operator_cmp_test_group(lt, <, i32, int32_t)
impl_binary_operator_cmp_test_group(lt, <, u64, uint64_t)
impl_binary_operator_cmp_test_group(lt, <, i64, int64_t)

impl_binary_operator_cmp_test_group(gt, >, float32, float)
impl_binary_operator_cmp_test_group(gt, >, float16, float16)
impl_binary_operator_cmp_test_group(gt, >, u8, uint8_t)
impl_binary_operator_cmp_test_group(gt, >, i8, int8_t)
impl_binary_operator_cmp_test_group(gt, >, u16, uint16_t)
impl_binary_operator_cmp_test_group(gt, >, i16, int16_t)
impl_binary_operator_cmp_test_group(gt, >, u32, uint32_t)
impl_binary_operator_cmp_test_group(gt, >, i32, int32_t)
impl_binary_operator_cmp_test_group(gt, >, u64, uint64_t)
impl_binary_operator_cmp_test_group(gt, >, i64, int64_t)

impl_binary_operator_cmp_test_group(le, <=, float32, float)
impl_binary_operator_cmp_test_group(le, <=, float16, float16)
impl_binary_operator_cmp_test_group(le, <=, u8, uint8_t)
impl_binary_operator_cmp_test_group(le, <=, i8, int8_t)
impl_binary_operator_cmp_test_group(le, <=, u16, uint16_t)
impl_binary_operator_cmp_test_group(le, <=, i16, int16_t)
impl_binary_operator_cmp_test_group(le, <=, u32, uint32_t)
impl_binary_operator_cmp_test_group(le, <=, i32, int32_t)
impl_binary_operator_cmp_test_group(le, <=, u64, uint64_t)
impl_binary_operator_cmp_test_group(le, <=, i64, int64_t)

impl_binary_operator_cmp_test_group(ge, >=, float32, float)
impl_binary_operator_cmp_test_group(ge, >=, float16, float16)
impl_binary_operator_cmp_test_group(ge, >=, u8, uint8_t)
impl_binary_operator_cmp_test_group(ge, >=, i8, int8_t)
impl_binary_operator_cmp_test_group(ge, >=, u16, uint16_t)
impl_binary_operator_cmp_test_group(ge, >=, i16, int16_t)
impl_binary_operator_cmp_test_group(ge, >=, u32, uint32_t)
impl_binary_operator_cmp_test_group(ge, >=, i32, int32_t)
impl_binary_operator_cmp_test_group(ge, >=, u64, uint64_t)
impl_binary_operator_cmp_test_group(ge, >=, i64, int64_t)

INSTANTIATE_TEST_SUITE_P(
    binary_operators_multi_backend,
    binary_operators,
    ValuesIn(get_supported_test_backends()),
    get_gtest_backend_name
);
