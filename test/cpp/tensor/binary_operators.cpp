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

#define impl_binary_operator_float_test_group(name, op, data_type) \
    TEST(cpu_tensor_binary_ops, name##_same_shape_##data_type) { \
        test::test_binary_float_operator<false, false>(lim, dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_broadcast_##data_type) { \
        test::test_binary_float_operator<true, false>(broadcast_lim, dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_inplace_same_shape_##data_type) { \
        test::test_binary_float_operator<false, true>(lim, dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_inplace_broadcast_##data_type) { \
        test::test_binary_float_operator<true, true>(broadcast_lim, dtype_eps_map.at(dtype::data_type), dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    }

#define impl_binary_operator_bool_test_group(name, op) \
    TEST(cpu_tensor_binary_ops, name##_same_shape_bool) { \
        test::test_binary_boolean_operator<false, false>(lim, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_broadcast_bool) { \
        test::test_binary_boolean_operator<true, false>(broadcast_lim, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_inplace_same_shape_bool) { \
        test::test_binary_boolean_operator<false, true>(lim, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_inplace_broadcast_bool) { \
        test::test_binary_boolean_operator<true, true>(broadcast_lim, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    }

#define impl_binary_operator_int_test_group(name, op, data_type, min, max) \
    TEST(cpu_tensor_binary_ops, name##_same_shape_##data_type) { \
        test::test_binary_int_operator<false, false>(lim, dtype::data_type, (min), (max), \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](std::int32_t a, std::int32_t b) -> std::int32_t { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_broadcast_##data_type) { \
        test::test_binary_int_operator<true, false>(broadcast_lim, dtype::data_type, (min), (max), \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](std::int32_t a, std::int32_t b) -> std::int32_t { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_inplace_same_shape_##data_type) { \
        test::test_binary_int_operator<false, true>(lim, dtype::data_type, (min), (max), \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](std::int32_t a, std::int32_t b) -> std::int32_t { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_inplace_broadcast_##data_type) { \
        test::test_binary_int_operator<true, true>(broadcast_lim, dtype::data_type, (min), (max), \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](std::int32_t a, std::int32_t b) -> std::int32_t { return a op b; } \
        ); \
    }

#define impl_binary_operator_cmp_float_test_group(name, op, data_type) \
    TEST(cpu_tensor_binary_ops, name##_same_shape_##data_type) { \
        test::test_binary_float_compare<false>(lim, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](float a, float b) -> bool { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_broadcast_##data_type) { \
        test::test_binary_float_compare<true>(broadcast_lim, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](float a, float b) -> bool { return a op b; } \
        ); \
    }

#define impl_binary_operator_cmp_int_test_group(name, op, data_type) \
    TEST(cpu_tensor_binary_ops, name##_same_shape_##data_type) { \
        test::test_binary_int_compare<false>(lim, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](std::int32_t a, std::int32_t b) -> bool { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_broadcast_##data_type) { \
        test::test_binary_int_compare<true>(broadcast_lim, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](std::int32_t a, std::int32_t b) -> bool { return a op b; } \
        ); \
    }

#define impl_binary_operator_cmp_bool_test_group(name, op) \
    TEST(cpu_tensor_binary_ops, name##_same_shape_bool) { \
        test::test_binary_bool_compare<false>(lim, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](bool a, bool b) -> bool { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_broadcast_bool) { \
        test::test_binary_bool_compare<true>(broadcast_lim, \
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

impl_binary_operator_int_test_group(add, +, i32, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max())
impl_binary_operator_int_test_group(sub, -, i32, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max())
impl_binary_operator_int_test_group(mul, *, i32, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max())
impl_binary_operator_int_test_group(div, /, i32, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max())
impl_binary_operator_int_test_group(and, &, i32, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max())
impl_binary_operator_int_test_group(or, |, i32, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max())
impl_binary_operator_int_test_group(xor, ^, i32, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max())
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
