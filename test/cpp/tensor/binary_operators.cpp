// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

static constexpr std::int64_t lim {4};
static constexpr std::int64_t broadcast_lim {lim-1};

#define impl_binary_operator_float_test_group(name, op, data_type) \
    TEST(cpu_tensor_binary_ops, name##_same_shape_##data_type) { \
        test::test_binary_float_operator<false, false>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_broadcast_##data_type) { \
        test::test_binary_float_operator<true, false>(broadcast_lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_inplace_same_shape_##data_type) { \
        test::test_binary_float_operator<false, true>(lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
            [](tensor a, tensor b) -> tensor { return a op##= b; }, \
            [](float a, float b) -> float { return a op b; } \
        ); \
    } \
    TEST(cpu_tensor_binary_ops, name##_inplace_broadcast_##data_type) { \
        test::test_binary_float_operator<true, true>(broadcast_lim, test::dtype_traits<test::data_type##_t>::test_eps, dtype::data_type, \
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

static auto naive_matmul(
    const e8m23_t* A,
    const e8m23_t* B,
    e8m23_t* C,
    std::int64_t M,
    std::int64_t N,
    std::int64_t K
) -> void {
    for (std::int64_t i {}; i < M*N; ++i) C[i] = 0.0f;
    for (std::int64_t i {}; i < M; ++i) {
        for (std::int64_t k {}; k < K; ++k) {
            e8m23_t a_ik {A[i*K + k]};
            for (std::int64_t j {}; j < N; ++j) {
                C[i*N + j] += a_ik * B[k*N + j];
            }
        }
    }
}

TEST(cpu_tensor_binary_ops, matmul_naive) {
    static constexpr std::array<e8m23_t, 6> A {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    static constexpr std::array<e8m23_t, 2> B {0.5f, -1.0f};
    std::array<e8m23_t, 3> C {};
    naive_matmul(A.data(), B.data(), C.data(), 3, 1, 2);
    ASSERT_FLOAT_EQ(C[0], -1.5f);
    ASSERT_FLOAT_EQ(C[1], -2.5f);
    ASSERT_FLOAT_EQ(C[2], -3.5f);
}

template <const std::size_t M, const std::size_t N, typename T>
[[nodiscard]] auto flatten(const std::array<std::array<T, N>, M>& array) -> std::span<const e8m23_t> {
    return std::span<const T> {
        reinterpret_cast<const T*>(&array),
        M*N
    };
}

TEST(cpu_tensor_binary_ops, matmul_square_e8m23) {
    static constexpr std::array<std::array<e8m23_t, 2>, 2> A {
        std::array<e8m23_t, 2>{1.6354027f, -1.3607267f},
        std::array<e8m23_t, 2>{1.8556793f, 1.1689897f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> B {
        std::array<e8m23_t, 2>{-0.6105532f, 0.10695228f},
        std::array<e8m23_t, 2>{-1.0069681f, -0.40955952f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> C {
        std::array<e8m23_t, 2>{0.3717081f, 0.7322086f},
        std::array<e8m23_t, 2>{-2.3101263f, -0.28030172f}
    };
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, A.size(), A[0].size()};
    tensor b {ctx, dtype::e8m23, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector<e8m23_t> cr {c.to_float_vector()};
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], reinterpret_cast<const e8m23_t*>(&C)[i]);
    }
}

TEST(cpu_tensor_binary_ops, matmul_square_e5m10) {
    static constexpr std::array<std::array<e8m23_t, 2>, 2> A {
        std::array<e8m23_t, 2>{1.6354027f, -1.3607267f},
        std::array<e8m23_t, 2>{1.8556793f, 1.1689897f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> B {
        std::array<e8m23_t, 2>{-0.6105532f, 0.10695228f},
        std::array<e8m23_t, 2>{-1.0069681f, -0.40955952f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> C {
        std::array<e8m23_t, 2>{0.3717081f, 0.7322086f},
        std::array<e8m23_t, 2>{-2.3101263f, -0.28030172f}
    };
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e5m10, A.size(), A[0].size()};
    tensor b {ctx, dtype::e5m10, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector<e8m23_t> cr {c.to_float_vector()};
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_NEAR(cr[i], reinterpret_cast<const e8m23_t*>(&C)[i], 1e-2);
    }
}

TEST(cpu_tensor_binary_ops, matmul_non_square_e8m23) {
    static constexpr std::array<std::array<e8m23_t, 2>, 3> A {
        std::array<e8m23_t, 2>{1.0f, 2.0f},
        std::array<e8m23_t, 2>{3.0f, 4.0f},
        std::array<e8m23_t, 2>{5.0f, 6.0f}
    };
    static constexpr std::array<std::array<e8m23_t, 4>, 2> B {
        std::array<e8m23_t, 4>{7.0f, 8.0f, 9.0f, 10.0f},
        std::array<e8m23_t, 4>{11.0f, 12.0f, 13.0f, 14.0f}
    };
    static constexpr std::array<std::array<e8m23_t, 4>, 3> C{{
        {{1.0f*7.0f + 2.0f*11, 1.0f*8 + 2.0f*12.0f, 1.0f*9.0f + 2.0f*13.0f, 1.0f*10.0f + 2.0f*14.0f}},
        {{3.0f*7.0f + 4.0f*11, 3.0f*8 + 4.0f*12.0f, 3.0f*9.0f + 4.0f*13.0f, 3.0f*10.0f + 4.0f*14.0f}},
        {{5.0f*7.0f + 6.0f*11, 5.0f*8 + 6.0f*12.0f, 5.0f*9.0f + 6.0f*13.0f, 5.0f*10.0f + 6.0f*14.0f}}
    }};
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, A.size(), A[0].size()};
    tensor b {ctx, dtype::e8m23, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 3);
    ASSERT_EQ(c.shape()[1], 4);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 12);
    std::vector<e8m23_t> cr {c.to_float_vector()};
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], reinterpret_cast<const e8m23_t*>(&C)[i]);
    }
}

TEST(cpu_tensor_binary_ops, matmul_non_square_e5m10) {
    static constexpr std::array<std::array<e8m23_t, 2>, 3> A {
        std::array<e8m23_t, 2>{1.0f, 2.0f},
        std::array<e8m23_t, 2>{3.0f, 4.0f},
        std::array<e8m23_t, 2>{5.0f, 6.0f}
    };
    static constexpr std::array<std::array<e8m23_t, 4>, 2> B {
        std::array<e8m23_t, 4>{7.0f, 8.0f, 9.0f, 10.0f},
        std::array<e8m23_t, 4>{11.0f, 12.0f, 13.0f, 14.0f}
    };
    static constexpr std::array<std::array<e8m23_t, 4>, 3> C{{
        {{1.0f*7.0f + 2.0f*11, 1.0f*8 + 2.0f*12.0f, 1.0f*9.0f + 2.0f*13.0f, 1.0f*10.0f + 2.0f*14.0f}},
        {{3.0f*7.0f + 4.0f*11, 3.0f*8 + 4.0f*12.0f, 3.0f*9.0f + 4.0f*13.0f, 3.0f*10.0f + 4.0f*14.0f}},
        {{5.0f*7.0f + 6.0f*11, 5.0f*8 + 6.0f*12.0f, 5.0f*9.0f + 6.0f*13.0f, 5.0f*10.0f + 6.0f*14.0f}}
    }};
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e5m10, A.size(), A[0].size()};
    tensor b {ctx, dtype::e5m10, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 3);
    ASSERT_EQ(c.shape()[1], 4);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 12);
    std::vector<e8m23_t> cr {c.to_float_vector()};
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_NEAR(cr[i], reinterpret_cast<const e8m23_t*>(&C)[i], 1e-2);
    }
}

TEST(cpu_tensor_binary_ops, matmul_square_zero_e8m23) {
    static constexpr std::array<std::array<e8m23_t, 2>, 2> A {
        std::array<e8m23_t, 2>{1.6354027f, -1.3607267f},
        std::array<e8m23_t, 2>{1.8556793f, 1.1689897f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> B {
        std::array<e8m23_t, 2>{0.0f, 0.0f},
        std::array<e8m23_t, 2>{0.0f, 0.0f}
    };
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, A.size(), A[0].size()};
    tensor b {ctx, dtype::e8m23, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector<e8m23_t> cr {c.to_float_vector()};
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], 0.0f);
    }
}

TEST(cpu_tensor_binary_ops, matmul_square_zero_e5m10) {
    static constexpr std::array<std::array<e8m23_t, 2>, 2> A {
        std::array<e8m23_t, 2>{1.6354027f, -1.3607267f},
        std::array<e8m23_t, 2>{1.8556793f, 1.1689897f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> B {
        std::array<e8m23_t, 2>{0.0f, 0.0f},
        std::array<e8m23_t, 2>{0.0f, 0.0f}
    };
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e5m10, A.size(), A[0].size()};
    tensor b {ctx, dtype::e5m10, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector<e8m23_t> cr {c.to_float_vector()};
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], 0.0f);
    }
}

TEST(cpu_tensor_binary_ops, matmul_square_identity_e8m23) {
    static constexpr std::array<std::array<e8m23_t, 2>, 2> A {
        std::array<e8m23_t, 2>{1.6354027f, -1.3607267f},
        std::array<e8m23_t, 2>{1.8556793f, 1.1689897f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> B {
        std::array<e8m23_t, 2>{1.0f, 0.0f},
        std::array<e8m23_t, 2>{0.0f, 1.0f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> C {A};
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, A.size(), A[0].size()};
    tensor b {ctx, dtype::e8m23, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector<e8m23_t> cr {c.to_float_vector()};
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], reinterpret_cast<const e8m23_t*>(&C)[i]);
    }
}

TEST(cpu_tensor_binary_ops, matmul_square_identity_e5m10) {
    static constexpr std::array<std::array<e8m23_t, 2>, 2> A {
        std::array<e8m23_t, 2>{1.6354027f, -1.3607267f},
        std::array<e8m23_t, 2>{1.8556793f, 1.1689897f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> B {
        std::array<e8m23_t, 2>{1.0f, 0.0f},
        std::array<e8m23_t, 2>{0.0f, 1.0f}
    };
    static constexpr std::array<std::array<e8m23_t, 2>, 2> C {A};
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e5m10, A.size(), A[0].size()};
    tensor b {ctx, dtype::e5m10, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector<e8m23_t> cr {c.to_float_vector()};
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_NEAR(cr[i], reinterpret_cast<const e8m23_t*>(&C)[i], 1e-2);
    }
}

TEST(cpu_tensor_binary_ops, matmul_matrix_vector_e8m23) {
    static constexpr std::array<std::array<e8m23_t, 2>, 3> A {
        std::array<e8m23_t, 2>{1.0f, 2.0f},
        std::array<e8m23_t, 2>{3.0f, 4.0f},
        std::array<e8m23_t, 2>{5.0f, 6.0f},
    };
    static constexpr std::array<e8m23_t, 2> B {
        0.5f, -1.0f
    };
    static constexpr std::array<e8m23_t, 3> C {
        {-1.5f, -2.5f, -3.5f}
    };
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, A.size(), A[0].size()};
    tensor b {ctx, dtype::e8m23, B.size()};
    a.fill_from(flatten(A));
    b.fill_from(B);
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 3);
    ASSERT_EQ(c.rank(), 1);
    ASSERT_EQ(c.numel(), 3);
    ASSERT_NE(c.numel(), a.numel());
    ASSERT_NE(c.numel(), b.numel());
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(c(i), C[i]);
    }
}

TEST(cpu_tensor_binary_ops, matmul_matrix_vector_e5m10) {
    static constexpr std::array<std::array<e8m23_t, 2>, 3> A {
        std::array<e8m23_t, 2>{1.0f, 2.0f},
        std::array<e8m23_t, 2>{3.0f, 4.0f},
        std::array<e8m23_t, 2>{5.0f, 6.0f},
    };
    static constexpr std::array<e8m23_t, 2> B {
        0.5f, -1.0f
    };
    static constexpr std::array<e8m23_t, 3> C {
        {-1.5f, -2.5f, -3.5f}
    };
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e5m10, A.size(), A[0].size()};
    tensor b {ctx, dtype::e5m10, B.size()};
    a.fill_from(flatten(A));
    b.fill_from(B);
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 3);
    ASSERT_EQ(c.rank(), 1);
    ASSERT_EQ(c.numel(), 3);
    ASSERT_NE(c.numel(), a.numel());
    ASSERT_NE(c.numel(), b.numel());
    for (std::int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(c(i), C[i]);
    }
}

