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

static constexpr int64_t lim {4};

static auto naive_matmul(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t N,
    int64_t K
) -> void {
    for (int64_t i {}; i < M*N; ++i) C[i] = 0.0f;
    for (int64_t i {}; i < M; ++i) {
        for (int64_t k {}; k < K; ++k) {
            float a_ik {A[i*K + k]};
            for (int64_t j {}; j < N; ++j) {
                C[i*N + j] += a_ik * B[k*N + j];
            }
        }
    }
}

TEST(cpu_tensor_binary_ops, matmul_naive) {
    static constexpr std::array A {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    static constexpr std::array B {0.5f, -1.0f};
    std::array<float, 3> C {};
    naive_matmul(A.data(), B.data(), C.data(), 3, 1, 2);
    ASSERT_FLOAT_EQ(C[0], -1.5f);
    ASSERT_FLOAT_EQ(C[1], -2.5f);
    ASSERT_FLOAT_EQ(C[2], -3.5f);
}

template <const size_t M, const size_t N, typename T>
[[nodiscard]] auto flatten(const std::array<std::array<T, N>, M>& array) -> std::vector<T> {
    auto* p = reinterpret_cast<const T*>(&array);
    return std::vector<T> {
        p,
        p+M*N
    };
}

TEST(cpu_tensor_binary_ops, matmul_fixed_square_float32) {
    static constexpr std::array A {
        std::array{1.6354027f, -1.3607267f},
        std::array{1.8556793f, 1.1689897f}
    };
    static constexpr std::array B {
        std::array{-0.6105532f, 0.10695228f},
        std::array{-1.0069681f, -0.40955952f}
    };
    static constexpr std::array C {
        std::array{0.3717081f, 0.7322086f},
        std::array{-2.3101263f, -0.28030172f}
    };
    context ctx {};
    tensor a {ctx, dtype::float32, A.size(), A[0].size()};
    tensor b {ctx, dtype::float32, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector cr {c.to_vector<float>()};
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], reinterpret_cast<const float*>(&C)[i]);
    }
}

#if 0
TEST(cpu_tensor_binary_ops, matmul_fixed_square_float16) {
    static constexpr std::array A {
        std::array{1.6354027f, -1.3607267f},
        std::array{1.8556793f, 1.1689897f}
    };
    static constexpr std::array B {
        std::array{-0.6105532f, 0.10695228f},
        std::array{-1.0069681f, -0.40955952f}
    };
    static constexpr std::array C {
        std::array{0.3717081f, 0.7322086f},
        std::array{-2.3101263f, -0.28030172f}
    };
    context ctx {};
    tensor a {ctx, dtype::float16, A.size(), A[0].size()};
    tensor b {ctx, dtype::float16, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector cr {c.to_vector<float16>()};
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_NEAR(cr[i], reinterpret_cast<const float*>(&C)[i], 1e-2);
    }
}
#endif

TEST(cpu_tensor_binary_ops, matmul_fixed_non_square_float32) {
    static constexpr std::array A {
        std::array{1.0f, 2.0f},
        std::array{3.0f, 4.0f},
        std::array{5.0f, 6.0f}
    };
    static constexpr std::array B {
        std::array{7.0f, 8.0f, 9.0f, 10.0f},
        std::array{11.0f, 12.0f, 13.0f, 14.0f}
    };
    static constexpr std::array<std::array<float, 4>, 3> C{{
        {{1.0f*7.0f + 2.0f*11, 1.0f*8 + 2.0f*12.0f, 1.0f*9.0f + 2.0f*13.0f, 1.0f*10.0f + 2.0f*14.0f}},
        {{3.0f*7.0f + 4.0f*11, 3.0f*8 + 4.0f*12.0f, 3.0f*9.0f + 4.0f*13.0f, 3.0f*10.0f + 4.0f*14.0f}},
        {{5.0f*7.0f + 6.0f*11, 5.0f*8 + 6.0f*12.0f, 5.0f*9.0f + 6.0f*13.0f, 5.0f*10.0f + 6.0f*14.0f}}
    }};
    context ctx {};
    tensor a {ctx, dtype::float32, A.size(), A[0].size()};
    tensor b {ctx, dtype::float32, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 3);
    ASSERT_EQ(c.shape()[1], 4);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 12);
    std::vector cr {c.to_vector<float>()};
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], reinterpret_cast<const float*>(&C)[i]);
    }
}

#if 0
TEST(cpu_tensor_binary_ops, matmul_fixed_non_square_float16) {
    static constexpr std::array A {
        std::array{1.0f, 2.0f},
        std::array{3.0f, 4.0f},
        std::array{5.0f, 6.0f}
    };
    static constexpr std::array B {
        std::array{7.0f, 8.0f, 9.0f, 10.0f},
        std::array{11.0f, 12.0f, 13.0f, 14.0f}
    };
    static constexpr std::array<std::array<float, 4>, 3> C{{
        {{1.0f*7.0f + 2.0f*11, 1.0f*8 + 2.0f*12.0f, 1.0f*9.0f + 2.0f*13.0f, 1.0f*10.0f + 2.0f*14.0f}},
        {{3.0f*7.0f + 4.0f*11, 3.0f*8 + 4.0f*12.0f, 3.0f*9.0f + 4.0f*13.0f, 3.0f*10.0f + 4.0f*14.0f}},
        {{5.0f*7.0f + 6.0f*11, 5.0f*8 + 6.0f*12.0f, 5.0f*9.0f + 6.0f*13.0f, 5.0f*10.0f + 6.0f*14.0f}}
    }};
    context ctx {};
    tensor a {ctx, dtype::float16, A.size(), A[0].size()};
    tensor b {ctx, dtype::float16, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 3);
    ASSERT_EQ(c.shape()[1], 4);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 12);
    std::vector cr {c.to_vector<float16>()};
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_NEAR(cr[i], reinterpret_cast<const float*>(&C)[i], 1e-2);
    }
}
#endif

TEST(cpu_tensor_binary_ops, matmul_fixed_square_zero_float32) {
    static constexpr std::array A {
        std::array{1.6354027f, -1.3607267f},
        std::array{1.8556793f, 1.1689897f}
    };
    static constexpr std::array B {
        std::array{0.0f, 0.0f},
        std::array{0.0f, 0.0f}
    };
    context ctx {};
    tensor a {ctx, dtype::float32, A.size(), A[0].size()};
    tensor b {ctx, dtype::float32, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector cr {c.to_vector<float>()};
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], 0.0f);
    }
}

#if 0
TEST(cpu_tensor_binary_ops, matmul_fixed_square_zero_float16) {
    static constexpr std::array A {
        std::array{1.6354027f, -1.3607267f},
        std::array{1.8556793f, 1.1689897f}
    };
    static constexpr std::array B {
        std::array{0.0f, 0.0f},
        std::array{0.0f, 0.0f}
    };
    context ctx {};
    tensor a {ctx, dtype::float16, A.size(), A[0].size()};
    tensor b {ctx, dtype::float16, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector cr {c.to_vector<float16>()};
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], 0.0f);
    }
}
#endif

TEST(cpu_tensor_binary_ops, matmul_fixed_square_identity_float32) {
    static constexpr std::array A {
        std::array{1.6354027f, -1.3607267f},
        std::array{1.8556793f, 1.1689897f}
    };
    static constexpr std::array B {
        std::array{1.0f, 0.0f},
        std::array{0.0f, 1.0f}
    };
    static constexpr std::array C {A};
    context ctx {};
    tensor a {ctx, dtype::float32, A.size(), A[0].size()};
    tensor b {ctx, dtype::float32, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector cr {c.to_vector<float>()};
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(cr[i], reinterpret_cast<const float*>(&C)[i]);
    }
}

#if 0
TEST(cpu_tensor_binary_ops, matmul_fixed_square_identity_float16) {
    static constexpr std::array A {
        std::array{1.6354027f, -1.3607267f},
        std::array{1.8556793f, 1.1689897f}
    };
    static constexpr std::array B {
        std::array{1.0f, 0.0f},
        std::array{0.0f, 1.0f}
    };
    static constexpr std::array C {A};
    context ctx {};
    tensor a {ctx, dtype::float16, A.size(), A[0].size()};
    tensor b {ctx, dtype::float16, B.size(), B[0].size()};
    a.fill_from(flatten(A));
    b.fill_from(flatten(B));
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.rank(), 2);
    ASSERT_EQ(c.numel(), 4);
    ASSERT_EQ(c.numel(), a.numel());
    ASSERT_EQ(c.numel(), b.numel());
    std::vector cr {c.to_vector<float16>()};
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_NEAR(cr[i], reinterpret_cast<const float*>(&C)[i], 1e-2);
    }
}
#endif

TEST(cpu_tensor_binary_ops, matmul_fixed_matrix_vector_float32) {
    static constexpr std::array A {
        std::array{1.0f, 2.0f},
        std::array{3.0f, 4.0f},
        std::array{5.0f, 6.0f},
    };
    std::vector<float> B {
        0.5f, -1.0f
    };
    static constexpr std::array<float, 3> C {
        {-1.5f, -2.5f, -3.5f}
    };
    context ctx {};
    tensor a {ctx, dtype::float32, A.size(), A[0].size()};
    tensor b {ctx, dtype::float32, B.size()};
    a.fill_from(flatten(A));
    b.fill_from(B);
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 3);
    ASSERT_EQ(c.rank(), 1);
    ASSERT_EQ(c.numel(), 3);
    ASSERT_NE(c.numel(), a.numel());
    ASSERT_NE(c.numel(), b.numel());
    std::vector<float> result = c.to_vector<float>();
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(result[i], C[i]);
    }
}

#if 0
TEST(cpu_tensor_binary_ops, matmul_fixed_matrix_vector_float16) {
    static constexpr std::array A {
        std::array{1.0f, 2.0f},
        std::array{3.0f, 4.0f},
        std::array{5.0f, 6.0f},
    };
    std::vector<float> B {
        0.5f, -1.0f
    };
    static constexpr std::array<float, 3> C {
        {-1.5f, -2.5f, -3.5f}
    };
    context ctx {};
    tensor a {ctx, dtype::float16, A.size(), A[0].size()};
    tensor b {ctx, dtype::float16, B.size()};
    a.fill_from(flatten(A));
    b.fill_from(B);
    tensor c {a%b};
    ASSERT_EQ(c.shape()[0], 3);
    ASSERT_EQ(c.rank(), 1);
    ASSERT_EQ(c.numel(), 3);
    ASSERT_NE(c.numel(), a.numel());
    ASSERT_NE(c.numel(), b.numel());
    std::vector result = c.to_vector<float16>();
    for (int64_t i {}; i < c.numel(); ++i) {
        ASSERT_FLOAT_EQ(result[i], C[i]);
    }
}
#endif
