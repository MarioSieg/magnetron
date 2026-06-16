/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include <prelude.hpp>
#include <core/mag_einsum.h>

using namespace magnetron;
using namespace magnetron::test;

static tensor einsum_eval(
    const char *equation,
    std::initializer_list<const tensor *> xs
) {
    std::vector<const mag_tensor_t *> args {};
    args.reserve(xs.size());
    for (const auto *x : xs)
        args.emplace_back(&**x);
    mag_tensor_t *out = nullptr;
    mag_error_t err {};
    handle_error(mag_einsum_eval(
        &err,
        &out,
        equation,
        args.data(),
        args.size()
    ));
    return tensor{out};
}

static void fill_iota(tensor &x, float start = 0.0f) {
    auto v = std::vector<float>(static_cast<size_t>(x.numel()));
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = start + static_cast<float>(i);
    x.copy_(v);
}

TEST(einsum, transpose_ij_to_ji) {
    context ctx{};
    tensor x{ctx, dtype::float32, 2, 3};
    fill_iota(x, 1.0f);

    tensor y = einsum_eval("ij->ji", {&x});

    ASSERT_EQ(y.rank(), 2);
    ASSERT_EQ(y.shape()[0], 3);
    ASSERT_EQ(y.shape()[1], 2);

    auto got = y.to_vector<float>();
    auto ref = x.to_vector<float>();

    for (int64_t i = 0; i < 2; ++i)
        for (int64_t j = 0; j < 3; ++j)
            ASSERT_FLOAT_EQ(got[j * 2 + i], ref[i * 3 + j]);
}

TEST(einsum, sum_all_ij_to_scalar) {
    context ctx{};
    tensor x{ctx, dtype::float32, 2, 3};
    fill_iota(x, 1.0f);

    tensor y = einsum_eval("ij->", {&x});

    ASSERT_EQ(y.rank(), 0);

    auto got = y.to_vector<float>();
    ASSERT_EQ(got.size(), 1);

    ASSERT_FLOAT_EQ(got[0], 21.0f);
}

TEST(einsum, row_sum_ij_to_i) {
    context ctx{};
    tensor x{ctx, dtype::float32, 2, 3};
    fill_iota(x, 1.0f);

    tensor y = einsum_eval("ij->i", {&x});

    ASSERT_EQ(y.rank(), 1);
    ASSERT_EQ(y.shape()[0], 2);

    auto got = y.to_vector<float>();

    ASSERT_FLOAT_EQ(got[0], 1.0f + 2.0f + 3.0f);
    ASSERT_FLOAT_EQ(got[1], 4.0f + 5.0f + 6.0f);
}

TEST(einsum, col_sum_ij_to_j) {
    context ctx{};
    tensor x{ctx, dtype::float32, 2, 3};
    fill_iota(x, 1.0f);

    tensor y = einsum_eval("ij->j", {&x});

    ASSERT_EQ(y.rank(), 1);
    ASSERT_EQ(y.shape()[0], 3);

    auto got = y.to_vector<float>();

    ASSERT_FLOAT_EQ(got[0], 1.0f + 4.0f);
    ASSERT_FLOAT_EQ(got[1], 2.0f + 5.0f);
    ASSERT_FLOAT_EQ(got[2], 3.0f + 6.0f);
}

TEST(einsum, dot_i_i_to_scalar) {
    context ctx{};
    tensor a{ctx, dtype::float32, 4};
    tensor b{ctx, dtype::float32, 4};

    a.copy_(std::vector<float>{1, 2, 3, 4});
    b.copy_(std::vector<float>{10, 20, 30, 40});

    tensor y = einsum_eval("i,i->", {&a, &b});

    ASSERT_EQ(y.rank(), 0);

    auto got = y.to_vector<float>();
    ASSERT_EQ(got.size(), 1);
    ASSERT_FLOAT_EQ(got[0], 300.0f);
}

TEST(einsum, outer_i_j_to_ij) {
    context ctx{};
    tensor a{ctx, dtype::float32, 2};
    tensor b{ctx, dtype::float32, 3};

    a.copy_(std::vector<float>{2, 3});
    b.copy_(std::vector<float>{10, 20, 30});

    tensor y = einsum_eval("i,j->ij", {&a, &b});

    ASSERT_EQ(y.rank(), 2);
    ASSERT_EQ(y.shape()[0], 2);
    ASSERT_EQ(y.shape()[1], 3);

    auto got = y.to_vector<float>();

    ASSERT_FLOAT_EQ(got[0], 20.0f);
    ASSERT_FLOAT_EQ(got[1], 40.0f);
    ASSERT_FLOAT_EQ(got[2], 60.0f);
    ASSERT_FLOAT_EQ(got[3], 30.0f);
    ASSERT_FLOAT_EQ(got[4], 60.0f);
    ASSERT_FLOAT_EQ(got[5], 90.0f);
}

TEST(einsum, matmul_ij_jk_to_ik) {
    context ctx{};
    tensor a{ctx, dtype::float32, 2, 3};
    tensor b{ctx, dtype::float32, 3, 2};

    a.copy_(std::vector<float>{
        1, 2, 3,
        4, 5, 6
    });

    b.copy_(std::vector<float>{
        10, 11,
        20, 21,
        30, 31
    });

    tensor y = einsum_eval("ij,jk->ik", {&a, &b});

    ASSERT_EQ(y.rank(), 2);
    ASSERT_EQ(y.shape()[0], 2);
    ASSERT_EQ(y.shape()[1], 2);

    auto got = y.to_vector<float>();

    ASSERT_FLOAT_EQ(got[0], 140.0f);
    ASSERT_FLOAT_EQ(got[1], 146.0f);
    ASSERT_FLOAT_EQ(got[2], 320.0f);
    ASSERT_FLOAT_EQ(got[3], 335.0f);
}

TEST(einsum, batch_matmul_bij_bjk_to_bik) {
    context ctx{};
    tensor a{ctx, dtype::float32, 2, 2, 3};
    tensor b{ctx, dtype::float32, 2, 3, 2};

    fill_iota(a, 1.0f);
    fill_iota(b, 1.0f);

    tensor y = einsum_eval("bij,bjk->bik", {&a, &b});

    ASSERT_EQ(y.rank(), 3);
    ASSERT_EQ(y.shape()[0], 2);
    ASSERT_EQ(y.shape()[1], 2);
    ASSERT_EQ(y.shape()[2], 2);

    auto av = a.to_vector<float>();
    auto bv = b.to_vector<float>();
    auto got = y.to_vector<float>();

    for (int64_t batch = 0; batch < 2; ++batch) {
        for (int64_t i = 0; i < 2; ++i) {
            for (int64_t k = 0; k < 2; ++k) {
                float acc = 0.0f;
                for (int64_t j = 0; j < 3; ++j) {
                    float aa = av[batch * 6 + i * 3 + j];
                    float bb = bv[batch * 6 + j * 2 + k];
                    acc += aa * bb;
                }

                ASSERT_FLOAT_EQ(got[batch * 4 + i * 2 + k], acc);
            }
        }
    }
}

TEST(einsum, diagonal_ii_to_i) {
    context ctx{};
    tensor x{ctx, dtype::float32, 3, 3};

    x.copy_(std::vector<float>{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    tensor y = einsum_eval("ii->i", {&x});

    ASSERT_EQ(y.rank(), 1);
    ASSERT_EQ(y.shape()[0], 3);

    auto got = y.to_vector<float>();

    ASSERT_FLOAT_EQ(got[0], 1.0f);
    ASSERT_FLOAT_EQ(got[1], 5.0f);
    ASSERT_FLOAT_EQ(got[2], 9.0f);
}

TEST(einsum, trace_ii_to_scalar) {
    context ctx{};
    tensor x{ctx, dtype::float32, 3, 3};

    x.copy_(std::vector<float>{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    tensor y = einsum_eval("ii->", {&x});

    ASSERT_EQ(y.rank(), 0);

    auto got = y.to_vector<float>();
    ASSERT_EQ(got.size(), 1);

    ASSERT_FLOAT_EQ(got[0], 15.0f);
}

TEST(einsum, implicit_matmul_ij_jk) {
    context ctx{};
    tensor a{ctx, dtype::float32, 2, 3};
    tensor b{ctx, dtype::float32, 3, 2};

    a.copy_(std::vector<float>{
        1, 2, 3,
        4, 5, 6
    });

    b.copy_(std::vector<float>{
        10, 11,
        20, 21,
        30, 31
    });

    tensor y = einsum_eval("ij,jk", {&a, &b});

    ASSERT_EQ(y.rank(), 2);
    ASSERT_EQ(y.shape()[0], 2);
    ASSERT_EQ(y.shape()[1], 2);

    auto got = y.to_vector<float>();

    ASSERT_FLOAT_EQ(got[0], 140.0f);
    ASSERT_FLOAT_EQ(got[1], 146.0f);
    ASSERT_FLOAT_EQ(got[2], 320.0f);
    ASSERT_FLOAT_EQ(got[3], 335.0f);
}
