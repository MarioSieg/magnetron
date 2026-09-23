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

static mag_einsum_path_t einsum_path(
    const char *equation,
    std::initializer_list<const tensor *> xs
) {
    std::vector<const mag_tensor_t *> args {};
    args.reserve(xs.size());
    for (const auto *x : xs)
        args.emplace_back(&**x);
    mag_einsum_path_t path {};
    mag_error_t err {};
    handle_error(mag_einsum_path(
        &err,
        &path,
        equation,
        args.data(),
        args.size()
    ));
    return path;
}

TEST(einsum, path_single_operand) {
    context ctx{};
    tensor x{ctx, dtype::float32, 8, 8};
    mag_einsum_path_t path = einsum_path("ij->ji", {&x});
    ASSERT_EQ(path.num_steps, 1);
    ASSERT_EQ(path.steps[0][0], 0u);
    ASSERT_EQ(path.steps[0][1], UINT32_MAX);
    ASSERT_EQ(path.opt_cost, path.naive_cost);
}

TEST(einsum, path_pair_is_single_step) {
    context ctx{};
    tensor a{ctx, dtype::float32, 32, 64};
    tensor b{ctx, dtype::float32, 64, 16};
    mag_einsum_path_t path = einsum_path("ij,jk->ik", {&a, &b});
    ASSERT_EQ(path.num_steps, 1);
    ASSERT_EQ(path.steps[0][0], 0u);
    ASSERT_EQ(path.steps[0][1], 1u);
    ASSERT_EQ(path.opt_cost, 2u*32*64*16);
}

TEST(einsum, path_optimal_beats_greedy_on_four_operands) {
    context ctx{};
    tensor a{ctx, dtype::float32, 32, 32, 64};
    tensor b{ctx, dtype::float32, 64, 64};
    tensor c{ctx, dtype::float32, 64, 64};
    tensor d{ctx, dtype::float32, 32, 64, 32};
    mag_einsum_path_t path = einsum_path("abc,cd,de,bef->af", {&a, &b, &c, &d});
    ASSERT_EQ(path.num_steps, 3);
    ASSERT_EQ(path.naive_cost, 34359738368u);
    ASSERT_EQ(path.opt_cost, 13107200u);
}

TEST(einsum, path_matches_opt_einsum_on_five_operands) {
    context ctx{};
    tensor p{ctx, dtype::float32, 24, 24};
    tensor q{ctx, dtype::float32, 24, 24};
    tensor t{ctx, dtype::float32, 24, 24, 24, 24};
    tensor r{ctx, dtype::float32, 24, 24};
    tensor s{ctx, dtype::float32, 24, 24};
    mag_einsum_path_t path = einsum_path("pi,qj,ijkl,rk,sl->pqrs", {&p, &q, &t, &r, &s});
    ASSERT_EQ(path.num_steps, 4);
    ASSERT_EQ(path.naive_cost, 550376570880u);
    ASSERT_EQ(path.opt_cost, 63700992u);
    ASSERT_EQ(path.opt_scaling, 5u);
}

TEST(einsum, path_no_memory_limit_fallback_on_three_operands) {
    context ctx{};
    tensor a{ctx, dtype::float32, 64, 128, 128};
    tensor b{ctx, dtype::float32, 128, 256};
    tensor c{ctx, dtype::float32, 128, 256};
    mag_einsum_path_t path = einsum_path("ijk,jl,kl->il", {&a, &b, &c});
    ASSERT_EQ(path.num_steps, 2);
    ASSERT_EQ(path.naive_cost, 805306368u);
    ASSERT_EQ(path.opt_cost, 541065216u);
}

TEST(einsum, path_branch_bound_on_six_operands) {
    context ctx{};
    tensor xs[6] = {
        tensor{ctx, dtype::float32, 48, 48}, tensor{ctx, dtype::float32, 48, 48}, tensor{ctx, dtype::float32, 48, 48},
        tensor{ctx, dtype::float32, 48, 48}, tensor{ctx, dtype::float32, 48, 48}, tensor{ctx, dtype::float32, 48, 48},
    };
    mag_einsum_path_t path = einsum_path("ab,ac,ad,bc,bd,cd->", {&xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5]});
    ASSERT_EQ(path.num_steps, 5);
    ASSERT_EQ(path.naive_cost, 31850496u);
    ASSERT_LE(path.opt_cost, 10953216u);
}

TEST(einsum, path_greedy_on_ten_operand_chain) {
    context ctx{};
    tensor head{ctx, dtype::float32, 4, 64};
    tensor mid[8] = {
        tensor{ctx, dtype::float32, 64, 64}, tensor{ctx, dtype::float32, 64, 64}, tensor{ctx, dtype::float32, 64, 64}, tensor{ctx, dtype::float32, 64, 64},
        tensor{ctx, dtype::float32, 64, 64}, tensor{ctx, dtype::float32, 64, 64}, tensor{ctx, dtype::float32, 64, 64}, tensor{ctx, dtype::float32, 64, 64},
    };
    tensor tail{ctx, dtype::float32, 64, 4};
    mag_einsum_path_t path = einsum_path("ab,bc,cd,de,ef,fg,gh,hi,ij,jk->ak", {&head, &mid[0], &mid[1], &mid[2], &mid[3], &mid[4], &mid[5], &mid[6], &mid[7], &tail});
    ASSERT_EQ(path.num_steps, 9);
    ASSERT_EQ(path.opt_cost, 264192u);
}

TEST(einsum, three_operand_contraction_values) {
    context ctx{};
    tensor a{ctx, dtype::float32, 2, 3, 3};
    tensor b{ctx, dtype::float32, 3, 4};
    tensor c{ctx, dtype::float32, 3, 4};
    fill_iota(a, 1.0f);
    fill_iota(b, 0.5f);
    fill_iota(c, -2.0f);
    tensor y = einsum_eval("ijk,jl,kl->il", {&a, &b, &c});
    ASSERT_EQ(y.rank(), 2);
    ASSERT_EQ(y.shape()[0], 2);
    ASSERT_EQ(y.shape()[1], 4);
    auto av = a.to_vector<float>();
    auto bv = b.to_vector<float>();
    auto cv = c.to_vector<float>();
    auto got = y.to_vector<float>();
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t l = 0; l < 4; ++l) {
            double ref = 0.0;
            for (int64_t j = 0; j < 3; ++j)
                for (int64_t k = 0; k < 3; ++k)
                    ref += static_cast<double>(av[(i*3 + j)*3 + k]) * bv[j*4 + l] * cv[k*4 + l];
            ASSERT_NEAR(got[i*4 + l], static_cast<float>(ref), 1e-3f * std::abs(static_cast<float>(ref)) + 1e-3f);
        }
    }
}
