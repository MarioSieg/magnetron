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

#include <atomic>
#include <thread>
#include <mutex>
#include <optional>
#include <vector>

using namespace magnetron;

namespace {
    constexpr int64_t k_parallel_rows {256};
    constexpr int64_t k_parallel_cols {256};

    auto worker_count() -> unsigned {
        unsigned hw {std::thread::hardware_concurrency()};
        return std::max(4u, std::min(2u*(hw ? hw : 4u), 32u));
    }
}

TEST(threading, allocator_storm) {
    context ctx {};
    const unsigned threads {worker_count()};
    constexpr int iterations {2000};
    std::atomic<int> failures {0};

    std::vector<std::thread> pool {};
    pool.reserve(threads);
    for (unsigned t {0}; t < threads; ++t) {
        pool.emplace_back([&ctx, &failures] {
            for (int i {0}; i < iterations; ++i) {
                tensor x {ctx, dtype::float32, 32, 32};
                tensor y {x.view(x.shape())};
                tensor z {y.abs()};
                if (z.numel() != 32*32) ++failures;
            }
        });
    }
    for (auto &th : pool) th.join();
    ASSERT_EQ(0, failures.load());
}

TEST(threading, cross_thread_ownership) {
    context ctx {};
    const unsigned threads {worker_count()};
    constexpr int per_thread {512};

    std::vector<std::optional<tensor>> handoff {};
    std::mutex handoff_mtx {};

    std::vector<std::thread> producers {};
    producers.reserve(threads);
    for (unsigned t {0}; t < threads; ++t) {
        producers.emplace_back([&] {
            std::vector<tensor> local {};
            local.reserve(per_thread);
            for (int i {0}; i < per_thread; ++i)
                local.emplace_back(ctx, dtype::float32, 16, 16);
            std::scoped_lock lock {handoff_mtx};
            for (auto &tn : local) handoff.emplace_back(tn);
        });
    }
    for (auto &th : producers) th.join();
    ASSERT_EQ(static_cast<size_t>(threads)*per_thread, handoff.size());

    std::vector<std::thread> consumers {};
    consumers.reserve(threads);
    const size_t chunk {handoff.size()/threads};
    for (unsigned t {0}; t < threads; ++t) {
        size_t begin {t*chunk};
        size_t end {t+1 == threads ? handoff.size() : begin+chunk};
        consumers.emplace_back([&handoff, begin, end] {
            for (size_t i {begin}; i < end; ++i) handoff[i].reset();
        });
    }
    for (auto &th : consumers) th.join();
}

TEST(threading, concurrent_intra_op_parallel_submits) {
    context ctx {};
    const unsigned threads {worker_count()};
    constexpr int iterations {24};
    std::atomic<int> failures {0};

    std::vector<std::thread> pool {};
    pool.reserve(threads);
    for (unsigned t {0}; t < threads; ++t) {
        pool.emplace_back([&ctx, &failures] {
            for (int i {0}; i < iterations; ++i) {
                tensor x {ctx, dtype::float32, k_parallel_rows, k_parallel_cols};
                x.fill_(-2.0f);
                tensor y {x.abs()};
                std::vector<float> host {y.to_vector<float>()};
                for (float v : host)
                    if (std::abs(v - 2.0f) > 1e-5f) ++failures;
            }
        });
    }
    for (auto &th : pool) th.join();
    ASSERT_EQ(0, failures.load());
}

TEST(threading, concurrent_backward_disjoint_graphs) {
    context ctx {};
    const unsigned threads {worker_count()};
    std::atomic<int> failures {0};

    std::vector<std::thread> pool {};
    pool.reserve(threads);
    for (unsigned t {0}; t < threads; ++t) {
        pool.emplace_back([&ctx, &failures, t] {
            for (int i {0}; i < 32; ++i) {
                tensor x {ctx, dtype::float32, 8};
                x.fill_(static_cast<float>(t+1));
                x.requires_grad(true);
                tensor loss {x.mul(x).sum()};
                loss.backward();
                std::vector<float> grad {x.grad()->to_vector<float>()};
                for (float g : grad)
                    if (std::abs(g - 2.0f*static_cast<float>(t+1)) > 1e-3f) ++failures;
            }
        });
    }
    for (auto &th : pool) th.join();
    ASSERT_EQ(0, failures.load());
}

TEST(threading, concurrent_backward_shared_graph_is_rejected) {
    context ctx {};
    const unsigned threads {worker_count()};
    constexpr int rounds {64};

    tensor x {ctx, dtype::float32, 4096};
    x.fill_(2.0f);
    x.requires_grad(true);

    std::atomic<int> rejected {0};
    std::atomic<int> succeeded {0};
    std::atomic<int> other_errors {0};

    std::vector<std::thread> pool {};
    pool.reserve(threads);
    for (unsigned t {0}; t < threads; ++t) {
        pool.emplace_back([&] {
            for (int i {0}; i < rounds; ++i) {
                tensor loss {x.mul(x).sum()};
                mag_error_t err {};
                mag_status_t stat {mag_tensor_backward(&err, &*loss)};
                if (stat == MAG_OK) ++succeeded;
                else if (stat == MAG_ERR_AUTOGRAD) ++rejected;
                else ++other_errors;
            }
        });
    }
    for (auto &th : pool) th.join();

    ASSERT_EQ(0, other_errors.load());
    ASSERT_GT(succeeded.load(), 0);
    ASSERT_GT(rejected.load(), 0);
    ASSERT_EQ(static_cast<int>(threads)*rounds, succeeded.load()+rejected.load());
}

TEST(threading, serialized_backward_shared_graph) {
    context ctx {};
    const unsigned threads {worker_count()};
    std::mutex backward_mtx {};

    tensor x {ctx, dtype::float32, 8};
    x.fill_(2.0f);
    x.requires_grad(true);

    std::vector<std::thread> pool {};
    pool.reserve(threads);
    for (unsigned t {0}; t < threads; ++t) {
        pool.emplace_back([&] {
            std::scoped_lock lock {backward_mtx};
            tensor loss {x.mul(x).sum()};
            loss.backward();
        });
    }
    for (auto &th : pool) th.join();

    std::vector<float> grad {x.grad()->to_vector<float>()};
    const float expected {4.0f*static_cast<float>(threads)};
    for (float g : grad)
        ASSERT_NEAR(expected, g, 1e-3f);
}

TEST(threading, telemetry_is_exact_after_storm) {
    context ctx {};
    mag_context_t *raw {&*ctx};
    const int64_t alive_tensors_before {mag_atomic64_load(&raw->telemetry.num_alive_tensors, MAG_MO_RELAXED)};
    const int64_t alive_storages_before {mag_atomic64_load(&raw->telemetry.num_alive_storages, MAG_MO_RELAXED)};
    const int64_t created_before {mag_atomic64_load(&raw->telemetry.num_created_tensors, MAG_MO_RELAXED)};

    const unsigned threads {worker_count()};
    constexpr int iterations {250};

    std::vector<std::thread> pool {};
    pool.reserve(threads);
    for (unsigned t {0}; t < threads; ++t) {
        pool.emplace_back([&ctx] {
            for (int i {0}; i < iterations; ++i) {
                tensor x {ctx, dtype::float32, 16, 16};
            }
        });
    }
    for (auto &th : pool) th.join();

    ASSERT_EQ(alive_tensors_before, mag_atomic64_load(&raw->telemetry.num_alive_tensors, MAG_MO_RELAXED));
    ASSERT_EQ(alive_storages_before, mag_atomic64_load(&raw->telemetry.num_alive_storages, MAG_MO_RELAXED));
    ASSERT_EQ(created_before + static_cast<int64_t>(threads)*iterations,
              mag_atomic64_load(&raw->telemetry.num_created_tensors, MAG_MO_RELAXED));
}
