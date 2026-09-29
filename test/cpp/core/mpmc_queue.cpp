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

#include <core/mag_mpmc_queue.h>

#include <atomic>
#include <thread>
#include <vector>

TEST(mpmc_queue, init_layout) {
    mag_mpmc_queue_t q {};
    ASSERT_TRUE(mag_mpmc_queue_init(&q, 16, sizeof(int), alignof(int)));
    ASSERT_EQ(mag_mpmc_queue_capacity(&q), 16u);
    ASSERT_TRUE(mag_mpmc_queue_empty(&q));
    ASSERT_EQ(mag_mpmc_queue_size(&q), 0);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(q.slots)%MAG_DESTRUCTIVE_INTERFERENCE_SIZE, 0u);
    ASSERT_EQ(q.slot_stride % MAG_DESTRUCTIVE_INTERFERENCE_SIZE, 0u);
    ASSERT_EQ(q.storage_off % alignof(int), 0u);
    mag_mpmc_queue_destroy(&q);
}

TEST(mpmc_queue, push_pop_single_thread) {
    mag_mpmc_queue_t q {};
    ASSERT_TRUE(mag_mpmc_queue_init(&q, 4, sizeof(int), alignof(int)));
    for (int i {0}; i < 4; ++i) {
        ASSERT_TRUE(mag_mpmc_queue_try_push(&q, &i));
        ASSERT_EQ(mag_mpmc_queue_size(&q), i+1);
    }
    int extra {99};
    ASSERT_FALSE(mag_mpmc_queue_try_push(&q, &extra));
    ASSERT_FALSE(mag_mpmc_queue_empty(&q));
    for (int i {0}; i < 4; ++i) {
        int out {-1};
        ASSERT_TRUE(mag_mpmc_queue_try_pop(&q, &out));
        ASSERT_EQ(out, i);
    }
    int out {-1};
    ASSERT_FALSE(mag_mpmc_queue_try_pop(&q, &out));
    ASSERT_TRUE(mag_mpmc_queue_empty(&q));
    mag_mpmc_queue_push(&q, &extra);
    mag_mpmc_queue_pop(&q, &out);
    ASSERT_EQ(out, 99);
    mag_mpmc_queue_destroy(&q);
}

TEST(mpmc_queue, wraparound_turns) {
    mag_mpmc_queue_t q {};
    ASSERT_TRUE(mag_mpmc_queue_init(&q, 3, sizeof(uint64_t), alignof(uint64_t)));
    for (uint64_t i {0}; i < 1000; ++i) {
        uint64_t v {i*7};
        mag_mpmc_queue_push(&q, &v);
        uint64_t out {};
        mag_mpmc_queue_pop(&q, &out);
        ASSERT_EQ(out, v);
    }
    mag_mpmc_queue_destroy(&q);
}

TEST(mpmc_queue, large_overaligned_elements) {
    struct alignas(256) big { uint8_t bytes[300]; };
    mag_mpmc_queue_t q {};
    ASSERT_TRUE(mag_mpmc_queue_init(&q, 2, sizeof(big), alignof(big)));
    ASSERT_EQ(q.storage_off % 256, 0u);
    ASSERT_EQ(q.slot_stride % 256, 0u);
    big a {};
    for (size_t i {0}; i < sizeof(a.bytes); ++i) a.bytes[i] = static_cast<uint8_t>(i);
    ASSERT_TRUE(mag_mpmc_queue_try_push(&q, &a));
    big b {};
    ASSERT_TRUE(mag_mpmc_queue_try_pop(&q, &b));
    ASSERT_EQ(std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)), 0);
    mag_mpmc_queue_destroy(&q);
}

TEST(mpmc_queue, multi_producer_multi_consumer) {
    constexpr uint64_t k_producers {4};
    constexpr uint64_t k_consumers {4};
    constexpr uint64_t k_per_producer {100'000};
    constexpr uint64_t k_total {k_producers*k_per_producer};
    mag_mpmc_queue_t q {};
    ASSERT_TRUE(mag_mpmc_queue_init(&q, 64, sizeof(uint64_t), alignof(uint64_t)));
    std::vector<std::thread> threads {};
    for (uint64_t p {0}; p < k_producers; ++p) {
        threads.emplace_back([&, p] {
            for (uint64_t i {0}; i < k_per_producer; ++i) {
                uint64_t v {p*k_per_producer + i};
                mag_mpmc_queue_push(&q, &v);
            }
        });
    }
    std::atomic<uint64_t> sum {0};
    std::atomic<uint64_t> count {0};
    for (uint64_t c {0}; c < k_consumers; ++c) {
        threads.emplace_back([&] {
            uint64_t local_sum {0};
            uint64_t local_count {0};
            uint64_t last_per_producer[k_producers] {};
            for (auto& x : last_per_producer) x = UINT64_MAX;
            bool ordered {true};
            for (;;) {
                uint64_t claimed {count.fetch_add(1)};
                if (claimed >= k_total) break;
                uint64_t v {};
                mag_mpmc_queue_pop(&q, &v);
                uint64_t p {v / k_per_producer};
                if (last_per_producer[p] != UINT64_MAX && v <= last_per_producer[p]) ordered = false;
                last_per_producer[p] = v;
                local_sum += v;
                ++local_count;
            }
            EXPECT_TRUE(ordered);
            sum.fetch_add(local_sum);
            (void)local_count;
        });
    }
    for (auto& t : threads) t.join();
    ASSERT_EQ(sum.load(), k_total*(k_total-1)/2);
    ASSERT_TRUE(mag_mpmc_queue_empty(&q));
    mag_mpmc_queue_destroy(&q);
}

TEST(mpmc_queue, multi_producer_multi_consumer_try_ops) {
    constexpr uint32_t k_producers {3};
    constexpr uint32_t k_consumers {3};
    constexpr uint32_t k_per_producer {50'000};
    constexpr uint32_t k_total {k_producers*k_per_producer};
    mag_mpmc_queue_t q {};
    ASSERT_TRUE(mag_mpmc_queue_init(&q, 8, sizeof(uint32_t), alignof(uint32_t)));
    std::vector<std::thread> threads {};
    for (uint32_t p {0}; p < k_producers; ++p) {
        threads.emplace_back([&, p] {
            for (uint32_t i {0}; i < k_per_producer; ++i) {
                uint32_t v {p*k_per_producer + i};
                while (!mag_mpmc_queue_try_push(&q, &v)) mag_cpu_pause();
            }
        });
    }
    std::vector<std::atomic<uint8_t>> seen(k_total);
    std::atomic<uint32_t> received {0};
    for (uint32_t c {0}; c < k_consumers; ++c) {
        threads.emplace_back([&] {
            while (received.load() < k_total) {
                uint32_t v {};
                if (!mag_mpmc_queue_try_pop(&q, &v)) { mag_cpu_pause(); continue; }
                seen[v].fetch_add(1);
                received.fetch_add(1);
            }
        });
    }
    for (auto& t : threads) t.join();
    for (uint32_t i {0}; i < k_total; ++i) ASSERT_EQ(seen[i].load(), 1);
    ASSERT_TRUE(mag_mpmc_queue_empty(&q));
    mag_mpmc_queue_destroy(&q);
}
