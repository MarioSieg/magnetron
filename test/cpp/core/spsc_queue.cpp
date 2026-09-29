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

#include <core/mag_spsc_queue.h>

#include <thread>
#include <vector>

TEST(spsc_queue, init_capacity) {
    mag_spsc_queue_t q {};
    ASSERT_TRUE(mag_spsc_queue_init(&q, 0, sizeof(int), alignof(int)));
    ASSERT_EQ(mag_spsc_queue_capacity(&q), 1u);
    ASSERT_TRUE(mag_spsc_queue_empty(&q));
    ASSERT_EQ(mag_spsc_queue_size(&q), 0u);
    ASSERT_EQ(mag_spsc_queue_front(&q), nullptr);
    mag_spsc_queue_destroy(&q);
    ASSERT_TRUE(mag_spsc_queue_init(&q, 16, sizeof(int), alignof(int)));
    ASSERT_EQ(mag_spsc_queue_capacity(&q), 16u);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(q.slots) % MAG_DESTRUCTIVE_INTERFERENCE_SIZE, 0u);
    mag_spsc_queue_destroy(&q);
}

TEST(spsc_queue, push_pop_single_thread) {
    mag_spsc_queue_t q {};
    ASSERT_TRUE(mag_spsc_queue_init(&q, 4, sizeof(int), alignof(int)));
    for (int i {0}; i < 4; ++i) {
        ASSERT_TRUE(mag_spsc_queue_try_push(&q, &i));
        ASSERT_EQ(mag_spsc_queue_size(&q), static_cast<size_t>(i+1));
    }
    int extra {99};
    ASSERT_FALSE(mag_spsc_queue_try_push(&q, &extra));
    ASSERT_EQ(mag_spsc_queue_try_reserve(&q), nullptr);
    ASSERT_FALSE(mag_spsc_queue_empty(&q));
    for (int i {0}; i < 4; ++i) {
        int* f {static_cast<int*>(mag_spsc_queue_front(&q))};
        ASSERT_NE(f, nullptr);
        ASSERT_EQ(*f, i);
        mag_spsc_queue_pop(&q);
    }
    ASSERT_TRUE(mag_spsc_queue_empty(&q));
    ASSERT_EQ(mag_spsc_queue_front(&q), nullptr);
    int out {-1};
    ASSERT_FALSE(mag_spsc_queue_try_pop(&q, &out));
    ASSERT_TRUE(mag_spsc_queue_try_push(&q, &extra));
    ASSERT_TRUE(mag_spsc_queue_try_pop(&q, &out));
    ASSERT_EQ(out, 99);
    mag_spsc_queue_destroy(&q);
}

TEST(spsc_queue, reserve_commit_wraparound) {
    mag_spsc_queue_t q {};
    ASSERT_TRUE(mag_spsc_queue_init(&q, 3, sizeof(uint64_t), alignof(uint64_t)));
    for (uint64_t i {0}; i < 1000; ++i) {
        auto* slot {static_cast<uint64_t*>(mag_spsc_queue_reserve(&q))};
        ASSERT_NE(slot, nullptr);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(slot) % alignof(uint64_t), 0u);
        *slot = i*7;
        mag_spsc_queue_commit(&q);
        ASSERT_EQ(mag_spsc_queue_size(&q), 1u);
        auto* f {static_cast<uint64_t*>(mag_spsc_queue_front(&q))};
        ASSERT_EQ(*f, i*7);
        mag_spsc_queue_pop(&q);
    }
    mag_spsc_queue_destroy(&q);
}

TEST(spsc_queue, large_elements) {
    struct big { uint8_t bytes[200]; };
    mag_spsc_queue_t q {};
    ASSERT_TRUE(mag_spsc_queue_init(&q, 2, sizeof(big), alignof(big)));
    big a {};
    for (size_t i {0}; i < sizeof(a.bytes); ++i) a.bytes[i] = static_cast<uint8_t>(i);
    ASSERT_TRUE(mag_spsc_queue_try_push(&q, &a));
    big b {};
    ASSERT_TRUE(mag_spsc_queue_try_pop(&q, &b));
    ASSERT_EQ(std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)), 0);
    mag_spsc_queue_destroy(&q);
}

TEST(spsc_queue, producer_consumer) {
    constexpr uint64_t k_n {1'000'000};
    mag_spsc_queue_t q {};
    ASSERT_TRUE(mag_spsc_queue_init(&q, 64, sizeof(uint64_t), alignof(uint64_t)));
    std::thread producer {[&] {
        for (uint64_t i {0}; i < k_n; ++i) mag_spsc_queue_push(&q, &i);
    }};
    uint64_t sum {0};
    uint64_t expect {0};
    bool ordered {true};
    for (uint64_t received {0}; received < k_n;) {
        uint64_t* f {static_cast<uint64_t*>(mag_spsc_queue_front(&q))};
        if (!f) { mag_cpu_pause(); continue; }
        if (*f != expect) ordered = false;
        sum += *f;
        ++expect;
        mag_spsc_queue_pop(&q);
        ++received;
    }
    producer.join();
    ASSERT_TRUE(ordered);
    ASSERT_EQ(sum, k_n*(k_n-1)/2);
    ASSERT_TRUE(mag_spsc_queue_empty(&q));
    mag_spsc_queue_destroy(&q);
}

TEST(spsc_queue, producer_consumer_try_ops) {
    constexpr uint32_t k_n {200'000};
    mag_spsc_queue_t q {};
    ASSERT_TRUE(mag_spsc_queue_init(&q, 8, sizeof(uint32_t), alignof(uint32_t)));
    std::thread producer {[&] {
        for (uint32_t i {0}; i < k_n; ++i) {
            while (!mag_spsc_queue_try_push(&q, &i)) mag_cpu_pause();
        }
    }};
    std::vector<uint32_t> got {};
    got.reserve(k_n);
    while (got.size() < k_n) {
        uint32_t v {};
        if (mag_spsc_queue_try_pop(&q, &v)) got.push_back(v);
        else mag_cpu_pause();
    }
    producer.join();
    for (uint32_t i {0}; i < k_n; ++i) ASSERT_EQ(got[i], i);
    mag_spsc_queue_destroy(&q);
}
