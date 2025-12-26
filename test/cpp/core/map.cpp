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

#include <core/mag_romap.h>

#include <unordered_set>

using namespace magnetron;

TEST(romap, init_empty_get_del) {
    mag_map_t m;
    mag_map_init(&m, 8, true);

    EXPECT_EQ(map_map_get(&m, "nope", 4), nullptr);
    EXPECT_EQ(map_map_del(&m, "nope", 4), nullptr);

    mag_map_free(&m);
}

TEST(romap, put_get_roundtrip) {
    mag_map_t m;
    mag_map_init(&m, 8, true);

    void *v1 = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1111));
    void *v2 = reinterpret_cast<void *>(static_cast<uintptr_t>(0x2222));

    EXPECT_EQ(map_map_put(&m, "hello", 5, v1), v1);
    EXPECT_EQ(map_map_get(&m, "hello", 5), v1);
    EXPECT_EQ(map_map_get(&m, "hella", 5), nullptr);

    EXPECT_EQ(map_map_put(&m, "world", 5, v2), v2);
    EXPECT_EQ(map_map_get(&m, "world", 5), v2);

    mag_map_free(&m);
}

TEST(romap, put_duplicate_key_returns_old_value_and_keeps_old) {
    mag_map_t m;
    mag_map_init(&m, 8, true);

    void *v1 = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1111));
    void *v2 = reinterpret_cast<void *>(static_cast<uintptr_t>(0x2222));

    EXPECT_EQ(map_map_put(&m, "k", 1, v1), v1);
    EXPECT_EQ(map_map_get(&m, "k", 1), v1);

    EXPECT_EQ(map_map_put(&m, "k", 1, v2), v1);
    EXPECT_EQ(map_map_get(&m, "k", 1), v1);

    mag_map_free(&m);
}

TEST(romap, delete_existing_and_missing) {
    mag_map_t m;
    mag_map_init(&m, 8, true);

    void *v1 = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1111));

    EXPECT_EQ(map_map_put(&m, "a", 1, v1), v1);
    EXPECT_EQ(map_map_get(&m, "a", 1), v1);

    EXPECT_EQ(map_map_del(&m, "a", 1), v1);
    EXPECT_EQ(map_map_get(&m, "a", 1), nullptr);

    EXPECT_EQ(map_map_del(&m, "a", 1), nullptr);

    mag_map_free(&m);
}

TEST(romap, many_inserts_trigger_resize_and_all_findable) {
    mag_map_t m;
    mag_map_init(&m, 8, true);

    constexpr int N = 3000;
    std::vector<std::string> keys;
    keys.reserve(static_cast<size_t>(N));

    for (int i = 0; i < N; i++) {
        keys.emplace_back("k" + std::to_string(i));
        void *v = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1000 + i));
        EXPECT_EQ(map_map_put(&m, keys.back().data(), keys.back().size(), v), v);
    }

    for (int i = 0; i < N; i++) {
        void *v = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1000 + i));
        EXPECT_EQ(map_map_get(&m, keys[static_cast<size_t>(i)].data(), keys[static_cast<size_t>(i)].size()), v);
    }

    mag_map_free(&m);
}

TEST(romap, delete_many_preserves_others) {
    mag_map_t m;
    mag_map_init(&m, 16, true);

    constexpr int N = 2000;
    std::vector<std::string> keys;
    keys.reserve(static_cast<size_t>(N));

    for (int i = 0; i < N; i++) {
        keys.emplace_back("k" + std::to_string(i));
        void *v = reinterpret_cast<void *>(static_cast<uintptr_t>(0x2000 + i));
        EXPECT_EQ(map_map_put(&m, keys.back().data(), keys.back().size(), v), v);
    }

    for (int i = 0; i < N; i += 2) {
        void *v = reinterpret_cast<void *>(static_cast<uintptr_t>(0x2000 + i));
        EXPECT_EQ(map_map_del(&m, keys[static_cast<size_t>(i)].data(), keys[static_cast<size_t>(i)].size()), v);
    }

    for (int i = 0; i < N; i++) {
        void *expect = (i % 2 == 0)
            ? nullptr
            : reinterpret_cast<void *>(static_cast<uintptr_t>(0x2000 + i));
        EXPECT_EQ(map_map_get(&m, keys[static_cast<size_t>(i)].data(), keys[static_cast<size_t>(i)].size()), expect);
    }

    mag_map_free(&m);
}

TEST(romap, iterator_on_empty) {
    mag_map_t m;
    mag_map_init(&m, 8, true);

    size_t it = 0;
    size_t len = 0;
    void  *val = nullptr;

    EXPECT_EQ(mag_map_next(&m, &it, &len, &val), nullptr);

    mag_map_free(&m);
}

TEST(romap, iterator_visits_all_items_exactly_once_and_values_match) {
    mag_map_t m;
    mag_map_init(&m, 16, true);

    constexpr int N = 1000;
    std::vector<std::string> keys;
    keys.reserve(static_cast<size_t>(N));

    for (int i = 0; i < N; i++) {
        keys.emplace_back("k" + std::to_string(i));
        void *v = reinterpret_cast<void *>(static_cast<uintptr_t>(0x3000 + i));
        EXPECT_EQ(map_map_put(&m, keys.back().data(), keys.back().size(), v), v);
    }

    std::unordered_set<std::string> seen;
    seen.reserve(static_cast<size_t>(N));

    size_t it = 0;
    size_t len = 0;
    void  *val = nullptr;
    void  *keyp = nullptr;

    while ((keyp = mag_map_next(&m, &it, &len, &val))) {
        std::string k(static_cast<const char *>(keyp), len);
        EXPECT_TRUE(seen.insert(k).second);
        EXPECT_EQ(map_map_get(&m, k.data(), k.size()), val);
    }

    EXPECT_EQ(seen.size(), static_cast<size_t>(N));

    mag_map_free(&m);
}

TEST(romap, clone_keys_true_key_bytes_stable) {
    mag_map_t m;
    mag_map_init(&m, 8, true);

    char buf[32];
    std::memcpy(buf, "temp-key", 8);

    void *v = reinterpret_cast<void *>(static_cast<uintptr_t>(0xdeadbeef));
    EXPECT_EQ(map_map_put(&m, buf, 8, v), v);

    std::memset(buf, 'X', 8);

    EXPECT_EQ(map_map_get(&m, "temp-key", 8), v);

    size_t it = 0;
    size_t len = 0;
    void  *val = nullptr;
    void  *keyp = mag_map_next(&m, &it, &len, &val);

    ASSERT_NE(keyp, nullptr);
    EXPECT_EQ(len, static_cast<size_t>(8));
    EXPECT_EQ(std::string(static_cast<const char *>(keyp), len), std::string("temp-key", 8));

    mag_map_free(&m);
}

TEST(romap, clone_keys_false_key_points_to_source_buffer) {
    mag_map_t m;
    mag_map_init(&m, 8, false);

    char buf[32];
    std::memcpy(buf, "temp-key", 8);

    void *v = reinterpret_cast<void *>(static_cast<uintptr_t>(0xdeadbeef));
    EXPECT_EQ(map_map_put(&m, buf, 8, v), v);
    EXPECT_EQ(map_map_get(&m, buf, 8), v);

    size_t it = 0;
    size_t len = 0;
    void  *val = nullptr;
    void  *keyp = mag_map_next(&m, &it, &len, &val);

    ASSERT_NE(keyp, nullptr);
    EXPECT_EQ(keyp, static_cast<void *>(buf));
    EXPECT_EQ(len, static_cast<size_t>(8));
    EXPECT_EQ(val, v);

    std::memset(buf, 'X', 8);
    EXPECT_EQ(map_map_get(&m, "temp-key", 8), nullptr);

    mag_map_free(&m);
}
