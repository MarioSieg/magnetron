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

#include <core/mag_hash.h>

using namespace magnetron;

TEST(hash, murmur3) {
    ASSERT_EQ(mag_murmur3_128_reduced_64("hello", 5, 0), mag_murmur3_128_reduced_64("hello", 5, 0));
    ASSERT_NE(mag_murmur3_128_reduced_64("hello", 5, 1), mag_murmur3_128_reduced_64("hello", 5, 0));
    ASSERT_NE(mag_murmur3_128_reduced_64("hello", 5, 0), mag_murmur3_128_reduced_64("hella", 5, 0));
}
