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

TEST(misc, hash_function) {
    ASSERT_EQ(mag_hash("hello", 5, 0), 15821672119091348640ull);
    ASSERT_EQ(mag_hash("hello", 5, 0), 15821672119091348640ull);
    ASSERT_NE(mag_hash("hello", 5, 1), 15821672119091348640ull);
    ASSERT_NE(mag_hash("helli", 5, 0), 15821672119091348640ull);
}
