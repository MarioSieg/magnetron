// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;

TEST(misc, pack_color_u8) {
    ASSERT_EQ(pack_color_u8(0, 0, 0), 0x000000);
    ASSERT_EQ(pack_color_u8(255, 255, 255), 0xffffff);
    ASSERT_EQ(pack_color_u8(255, 0, 0), 0xff0000);
    ASSERT_EQ(pack_color_u8(0, 255, 0), 0x00ff00);
    ASSERT_EQ(pack_color_u8(0, 0, 255), 0x0000ff);
}

TEST(misc, compute_device_name) {
    ASSERT_TRUE(!compute_device_name(compute_device::cpu).empty());
}

TEST(misc, enable_disable_logging) {
    enable_logging(false);
    enable_logging(true);
}

TEST(misc, hash_function) {
    ASSERT_EQ(mag_hash("hello", 5, 0), 15821672119091348640ull);
    ASSERT_EQ(mag_hash("hello", 5, 0), 15821672119091348640ull);
    ASSERT_NE(mag_hash("hello", 5, 1), 15821672119091348640ull);
    ASSERT_NE(mag_hash("helli", 5, 0), 15821672119091348640ull);
}
