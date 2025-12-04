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

TEST(tensor, load_image_no_resize_planar) {
    context ctx {};
    mag_tensor_t *img;
    mag_status_t stat = mag_load_image(&img, &*ctx, "media/xoxja.png", "RGB", 0, 0);
    ASSERT_EQ(stat, MAG_STATUS_OK);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[0], 3);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[1], 512);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[2], 512);
    mag_rc_decref(img);
}

TEST(tensor, load_image_resize_planar) {
    context ctx {};
    mag_tensor_t *img;
    mag_status_t stat = mag_load_image(&img, &*ctx, "media/xoxja.png", "RGB", 22, 111);
    ASSERT_EQ(stat, MAG_STATUS_OK);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[0], 3);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[1], 111);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[2], 22);
    mag_rc_decref(img);
}
