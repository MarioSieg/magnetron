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
#include <filesystem>

using namespace magnetron;

TEST(io, image_load) {
    context ctx {};
    mag_tensor_t *img;
    mag_status_t stat = mag_load_image(nullptr, &img, &*ctx, "media/logo.png", "RGB", 0, 0, mag_device(CPU, 0));
    ASSERT_EQ(stat, MAG_OK);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[0], 3);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[1], 1498);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[2], 1498);
    mag_rc_decref(img);
}

TEST(io, image_load_resized) {
    context ctx {};
    mag_tensor_t *img;
    mag_status_t stat = mag_load_image(nullptr, &img, &*ctx, "media/logo.png", "RGB", 22, 111, mag_device(CPU, 0));
    ASSERT_EQ(stat, MAG_OK);
    ASSERT_EQ(mag_tensor_rank(img), 3);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[0], 3);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[1], 111);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[2], 22);
    mag_rc_decref(img);
}

TEST(io, image_save_load_roundtrip) {
    context ctx {};
    mag_tensor_t *img;
    mag_status_t stat = mag_load_image(nullptr, &img, &*ctx, "media/logo.png", "RGB", 0, 0, mag_device(CPU, 0));
    ASSERT_NE(img, nullptr);
    ASSERT_EQ(stat, MAG_OK);
    ASSERT_EQ(mag_tensor_rank(img), 3);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[0], 3);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[1], 1498);
    ASSERT_EQ(mag_tensor_shape_ptr(img)[2], 1498);
    stat = mag_save_image(nullptr, img, "tmp.png");
    ASSERT_EQ(stat, MAG_OK);
    mag_tensor_t *img2;
    stat = mag_load_image(nullptr, &img2, &*ctx, "tmp.png", "RGB", 0, 0, mag_device(CPU, 0));
    ASSERT_NE(img2, nullptr);
    ASSERT_EQ(stat, MAG_OK);
    ASSERT_EQ(mag_tensor_rank(img2), 3);
    ASSERT_EQ(mag_tensor_shape_ptr(img2)[0], 3);
    ASSERT_EQ(mag_tensor_shape_ptr(img2)[1], 1498);
    ASSERT_EQ(mag_tensor_shape_ptr(img2)[2], 1498);
    ASSERT_EQ(mag_tensor_numbytes(img), mag_tensor_numbytes(img2));
    ASSERT_EQ(std::memcmp(reinterpret_cast<void *>(mag_tensor_data_ptr(img)), reinterpret_cast<void *>(mag_tensor_data_ptr(img2)), mag_tensor_numbytes(img)), 0);
    mag_rc_decref(img2);
    mag_rc_decref(img);

    if (std::filesystem::exists("tmp.png"))
        std::filesystem::remove("tmp.png");
}
