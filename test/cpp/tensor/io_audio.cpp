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

TEST(io, audio_load) {
    context ctx {};
    mag_tensor_t *song;
    uint32_t sample_rate = 0;
    mag_status_t stat = mag_load_audio(nullptr, &song, &*ctx, "media/cat_doggo_song.mp3", &sample_rate, mag_device(CPU, 0));
    ASSERT_NE(song, nullptr);
    ASSERT_EQ(stat, MAG_OK);
    ASSERT_EQ(mag_tensor_rank(song), 2);
    ASSERT_EQ(sample_rate, 44100);
    ASSERT_EQ(mag_tensor_shape_ptr(song)[0], 2);
    ASSERT_EQ(mag_tensor_shape_ptr(song)[1], 653184);
    mag_rc_decref(song);
}

TEST(io, audio_save_load_roundtrip) {
    context ctx {};
    uint32_t sample_rate = 0;
    mag_tensor_t *song;
    mag_status_t stat = mag_load_audio(nullptr, &song, &*ctx, "media/cat_doggo_song.mp3", &sample_rate, mag_device(CPU, 0));
    ASSERT_NE(song, nullptr);
    ASSERT_EQ(stat, MAG_OK);
    ASSERT_EQ(mag_tensor_rank(song), 2);
    ASSERT_EQ(sample_rate, 44100);
    ASSERT_EQ(mag_tensor_shape_ptr(song)[0], 2);
    ASSERT_EQ(mag_tensor_shape_ptr(song)[1], 653184);
    stat = mag_save_audio(nullptr, song, "tmp.wav", sample_rate);
    ASSERT_EQ(stat, MAG_OK);
    mag_tensor_t *song2;
    stat = mag_load_audio(nullptr, &song2, &*ctx, "tmp.wav", &sample_rate, mag_device(CPU, 0));
    ASSERT_NE(song2, nullptr);
    ASSERT_EQ(stat, MAG_OK);
    ASSERT_EQ(mag_tensor_rank(song2), 2);
    ASSERT_EQ(sample_rate, 44100);
    ASSERT_EQ(mag_tensor_shape_ptr(song2)[0], 2);
    ASSERT_EQ(mag_tensor_shape_ptr(song2)[1], 653184);
    ASSERT_EQ(mag_tensor_numbytes(song), mag_tensor_numbytes(song2));
    ASSERT_EQ(std::memcmp(reinterpret_cast<void *>(mag_tensor_data_ptr(song)), reinterpret_cast<void *>(mag_tensor_data_ptr(song2)), mag_tensor_numbytes(song)), 0);
    mag_rc_decref(song2);
    mag_rc_decref(song);

    if (std::filesystem::exists("tmp.wav"))
        std::filesystem::remove("tmp.wav");
}
