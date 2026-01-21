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

#include <filesystem>
#include <prelude.hpp>

#include <core/mag_snapshot.h>

using namespace magnetron;

TEST(snapshot, metadata) {
    context ctx {};
    mag_snapshot_t *snap = mag_snapshot_new(&*ctx);

    tensor test {ctx, dtype::bfloat16, 64, 64};
    test.uniform_(0.0f, 1.0f);

    tensor test2 {ctx, dtype::i8, 9,9,9,9};
    test2.uniform_(-128, 127);

    ASSERT_TRUE(mag_snapshot_put_tensor(snap, "xiMat2x2", &*test));
    ASSERT_TRUE(mag_snapshot_put_tensor(snap, "mask", &*test2));
    ASSERT_TRUE(mag_snapshot_save(snap, "snap.mag"));

    ASSERT_TRUE(std::filesystem::exists("snap.mag"));
    //ASSERT_TRUE(std::filesystem::remove("snap.mag"));

    mag_snapshot_free(snap);
}
