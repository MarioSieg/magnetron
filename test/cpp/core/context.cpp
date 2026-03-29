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

#include <core/mag_bfloat16.h>
#include <core/mag_float16.h>

using namespace magnetron;

TEST(context, create_cpu) {
    mag_set_log_level(MAG_LOG_LEVEL_DEBUG);
    context ctx {};
    ASSERT_TRUE(ctx.is_recording_gradients());
    ctx.start_grad_recorder();
    ctx.stop_grad_recorder();

    // crate a tensor
    tensor t {ctx, dtype::bfloat16, 4, 8, 4, 3};
    std::cout << t.to_string() << std::endl;
}
