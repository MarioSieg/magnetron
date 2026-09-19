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
    ctx.stop_grad_recorder();
    ASSERT_FALSE(ctx.is_recording_gradients());
    ctx.start_grad_recorder();
    ASSERT_TRUE(ctx.is_recording_gradients());

    // crate a tensor
    tensor t {ctx, dtype::bfloat16, 4, 8, 4, 3};
    std::cout << t.to_string() << std::endl;
}

TEST(context, simple_init) {
    mag_context_t *ctx = nullptr;
    assert(mag_ctx_create(nullptr, &ctx) == MAG_OK); // create context to use magnetron
    mag_tensor_t *random = nullptr;
    assert(mag_uniform(
        nullptr,
        &random,
        ctx,
        MAG_DTYPE_FLOAT8_E4M3FN, // float8 datatype
        2, // rank=2
        (int64_t[]){2, 2}, // shape=2x2 matrix
        mag_scalar_from_float64(-1.0), // sample from uniform from -1
        mag_scalar_from_float64(1.0), // to +1
        mag_device(CPU, 0) // place on device cpu:0
    ) == MAG_OK);
    mag_tensor_decref(random); // decrease refcount by 1 to free
    mag_ctx_destroy(ctx, false); // destroy context and free resources
}
