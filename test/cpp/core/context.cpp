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

TEST(context, create_cpu) {
    enable_logging(true);
    context ctx {};
    ASSERT_TRUE(mag_device_is((*ctx).device, "cpu"));
    ASSERT_FALSE(mag_device_is((*ctx).device, "cuda"));
    ASSERT_TRUE(!ctx.cpu_name().empty());
    ASSERT_TRUE(!ctx.device_name().empty());
    ASSERT_TRUE(!ctx.os_name().empty());
    ASSERT_TRUE(!ctx.cpu_name().empty());
    ASSERT_NE(ctx.cpu_virtual_cores(), 0);
    ASSERT_NE(ctx.cpu_physical_cores(), 0);
    ASSERT_NE(ctx.cpu_sockets(), 0);
    ASSERT_NE(ctx.physical_memory_total(), 0);
    ASSERT_NE(ctx.physical_memory_free(), 0);
    ASSERT_EQ(ctx.total_tensors_created(), 0);
    ASSERT_TRUE(ctx.is_recording_gradients());
    ctx.start_grad_recorder();
    ctx.stop_grad_recorder();
    std::cout << ctx.device_name() << std::endl;
    enable_logging(false);
}

#ifdef MAG_ENABLE_CUDA

TEST(context, create_cuda) {
    enable_logging(true);

    context ctx {"cuda:0"};
    tensor y {ctx, dtype::float32, 8};
    y.uniform_(-128.f, 127.f);
    tensor a {y.cast(dtype::i8)};
    std::cout << a.to_string() << std::endl;
    std::cout << a.min().to_string() << std::endl;
    std::cout << a.sum().to_string() << std::endl;

    enable_logging(false);
}

#endif
