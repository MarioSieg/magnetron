// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;

TEST(context, create_cpu) {
    enable_logging(true);
    context ctx {compute_device::cpu};
    ASSERT_EQ(ctx.device_type(), compute_device::cpu);
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
    context ctx {compute_device::gpu_cuda};
    ASSERT_EQ(ctx.device_type(), compute_device::gpu_cuda);
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

#endif
