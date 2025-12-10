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
using namespace magnetron::test;

TEST(core_tensor_logic, ref_count_raii) {
    context ctx {};
    tensor a {ctx, dtype::float32, 10};
    
    ASSERT_EQ((*a).__rcb.rc_strong, 1);
    {
        tensor b {a};
        ASSERT_EQ((*a).__rcb.rc_strong, 2);
        ASSERT_EQ((*b).__rcb.rc_strong, 2);
        {
            tensor c {b};
            ASSERT_EQ((*a).__rcb.rc_strong, 3);
            ASSERT_EQ((*b).__rcb.rc_strong, 3);
            ASSERT_EQ((*c).__rcb.rc_strong, 3);
        }
        ASSERT_EQ((*a).__rcb.rc_strong, 2);
        ASSERT_EQ((*b).__rcb.rc_strong, 2);
    }
    ASSERT_EQ((*a).__rcb.rc_strong, 1);
}

TEST(core_tensor_logic, ref_count_assign) {
    context ctx {};
    tensor a {ctx, dtype::float32, 10};
    ASSERT_EQ((*a).__rcb.rc_strong, 1);
    {
        tensor b = a;
        ASSERT_EQ((*a).__rcb.rc_strong, 2);
        ASSERT_EQ((*b).__rcb.rc_strong, 2);
        {
            tensor c = b;
            ASSERT_EQ((*a).__rcb.rc_strong, 3);
            ASSERT_EQ((*b).__rcb.rc_strong, 3);
            ASSERT_EQ((*c).__rcb.rc_strong, 3);
        }
        ASSERT_EQ((*a).__rcb.rc_strong, 2);
        ASSERT_EQ((*b).__rcb.rc_strong, 2);
    }
    ASSERT_EQ((*a).__rcb.rc_strong, 1);
}

TEST(core_tensor_logic, ref_count_clone) {
    context ctx {};
    tensor a {ctx, dtype::float32, 10};
    ASSERT_EQ((*a).__rcb.rc_strong, 1);
    {
        tensor b = a.clone();
        ASSERT_EQ((*a).__rcb.rc_strong, 2);
        ASSERT_EQ((*b).__rcb.rc_strong, 1);
        {
            tensor c = b.clone();
            ASSERT_EQ((*a).__rcb.rc_strong, 2);
            ASSERT_EQ((*b).__rcb.rc_strong, 2);
            ASSERT_EQ((*c).__rcb.rc_strong, 1);
        }
        ASSERT_EQ((*a).__rcb.rc_strong, 2);
        ASSERT_EQ((*b).__rcb.rc_strong, 1);
    }
    ASSERT_EQ((*a).__rcb.rc_strong, 1);
}

TEST(core_tensor_logic, ref_count_move_constructor) {
    context ctx {};
    tensor a {ctx, dtype::float32, 10};
    auto original_ref {(*a).__rcb.rc_strong};
    tensor b {std::move(a)};
    ASSERT_EQ((*b).__rcb.rc_strong, original_ref);
}

TEST(core_tensor_logic, ref_count_self_assignment) {
    context ctx {};
    tensor a {ctx, dtype::float32, 10};
    size_t original_ref = (*a).__rcb.rc_strong;
    a = a;
    ASSERT_EQ((*a).__rcb.rc_strong, original_ref);
}

TEST(core_tensor_logic, ref_count_reassign_tensor) {
    context ctx {};
    tensor a {ctx, dtype::float32, 10};
    {
        tensor b = a;
        ASSERT_EQ((*a).__rcb.rc_strong, 2);
        a = tensor(ctx, dtype::float32, 30);
        ASSERT_EQ((*a).__rcb.rc_strong, 1);
        ASSERT_EQ((*b).__rcb.rc_strong, 1);
    }
}

TEST(core_tensor_logic, init_1d) {
    context ctx {};
    tensor t {ctx, dtype::float32, 10};
    ASSERT_EQ(t.dtype(), dtype::float32);
    ASSERT_EQ(t.rank(), 1);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.strides()[0], 1);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.data_size(), 10 * sizeof(float));
    ASSERT_EQ(t.numel(), 10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(float));
    ASSERT_EQ((*t).__rcb.rc_strong, 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->device, nullptr);

    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_2d) {
    context ctx {};
    tensor t {ctx, dtype::float32, 10, 10};
    ASSERT_EQ(t.dtype(), dtype::float32);
    ASSERT_EQ(t.rank(), 2);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.shape()[1], 10);
    ASSERT_EQ(t.strides()[0], 10);
    ASSERT_EQ(t.strides()[1], 1);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.numel(), 10*10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(float));
    ASSERT_EQ(t.data_size(), 10*10 * sizeof(float));
    ASSERT_EQ((*t).__rcb.rc_strong, 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->device, nullptr);

    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_3d) {
    context ctx {};
    tensor t {ctx, dtype::float32, 10, 10, 10};
    ASSERT_EQ(t.dtype(), dtype::float32);
    ASSERT_EQ(t.rank(), 3);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.shape()[1], 10);
    ASSERT_EQ(t.shape()[2], 10);
    ASSERT_EQ(t.strides()[0], 100);
    ASSERT_EQ(t.strides()[1], 10);
    ASSERT_EQ(t.strides()[2], 1);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.data_size(), 10*10*10 * sizeof(float));
    ASSERT_EQ(t.numel(), 10*10*10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(float));
    ASSERT_EQ((*t).__rcb.rc_strong, 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->device, nullptr);
    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_4d) {
    context ctx {};
    tensor t {ctx, dtype::float32, 10, 10, 10, 10};
    ASSERT_EQ(t.dtype(), dtype::float32);
    ASSERT_EQ(t.rank(), 4);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.shape()[1], 10);
    ASSERT_EQ(t.shape()[2], 10);
    ASSERT_EQ(t.shape()[3], 10);
    ASSERT_EQ(t.strides()[0], 1000);
    ASSERT_EQ(t.strides()[1], 100);
    ASSERT_EQ(t.strides()[2], 10);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.data_size(), 10*10*10*10 * sizeof(float));
    ASSERT_EQ(t.numel(), 10*10*10*10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(float));
    ASSERT_EQ((*t).__rcb.rc_strong, 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->device, nullptr);

    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_5d) {
    context ctx {};
    tensor t {ctx, dtype::float32, 10, 10, 10, 10, 10};
    ASSERT_EQ(t.dtype(), dtype::float32);
    ASSERT_EQ(t.rank(), 5);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.shape()[1], 10);
    ASSERT_EQ(t.shape()[2], 10);
    ASSERT_EQ(t.shape()[3], 10);
    ASSERT_EQ(t.shape()[4], 10);
    ASSERT_EQ(t.strides()[0], 10000);
    ASSERT_EQ(t.strides()[1], 1000);
    ASSERT_EQ(t.strides()[2], 100);
    ASSERT_EQ(t.strides()[3], 10);
    ASSERT_EQ(t.strides()[4], 1);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.data_size(), 10*10*10*10*10 * sizeof(float));
    ASSERT_EQ(t.numel(), 10*10*10*10*10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(float));
    ASSERT_EQ((*t).__rcb.rc_strong, 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->device, nullptr);

    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_6d) {
    context ctx {};
    tensor t {ctx, dtype::float32, 2, 2, 2, 2, 2, 2};
    ASSERT_EQ(t.dtype(), dtype::float32);
    ASSERT_EQ(t.rank(), 6);
    ASSERT_EQ(t.shape()[0], 2);
    ASSERT_EQ(t.shape()[1], 2);
    ASSERT_EQ(t.shape()[2], 2);
    ASSERT_EQ(t.shape()[3], 2);
    ASSERT_EQ(t.shape()[4], 2);
    ASSERT_EQ(t.shape()[5], 2);
    ASSERT_EQ(t.strides()[0], 2*2*2*2*2);
    ASSERT_EQ(t.strides()[1], 2*2*2*2);
    ASSERT_EQ(t.strides()[2], 2*2*2);
    ASSERT_EQ(t.strides()[3], 2*2);
    ASSERT_EQ(t.strides()[4], 2);
    ASSERT_EQ(t.strides()[5], 1);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.numel(), 2*2*2*2*2*2);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(float));
    ASSERT_EQ(t.data_size(), 2*2*2*2*2*2 * sizeof(float));
    ASSERT_EQ((*t).__rcb.rc_strong, 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->device, nullptr);

    auto str = t.to_string();
    std::cout << str << std::endl;
}
