// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace magnetron::test;

TEST(core_tensor_logic, ref_count_raii) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    ASSERT_EQ(a.refcount(), 1);
    {
        tensor b {a};
        ASSERT_EQ(a.refcount(), 2);
        ASSERT_EQ(b.refcount(), 2);
        {
            tensor c {b};
            ASSERT_EQ(a.refcount(), 3);
            ASSERT_EQ(b.refcount(), 3);
            ASSERT_EQ(c.refcount(), 3);
        }
        ASSERT_EQ(a.refcount(), 2);
        ASSERT_EQ(b.refcount(), 2);
    }
    ASSERT_EQ(a.refcount(), 1);
}

TEST(core_tensor_logic, ref_count_assign) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    ASSERT_EQ(a.refcount(), 1);
    {
        tensor b = a;
        ASSERT_EQ(a.refcount(), 2);
        ASSERT_EQ(b.refcount(), 2);
        {
            tensor c = b;
            ASSERT_EQ(a.refcount(), 3);
            ASSERT_EQ(b.refcount(), 3);
            ASSERT_EQ(c.refcount(), 3);
        }
        ASSERT_EQ(a.refcount(), 2);
        ASSERT_EQ(b.refcount(), 2);
    }
    ASSERT_EQ(a.refcount(), 1);
}

TEST(core_tensor_logic, ref_count_clone) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    ASSERT_EQ(a.refcount(), 1);
    {
        tensor b = a.clone();
        ASSERT_EQ(a.refcount(), 2);
        ASSERT_EQ(b.refcount(), 1);
        {
            tensor c = b.clone();
            ASSERT_EQ(a.refcount(), 2);
            ASSERT_EQ(b.refcount(), 2);
            ASSERT_EQ(c.refcount(), 1);
        }
        ASSERT_EQ(a.refcount(), 2);
        ASSERT_EQ(b.refcount(), 1);
    }
    ASSERT_EQ(a.refcount(), 1);
}

TEST(core_tensor_logic, ref_count_move_constructor) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    auto original_ref {a.refcount()};
    tensor b {std::move(a)};
    ASSERT_EQ(b.refcount(), original_ref);
}

TEST(core_tensor_logic, ref_count_self_assignment) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    size_t original_ref = a.refcount();
    a = a;
    ASSERT_EQ(a.refcount(), original_ref);
}

TEST(core_tensor_logic, ref_count_reassign_tensor) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    {
        tensor b = a;
        ASSERT_EQ(a.refcount(), 2);
        a = tensor(ctx, dtype::e8m23, 30);
        ASSERT_EQ(a.refcount(), 1);
        ASSERT_EQ(b.refcount(), 1);
    }
}

TEST(core_tensor_logic, init_1d) {
    context ctx {compute_device::cpu};
    tensor t {ctx, dtype::e8m23, 10};
    ASSERT_EQ(t.dtype(), dtype::e8m23);
    ASSERT_EQ(t.rank(), 1);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.strides()[0], 1);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.data_size(), 10 * sizeof(e8m23_t));
    ASSERT_EQ(t.numel(), 10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(e8m23_t));
    ASSERT_EQ(t.refcount(), 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->host, nullptr);
    ASSERT_NE(internal->storage->broadcast, nullptr);
    ASSERT_NE(internal->storage->transfer, nullptr);

    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_2d) {
    context ctx {compute_device::cpu};
    tensor t {ctx, dtype::e8m23, 10, 10};
    ASSERT_EQ(t.dtype(), dtype::e8m23);
    ASSERT_EQ(t.rank(), 2);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.shape()[1], 10);
    ASSERT_EQ(t.strides()[0], 10);
    ASSERT_EQ(t.strides()[1], 1);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.numel(), 10*10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(e8m23_t));
    ASSERT_EQ(t.data_size(), 10*10 * sizeof(e8m23_t));
    ASSERT_EQ(t.refcount(), 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->host, nullptr);
    ASSERT_NE(internal->storage->broadcast, nullptr);
    ASSERT_NE(internal->storage->transfer, nullptr);

    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_3d) {
    context ctx {compute_device::cpu};
    tensor t {ctx, dtype::e8m23, 10, 10, 10};
    ASSERT_EQ(t.dtype(), dtype::e8m23);
    ASSERT_EQ(t.rank(), 3);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.shape()[1], 10);
    ASSERT_EQ(t.shape()[2], 10);
    ASSERT_EQ(t.strides()[0], 100);
    ASSERT_EQ(t.strides()[1], 10);
    ASSERT_EQ(t.strides()[2], 1);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.data_size(), 10*10*10 * sizeof(e8m23_t));
    ASSERT_EQ(t.numel(), 10*10*10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(e8m23_t));
    ASSERT_EQ(t.refcount(), 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->host, nullptr);
    ASSERT_NE(internal->storage->broadcast, nullptr);
    ASSERT_NE(internal->storage->transfer, nullptr);
    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_4d) {
    context ctx {compute_device::cpu};
    tensor t {ctx, dtype::e8m23, 10, 10, 10, 10};
    ASSERT_EQ(t.dtype(), dtype::e8m23);
    ASSERT_EQ(t.rank(), 4);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.shape()[1], 10);
    ASSERT_EQ(t.shape()[2], 10);
    ASSERT_EQ(t.shape()[3], 10);
    ASSERT_EQ(t.strides()[0], 1000);
    ASSERT_EQ(t.strides()[1], 100);
    ASSERT_EQ(t.strides()[2], 10);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.data_size(), 10*10*10*10 * sizeof(e8m23_t));
    ASSERT_EQ(t.numel(), 10*10*10*10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(e8m23_t));
    ASSERT_EQ(t.refcount(), 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->host, nullptr);
    ASSERT_NE(internal->storage->broadcast, nullptr);
    ASSERT_NE(internal->storage->transfer, nullptr);

    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_5d) {
    context ctx {compute_device::cpu};
    tensor t {ctx, dtype::e8m23, 10, 10, 10, 10, 10};
    ASSERT_EQ(t.dtype(), dtype::e8m23);
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
    ASSERT_EQ(t.data_size(), 10*10*10*10*10 * sizeof(e8m23_t));
    ASSERT_EQ(t.numel(), 10*10*10*10*10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(e8m23_t));
    ASSERT_EQ(t.refcount(), 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->host, nullptr);
    ASSERT_NE(internal->storage->broadcast, nullptr);
    ASSERT_NE(internal->storage->transfer, nullptr);

    std::cout << t.to_string() << std::endl;
}

TEST(core_tensor_logic, init_6d) {
    context ctx {compute_device::cpu};
    tensor t {ctx, dtype::e8m23, 10, 10, 10, 10, 10, 10};
    ASSERT_EQ(t.dtype(), dtype::e8m23);
    ASSERT_EQ(t.rank(), 6);
    ASSERT_EQ(t.shape()[0], 10);
    ASSERT_EQ(t.shape()[1], 10);
    ASSERT_EQ(t.shape()[2], 10);
    ASSERT_EQ(t.shape()[3], 10);
    ASSERT_EQ(t.shape()[4], 10);
    ASSERT_EQ(t.shape()[5], 10);
    ASSERT_EQ(t.strides()[0], 100000);
    ASSERT_EQ(t.strides()[1], 10000);
    ASSERT_EQ(t.strides()[2], 1000);
    ASSERT_EQ(t.strides()[3], 100);
    ASSERT_EQ(t.strides()[4], 10);
    ASSERT_EQ(t.strides()[5], 1);
    ASSERT_NE(t.data_ptr(), nullptr);
    ASSERT_EQ(t.numel(), 10*10*10*10*10*10);
    ASSERT_EQ(t.data_size(), t.numel() * sizeof(e8m23_t));
    ASSERT_EQ(t.data_size(), 10*10*10*10*10*10 * sizeof(e8m23_t));
    ASSERT_EQ(t.refcount(), 1);

    // now check some internal data
    mag_tensor_t* internal {&*t};
    ASSERT_NE(internal->storage->alignment, 0);
    ASSERT_NE(internal->storage->base, 0);
    ASSERT_NE(internal->storage->size, 0);
    ASSERT_NE(internal->storage->host, nullptr);
    ASSERT_NE(internal->storage->broadcast, nullptr);
    ASSERT_NE(internal->storage->transfer, nullptr);

    auto str = t.to_string();
    std::cout << str << std::endl;
}
