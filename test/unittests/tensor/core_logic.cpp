// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace magnetron::test;

TEST(core_tensor_logic, dynamic_graph_complex) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    a.fill_float(2.5f);

    tensor b {a.clone()};
    tensor c {a*b};
    tensor d {c.tanh()};

    mag_tensor_t* ta {&*a};
    ASSERT_EQ(ta->op, MAG_OP_NOP);
    ASSERT_EQ(ta->flags, 0);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(ta->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(ta->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(ta->init_op, MAG_IOP_BROADCAST);
    ASSERT_EQ(ta->init_op_params[0].type, MAG_OPP_E8M23);
    ASSERT_FLOAT_EQ(mag_op_param_unpack_e8m23_or_panic(ta->init_op_params[0]), 2.5f);

    mag_tensor_t* tb {&*b};
    ASSERT_EQ(tb->op, MAG_OP_CLONE);
    ASSERT_EQ(tb->flags, 0);
    ASSERT_EQ(tb->op_inputs[0], ta);
    ASSERT_EQ(tb->op_inputs[1], nullptr);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tb->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(tb->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tb->init_op_params[i].type, MAG_OPP_NONE);
    }

    mag_tensor_t* tc {&*c};
    ASSERT_EQ(tc->op, MAG_OP_MUL);
    ASSERT_EQ(tc->flags, 0);
    ASSERT_EQ(tc->op_inputs[0], ta);
    ASSERT_EQ(tc->op_inputs[1], tb);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tc->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(tc->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tc->init_op_params[i].type, MAG_OPP_NONE);
    }

    mag_tensor_t* td {&*d};
    ASSERT_EQ(td->op, MAG_OP_TANH);
    ASSERT_EQ(td->flags, 0);
    ASSERT_EQ(td->op_inputs[0], tc);
    ASSERT_EQ(td->op_inputs[1], nullptr);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(td->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(td->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(td->init_op_params[i].type, MAG_OPP_NONE);
    }
}

TEST(core_tensor_logic, dynamic_graph_init_op) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    a.fill_rand_uniform_float(0.0f, 1.0f);

    mag_tensor_t* ta {&*a};
    ASSERT_EQ(ta->op, MAG_OP_NOP);
    ASSERT_EQ(ta->flags, 0);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(ta->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(ta->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(ta->init_op, MAG_IOP_RAND_UNIFORM);
    ASSERT_EQ(ta->init_op_params[0].type, MAG_OPP_E8M23);
    ASSERT_FLOAT_EQ(mag_op_param_unpack_e8m23_or_panic(ta->init_op_params[0]), 0.0f);
    ASSERT_EQ(ta->init_op_params[1].type, MAG_OPP_E8M23);
    ASSERT_FLOAT_EQ(mag_op_param_unpack_e8m23_or_panic(ta->init_op_params[1]), 1.0f);
}

TEST(core_tensor_logic, dynamic_graph_binary_op) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    tensor b {ctx, dtype::e8m23, 10};
    tensor c {a + b};

    mag_tensor_t* ta {&*a};
    ASSERT_EQ(ta->op, MAG_OP_NOP);
    ASSERT_EQ(ta->flags, 0);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(ta->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(ta->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(ta->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(ta->init_op_params[i].type, MAG_OPP_NONE);
    }

    mag_tensor_t* tb {&*b};
    ASSERT_EQ(tb->op, MAG_OP_NOP);
    ASSERT_EQ(tb->flags, 0);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(tb->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tb->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(tb->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tb->init_op_params[i].type, MAG_OPP_NONE);
    }

    mag_tensor_t* tc {&*c};
    ASSERT_EQ(tc->op, MAG_OP_ADD);
    ASSERT_EQ(tc->flags, 0);
    ASSERT_EQ(tc->op_inputs[0], ta);
    ASSERT_EQ(tc->op_inputs[1], tb);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tc->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(tc->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tc->init_op_params[i].type, MAG_OPP_NONE);
    }
}

TEST(core_tensor_logic, dynamic_graph_unary_op) {
    context ctx {compute_device::cpu};
    tensor a {ctx, dtype::e8m23, 10};
    tensor b {a.neg()};

    mag_tensor_t* ta {&*a};
    ASSERT_EQ(ta->op, MAG_OP_NOP);
    ASSERT_EQ(ta->flags, 0);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(ta->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(ta->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(ta->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(ta->init_op_params[i].type, MAG_OPP_NONE);
    }

    mag_tensor_t* tb {&*b};
    ASSERT_EQ(tb->op, MAG_OP_NEG);
    ASSERT_EQ(tb->flags, 0);
    ASSERT_EQ(tb->op_inputs[0], ta);
    ASSERT_EQ(tb->op_inputs[1], nullptr);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tb->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(tb->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(tb->init_op_params[i].type, MAG_OPP_NONE);
    }
}

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
    ASSERT_EQ(internal->view_offs, 0);
    ASSERT_EQ(internal->view_uplink, nullptr);
    ASSERT_EQ(internal->op, MAG_OP_NOP);
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(internal->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->init_op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->flags, 0);
    ASSERT_EQ(internal->grad, nullptr); // gradient is allocated lazily
    ASSERT_EQ(internal->ud, nullptr);

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
    ASSERT_EQ(internal->view_offs, 0);
    ASSERT_EQ(internal->view_uplink, nullptr);
    ASSERT_EQ(internal->op, MAG_OP_NOP);
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(internal->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->init_op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->flags, 0);
    ASSERT_EQ(internal->grad, nullptr); // gradient is allocated lazily
    ASSERT_EQ(internal->ud, nullptr);

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
    ASSERT_EQ(internal->view_offs, 0);
    ASSERT_EQ(internal->view_uplink, nullptr);
    ASSERT_EQ(internal->op, MAG_OP_NOP);
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(internal->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->init_op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->flags, 0);
    ASSERT_EQ(internal->grad, nullptr); // gradient is allocated lazily
    ASSERT_EQ(internal->ud, nullptr);

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
    ASSERT_EQ(internal->view_offs, 0);
    ASSERT_EQ(internal->view_uplink, nullptr);
    ASSERT_EQ(internal->op, MAG_OP_NOP);
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(internal->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->init_op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->flags, 0);
    ASSERT_EQ(internal->grad, nullptr); // gradient is allocated lazily
    ASSERT_EQ(internal->ud, nullptr);

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
    ASSERT_EQ(internal->view_offs, 0);
    ASSERT_EQ(internal->view_uplink, nullptr);
    ASSERT_EQ(internal->op, MAG_OP_NOP);
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(internal->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->init_op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->flags, 0);
    ASSERT_EQ(internal->grad, nullptr); // gradient is allocated lazily
    ASSERT_EQ(internal->ud, nullptr);

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
    ASSERT_EQ(internal->view_offs, 0);
    ASSERT_EQ(internal->view_uplink, nullptr);
    ASSERT_EQ(internal->op, MAG_OP_NOP);
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_input_tensors; ++i) {
        ASSERT_EQ(internal->op_inputs[i], nullptr);
    }
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->init_op, MAG_IOP_NOP);
    for (std::size_t i {}; i < k_max_op_params; ++i) {
        ASSERT_EQ(internal->init_op_params[i].type, MAG_OPP_NONE);
    }
    ASSERT_EQ(internal->flags, 0);
    ASSERT_EQ(internal->grad, nullptr); // gradient is allocated lazily
    ASSERT_EQ(internal->ud, nullptr);

    auto str = t.to_string();
    std::cout << str << std::endl;
}
