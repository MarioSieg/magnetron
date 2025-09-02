// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;

TEST(cpu_autograd, simple) {
    context ctx {compute_device::cpu};
    tensor x {ctx, dtype::e8m23, 1};
    x.fill_float(3.0f);
    x.requires_grad(true);
    tensor y {ctx, dtype::e8m23, 1};
    y.fill_float(2.0f);
    y.requires_grad(true);
    tensor k {ctx, dtype::e8m23, 1};
    k.fill_float(10.0f);
    k.requires_grad(true);

    tensor z {(x + y)*(x - y)/k};
    z.backward();

    ASSERT_TRUE(x.requires_grad());
    ASSERT_TRUE(y.requires_grad());
    ASSERT_TRUE(k.requires_grad());
    ASSERT_TRUE(z.requires_grad());
    ASSERT_TRUE(x.grad().has_value());
    ASSERT_TRUE(y.grad().has_value());
    ASSERT_TRUE(k.grad().has_value());
    ASSERT_TRUE(z.grad().has_value());

    ASSERT_EQ(x.numel(), y.numel());
    ASSERT_EQ(x.numel(), z.numel());

    // check forward pass
    for (std::int64_t i {}; i < x.numel(); ++i) {
        ASSERT_FLOAT_EQ(x(0), 3.0f);
        ASSERT_FLOAT_EQ(y(0), 2.0f);
        ASSERT_FLOAT_EQ(z(0), 0.5f);
    }

    // check backward pass
    for (std::int64_t i {}; i < x.grad().value().numel(); ++i) { // ∂z/∂x = 0.6
        ASSERT_FLOAT_EQ(x.grad().value()(i), 0.6f);
    }
    for (std::int64_t i {}; i < y.grad().value().numel(); ++i) { // ∂z/∂x = -0.4
        ASSERT_FLOAT_EQ(y.grad().value()(i), -0.4f);
    }
    for (std::int64_t i {}; i < z.grad().value().numel(); ++i) { // ∂z/∂x = 1
        ASSERT_FLOAT_EQ(z.grad().value()(i), 1.0f);
    }
}

TEST(cpu_autograd, scalar_complex) {
    context ctx {compute_device::cpu};
    tensor two {ctx, dtype::e8m23, 1};
    two.fill_float(2.0f);
    two.requires_grad(true);
    tensor x {ctx, dtype::e8m23, 1};
    x.fill_float(-4.0f);
    x.requires_grad(true);
    tensor z {two*x+two+x};
    tensor q {z.relu()+z*x};
    tensor h {(z*z).relu()};
    tensor y {h+q+q*x};
    y.backward();

    ASSERT_TRUE(two.requires_grad());
    ASSERT_TRUE(x.requires_grad());
    ASSERT_TRUE(z.requires_grad());
    ASSERT_TRUE(q.requires_grad());
    ASSERT_TRUE(h.requires_grad());
    ASSERT_TRUE(y.requires_grad());
    ASSERT_TRUE(two.grad().has_value());
    ASSERT_TRUE(x.grad().has_value());
    ASSERT_TRUE(z.grad().has_value());
    ASSERT_TRUE(q.grad().has_value());
    ASSERT_TRUE(h.grad().has_value());
    ASSERT_TRUE(y.grad().has_value());

    ASSERT_EQ(x.numel(), y.numel());

    // check forward pass
    for (std::int64_t i {}; i < x.numel(); ++i) {
        ASSERT_FLOAT_EQ(x(i), -4.0f);
        ASSERT_FLOAT_EQ(y(i), -20.0f);
    }

    // check backward pass
    for (std::int64_t i {}; i < x.grad().value().numel(); ++i) { // ∂z/∂x = 46.0
        ASSERT_FLOAT_EQ(x.grad().value()(i), 46.0f);
    }
    for (std::int64_t i {}; i < y.grad().value().numel(); ++i) { // ∂z/∂y = 1.0
        ASSERT_FLOAT_EQ(y.grad().value()(i), 1.0f);
    }
}

TEST(cpu_autograd, broadcast) {
    context ctx {compute_device::cpu};
    tensor x {ctx, dtype::e8m23, 3, 3, 3, 3};
    x.fill_float(3.0f);
    x.requires_grad(true);
    tensor y {ctx, dtype::e8m23, 3, 3, };
    y.fill_float(2.0f);
    y.requires_grad(true);
    tensor k {ctx, dtype::e8m23, 1};
    k.fill_float(10.0f);
    k.requires_grad(true);

    tensor z {((x + y)*(x - y)/k).sum()};
    z.backward();

    ASSERT_TRUE(x.requires_grad());
    ASSERT_TRUE(y.requires_grad());
    ASSERT_TRUE(k.requires_grad());
    ASSERT_TRUE(z.requires_grad());
    ASSERT_TRUE(x.grad().has_value());
    ASSERT_TRUE(y.grad().has_value());
    ASSERT_TRUE(k.grad().has_value());
    ASSERT_TRUE(z.grad().has_value());

    // check forward pass
    for (std::int64_t i {}; i < x.numel(); ++i) {
        ASSERT_FLOAT_EQ(x(0), 3.0f);
    }
    for (std::int64_t i {}; i < y.numel(); ++i) {
        ASSERT_FLOAT_EQ(y(0), 2.0f);
    }
    for (std::int64_t i {}; i < z.numel(); ++i) {
        ASSERT_FLOAT_EQ(z(0), 40.5f);
    }

    // check backward pass
    auto x_grad {x.grad().value()};
    for (std::int64_t i {}; i < x_grad.numel(); ++i) { // ∂z/∂x = 0.6
        ASSERT_FLOAT_EQ(x_grad(i), 0.6f);
    }
    auto y_grad {y.grad().value()};
    for (std::int64_t i {}; i < y_grad.numel(); ++i) { // ∂z/∂x = -3.6f
        ASSERT_FLOAT_EQ(y_grad(i), -3.6f);
    }
    auto z_grad {z.grad().value()};
    for (std::int64_t i {}; i < z_grad.numel(); ++i) { // ∂z/∂x = 1
        ASSERT_FLOAT_EQ(z_grad(i), 1.0f);
    }
}
