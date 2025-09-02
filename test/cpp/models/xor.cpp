// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

class xor_model final : public nn::module {
public:
    explicit xor_model(context& ctx, dtype type)
        : l1{ctx, 2, 2, type}, l2{ctx, 2, 1, type} {
        register_params(l1.params());
        register_params(l2.params());
    }

    [[nodiscard]] auto operator()(tensor x) const -> tensor {
        tensor y {l1(x).tanh()};
        y = l2(y).tanh();
        return y;
    }

private:
    nn::linear_layer l1;
    nn::linear_layer l2;
};

TEST(models, xor_e8m23) {
    context ctx {compute_device::cpu};
    xor_model model{ctx, dtype::e8m23};
    nn::sgd optimizer{model.params(), 0.1f};

    static constexpr std::array<mag_e8m23_t, 2*4> x_data {
        0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 1.0f,1.0f
    };
    static constexpr std::array<mag_e8m23_t, 4> y_data {
        0.0f, 1.0f, 1.0f, 0.0f
    };

    tensor x {ctx, dtype::e8m23, 4, 2};
    x.fill_from(x_data);

    tensor y {ctx, dtype::e8m23, 4, 1};
    y.fill_from(y_data);

    constexpr std::int64_t epochs {2000};
    for (std::int64_t epoch = 0; epoch < epochs; ++epoch) {
        tensor y_hat {model(x)};
        tensor loss {nn::optimizer::mse(y_hat, y)};
        loss.backward();
        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss(0) << std::endl;
        }
        optimizer.step();
        optimizer.zero_grad();
    }

    tensor y_hat {model(x)};

    std::vector<e8m23_t> output {y_hat.round().to_float_vector()};
    ASSERT_EQ(y_data.size(), output.size());
    for (std::int64_t i = 0; i < output.size(); ++i) {
        ASSERT_EQ(y_data[i], output[i]);
    }
}

TEST(models, xor_e5m10) {
    context ctx {compute_device::cpu};
    xor_model model{ctx, dtype::e5m10};
    nn::sgd optimizer{model.params(), 0.1f};

    static constexpr std::array<mag_e8m23_t, 2*4> x_data {
        0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 1.0f,1.0f
    };
    static constexpr std::array<mag_e8m23_t, 4> y_data {
        0.0f, 1.0f, 1.0f, 0.0f
    };

    tensor x {ctx, dtype::e5m10, 4, 2};
    x.fill_from(x_data);

    tensor y {ctx, dtype::e5m10, 4, 1};
    y.fill_from(y_data);

    constexpr std::int64_t epochs {2000};
    for (std::int64_t epoch = 0; epoch < epochs; ++epoch) {
        tensor y_hat {model(x)};
        tensor loss {nn::optimizer::mse(y_hat, y)};
        loss.backward();
        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss(0) << std::endl;
        }
        optimizer.step();
        optimizer.zero_grad();
    }

    tensor y_hat {model(x)};

    std::vector<e8m23_t> output {y_hat.round().to_float_vector()};
    ASSERT_EQ(y_data.size(), output.size());
    for (std::int64_t i = 0; i < output.size(); ++i) {
        ASSERT_EQ(y_data[i], output[i]);
    }
}
