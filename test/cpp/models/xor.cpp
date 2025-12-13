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
#include <nn.hpp>

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

TEST(models, xor_float32) {
    context ctx {};
    ctx.manual_seed(0x9032002);
    xor_model model{ctx, dtype::float32};
    nn::sgd optimizer{model.params(), 0.1f};

    std::vector<float> x_data {
        0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 1.0f,1.0f
    };
    std::vector<float> y_data {
        0.0f, 1.0f, 1.0f, 0.0f
    };

    tensor x {ctx, dtype::float32, 4, 2};
    x.copy_(x_data);

    tensor y {ctx, dtype::float32, 4, 1};
    y.copy_(y_data);

    constexpr int64_t epochs {2000};
    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        tensor y_hat {model(x)};
        tensor loss {nn::optimizer::mse(y_hat, y)};
        loss.backward();
        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss.to_vector<float>()[0] << std::endl;
        }
        optimizer.step();
        optimizer.zero_grad();
    }

    tensor y_hat {model(x)};

    std::vector<float> output {y_hat.round().to_vector<float>()};
    ASSERT_EQ(y_data.size(), output.size());
    for (int64_t i = 0; i < output.size(); ++i) {
        ASSERT_EQ(y_data[i], output[i]);
    }
}

#if 0 // TODO: float16 matmul
TEST(models, xor_float16) {
    context ctx {};
    ctx.manual_seed(0x9032002);
    xor_model model{ctx, dtype::float16};
    nn::sgd optimizer{model.params(), 0.1f};

    std::vector<float> x_data {
        0.0f,0.0f, 0.0f,1.0f, 1.0f,0.0f, 1.0f,1.0f
    };
    std::vector<float> y_data {
        0.0f, 1.0f, 1.0f, 0.0f
    };

    tensor x {ctx, dtype::float16, 4, 2};
    x.copy_(x_data);

    tensor y {ctx, dtype::float16, 4, 1};
    y.copy_(y_data);

    constexpr int64_t epochs {2000};
    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        tensor y_hat {model(x)};
        tensor loss {nn::optimizer::mse(y_hat, y)};
        loss.backward();
        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << static_cast<float>(loss.to_vector<float16>()[0]) << std::endl;
        }
        optimizer.step();
        optimizer.zero_grad();
    }

    tensor y_hat {model(x)};

    std::vector output {y_hat.round().to_vector<float16>()};
    ASSERT_EQ(y_data.size(), output.size());
    for (int64_t i = 0; i < output.size(); ++i) {
        ASSERT_EQ(y_data[i], output[i]);
    }
}
#endif
