// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <cstdint>
#include <vector>

#include <magnetron.hpp>

// Tiny module-like library for testing training and models
namespace magnetron::test::nn {
    // Base class for all modules
    class module {
    public:
        module(const module&) = delete;
        module(module&&) = delete;
        auto operator=(const module&) -> module& = delete;
        auto operator=(module&&) -> module& = delete;
        virtual ~module() = default;

        [[nodiscard]] auto params() noexcept -> std::span<tensor> { return m_params; }

    protected:
        module() = default;

        auto register_param(tensor param) -> void {
            param.requires_grad(true);
            m_params.emplace_back(param);
        }

        auto register_params(std::span<tensor> params) -> void {
            for (auto param : params)
                register_param(param);
        }

    private:
        std::vector<tensor> m_params {};
    };

    // Base class for all optimizers
    class optimizer {
    public:
        optimizer(const optimizer&) = delete;
        optimizer(optimizer&&) = delete;
        auto operator=(const optimizer&) -> optimizer& = delete;
        auto operator=(optimizer&&) -> optimizer& = delete;
        virtual ~optimizer() = default;

        virtual auto step() -> void = 0;

        [[nodiscard]] auto params() noexcept -> std::span<tensor> { return m_params; }
        auto set_params(std::span<tensor> params) -> void {
            m_params = params;
        }

        auto zero_grad() -> void {
            for (auto param : params()) {
               param.zero_grad();
            }
        }

        [[nodiscard]] static auto mse(tensor y_hat, tensor y) -> tensor {
            tensor delta {y_hat - y};
            return (delta*delta).mean();
        }

    protected:
        explicit optimizer(std::span<tensor> params) : m_params{params} {}

    private:
        std::span<tensor> m_params{};
    };

    // Stochastic Gradient Descent optimizer
    class sgd final : public optimizer {
    public:
        explicit sgd(std::span<tensor> params, float lr) : optimizer{params}, lr{lr} {}

        auto step() -> void override {
            for (auto& param : params()) {
                auto grad {param.grad()};
                if (!grad.has_value()) [[unlikely]] {
                    throw std::runtime_error{"Parameter has no gradient"};
                }
                tensor delta {param - *grad*lr};
                param.fill_from(delta.data_ptr(), delta.data_size());
            }
        }

        float lr {};
    };

    // Linear/Dense layer
    class linear_layer final : public module {
    public:
        linear_layer(context& ctx, int64_t in_features, int64_t out_features, dtype type = dtype::e8m23, bool has_bias = true) {
            tensor weight {ctx, type, out_features, in_features};
            weight.fill_rand_normal(0.0f, 1.0f);
            weight = weight / static_cast<float>(std::sqrt(in_features + out_features));
            register_param(weight);
            this->weight = weight;
            if (has_bias) {
                tensor bias {ctx, type, out_features};
                bias.fill(0.f);
                register_param(bias);
                this->bias = bias;
            }
        }

        [[nodiscard]] auto operator()(tensor x) const -> tensor {
            tensor y {x % weight->T()};
            if (bias)
                y = y + *bias;
            return y;
        }

        std::optional<tensor> weight {};
        std::optional<tensor> bias {};
    };
}