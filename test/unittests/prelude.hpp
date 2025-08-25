// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <bit>
#include <cstdint>
#include <random>

#include <magnetron/magnetron.hpp>
#include <magnetron_internal.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "prelude.hpp"

#include <half.hpp>

using namespace testing;

namespace magnetron::test {
    using e11m52_t = double;
    using e8m23_t = float;
    using e5m10_t = half_float::half;

    template <typename T>
    struct dtype_traits final {
        static constexpr T min {std::numeric_limits<T>::min()};
        static constexpr T max {std::numeric_limits<T>::min()};
        static constexpr e8m23_t eps {std::numeric_limits<T>::epsilon()};
        static inline const e8m23_t test_eps {std::numeric_limits<T>::epsilon()};
    };

    template <>
    struct dtype_traits<e5m10_t> final {
        static constexpr e5m10_t min {std::numeric_limits<e5m10_t>::min()};
        static constexpr e5m10_t max {std::numeric_limits<e5m10_t>::min()};
        static inline const e8m23_t eps {std::numeric_limits<e5m10_t>::epsilon()};
        static inline const e8m23_t test_eps {std::numeric_limits<e5m10_t>::epsilon()+0.04f}; // We increase the epsilon for f16 a little, as multiplication fails if not
    };

    [[nodiscard]] inline auto shape_as_vec(tensor t) -> std::vector<std::int64_t> {
        mag_tensor_t* internal {&*t};
        return {std::begin(internal->shape), std::end(internal->shape)};
    }

    [[nodiscard]] inline auto strides_as_vec(tensor t) -> std::vector<std::int64_t> {
        mag_tensor_t* internal {&*t};
        return {std::begin(internal->strides), std::end(internal->strides)};
    }

    [[nodiscard]] inline auto shape_to_string(std::span<const std::int64_t> shape) -> std::string {
        std::stringstream ss {};
        ss << "(";
        for (std::size_t i {}; i < shape.size(); ++i) {
            ss << shape[i];
            if (i != shape.size() - 1) {
                ss << ", ";
            }
        }
        ss << ")";
        return ss.str();
    }

    inline thread_local std::random_device rd {};
    inline thread_local std::mt19937_64 gen {rd()};

    template <typename F> requires std::is_invocable_v<F, std::span<const std::int64_t>>
    auto for_all_shape_perms(std::int64_t lim, std::int64_t fac, F&& f) -> void {
        assert(lim > 0);
        ++lim;
        std::vector<std::int64_t> shape {};
        shape.reserve(k_max_dims);
        for (std::int64_t i0 = 1; i0 < lim; ++i0) {
            for (std::int64_t i1 = 0; i1 < lim; ++i1) {
                for (std::int64_t i2 = 0; i2 < lim; ++i2) {
                    for (std::int64_t i3 = 0; i3 < lim; ++i3) {
                        for (std::int64_t i4 = 0; i4 < lim; ++i4) {
                            for (std::int64_t i5 = 0; i5 < lim; ++i5) {
                                shape.clear();
                                shape.reserve(k_max_dims);
                                if (i0 > 0) shape.emplace_back(i0*fac);
                                if (i1 > 0) shape.emplace_back(i1*fac);
                                if (i2 > 0) shape.emplace_back(i2*fac);
                                if (i3 > 0) shape.emplace_back(i3*fac);
                                if (i4 > 0) shape.emplace_back(i4*fac);
                                if (i5 > 0) shape.emplace_back(i5*fac);
                                f(std::span{shape});
                            }
                        }
                    }
                }
            }
        }
    }

    template <bool BROADCAST, bool INPLACE, typename A, typename B>
        requires std::is_invocable_r_v<tensor, A, tensor, tensor> && std::is_invocable_v<B, e8m23_t, e8m23_t>
    auto test_binary_float_operator(std::int64_t lim, e8m23_t eps, dtype ty, A&& a, B&& b, e8m23_t min = -10.0, e8m23_t max = 10.0) -> decltype(auto) {
        auto ctx = context{compute_device::cpu};
        ctx.stop_grad_recorder();
        for_all_shape_perms(lim, BROADCAST ? 2 : 1, [&](std::span<const std::int64_t> shape) {
            tensor t_a {ctx, ty, shape};
            t_a.fill_rand_uniform_float(min, max);
            tensor t_b {t_a.clone()};
            std::vector<e8m23_t> d_a {t_a.to_float_vector()};
            std::vector<e8m23_t> d_b {t_b.to_float_vector()};
            tensor t_r {std::invoke(a, t_a, t_b)};
            if constexpr (INPLACE) {
                ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
            } else {
                ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
            }
            std::vector<e8m23_t> d_r {t_r.to_float_vector()};
            ASSERT_EQ(d_a.size(), d_b.size());
            ASSERT_EQ(d_a.size(), d_r.size());
            ASSERT_EQ(t_a.dtype(), t_b.dtype());
            ASSERT_EQ(t_a.dtype(), t_r.dtype());
            for (std::int64_t i = 0; i < d_r.size(); ++i) {
                ASSERT_NEAR(std::invoke(b, d_a[i], d_b[i]), d_r[i], eps) << d_a[i] << " op " << d_b[i] << " = " << d_r[i];
            }
        });
    }

  template <bool BROADCAST, bool INPLACE, typename A, typename B>
        requires std::is_invocable_r_v<tensor, A, tensor, tensor> && std::is_invocable_v<B, std::int32_t, std::int32_t>
    auto test_binary_int_operator(std::int64_t lim, dtype ty, std::int32_t min, std::int32_t max, A&& a, B&& b) -> decltype(auto) {
        auto ctx = context{compute_device::cpu};
        ctx.stop_grad_recorder();
        for_all_shape_perms(lim, BROADCAST ? 2 : 1, [&](std::span<const std::int64_t> shape) {
            tensor t_a {ctx, ty, shape};
            std::uniform_int_distribution<std::int32_t> fill_val {min, max};
            t_a.fill_int(fill_val(gen));
            tensor t_b {t_a.clone()};
            std::vector<std::int32_t> d_a {t_a.to_int_vector()};
            std::vector<std::int32_t> d_b {t_b.to_int_vector()};
            tensor t_r {std::invoke(a, t_a, t_b)};
            if constexpr (INPLACE) {
                ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
            } else {
                ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
            }
            std::vector<std::int32_t> d_r {t_r.to_int_vector()};
            ASSERT_EQ(d_a.size(), d_b.size());
            ASSERT_EQ(d_a.size(), d_r.size());
            ASSERT_EQ(t_a.dtype(), t_b.dtype());
            ASSERT_EQ(t_a.dtype(), t_r.dtype());
            for (std::int64_t i = 0; i < d_r.size(); ++i) {
                ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " op " << d_b[i] << " = " << d_r[i];
            }
        });
    }

    template <bool BROADCAST, bool INPLACE, typename A, typename B>
        requires std::is_invocable_r_v<tensor, A, tensor, tensor> && std::is_invocable_v<B, bool, bool>
    auto test_binary_boolean_operator(std::int64_t lim, A&& a, B&& b) -> decltype(auto) {
        auto ctx = context{compute_device::cpu};
        ctx.stop_grad_recorder();
        for_all_shape_perms(lim, BROADCAST ? 2 : 1, [&](std::span<const std::int64_t> shape) {
            tensor t_a {ctx, dtype::boolean, shape};
            t_a.fill_int(true);
            tensor t_b {t_a.clone()};
            t_b.fill_int(false);
            std::vector<bool> d_a {t_a.to_bool_vector()};
            std::vector<bool> d_b {t_b.to_bool_vector()};
            tensor t_r {std::invoke(a, t_a, t_b)};
            if constexpr (INPLACE) {
                ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
            } else {
                ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
            }
            std::vector<bool> d_r {t_r.to_bool_vector()};
            ASSERT_EQ(d_a.size(), d_b.size());
            ASSERT_EQ(d_a.size(), d_r.size());
            ASSERT_EQ(t_a.dtype(), t_b.dtype());
            ASSERT_EQ(t_a.dtype(), t_r.dtype());
            for (std::int64_t i = 0; i < d_r.size(); ++i) {
                ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " op " << d_b[i] << " = " << d_r[i];
            }
        });
    }

    template <bool BROADCAST, typename A, typename B>
        requires std::is_invocable_r_v<tensor, A, tensor, tensor> && std::is_invocable_v<B, e8m23_t, e8m23_t>
    auto test_binary_float_compare(std::int64_t lim, dtype ty, A&& a, B&& b, e8m23_t min = -10.0, e8m23_t max = 10.0) -> decltype(auto) {
        auto ctx = context{compute_device::cpu};
        ctx.stop_grad_recorder();
        for_all_shape_perms(lim, BROADCAST ? 2 : 1, [&](std::span<const std::int64_t> shape){
            tensor t_a{ctx, ty, shape};
            t_a.fill_rand_uniform_float(min, max);
            tensor t_b{t_a.clone()};
            std::vector<e8m23_t> d_a {t_a.to_float_vector()};
            std::vector<e8m23_t> d_b {t_b.to_float_vector()};
            tensor t_r {std::invoke(a, t_a, t_b)};
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
            ASSERT_EQ(t_r.dtype(), dtype::boolean);
            ASSERT_EQ(d_a.size(), d_b.size());
            ASSERT_EQ(d_a.size(), t_r.numel());
            std::vector<bool> d_r {t_r.to_bool_vector()};
            for (std::int64_t i = 0; i < d_r.size(); ++i)
                ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " ? " << d_b[i] << " = " << d_r[i];
        });
    }

    template <bool BROADCAST, typename A, typename B>
        requires std::is_invocable_r_v<tensor, A, tensor, tensor> && std::is_invocable_v<B, std::int32_t, std::int32_t>
    auto test_binary_int_compare(std::int64_t lim, dtype ty, A&& a, B&& b, std::int32_t min = -10, std::int32_t max =  10) -> decltype(auto) {
        auto ctx = context{compute_device::cpu};
        ctx.stop_grad_recorder();
        std::uniform_int_distribution<std::int32_t> dist{min, max};
        for_all_shape_perms(lim, BROADCAST ? 2 : 1, [&](std::span<const std::int64_t> shape){
            tensor t_a{ctx, ty, shape};  t_a.fill_int(dist(gen));
            tensor t_b{t_a.clone()};
            std::vector<std::int32_t> d_a{t_a.to_int_vector()};
            std::vector<std::int32_t> d_b{t_b.to_int_vector()};
            tensor t_r{std::invoke(a, t_a, t_b)};
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
            ASSERT_EQ(t_r.dtype(), dtype::boolean);
            ASSERT_EQ(d_a.size(), d_b.size());
            ASSERT_EQ(d_a.size(), t_r.numel());
            std::vector<bool> d_r{t_r.to_bool_vector()};
            for (std::int64_t i = 0; i < d_r.size(); ++i)
                ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " ? " << d_b[i] << " = " << d_r[i];
        });
    }

    template <bool BROADCAST, typename A, typename B>
        requires std::is_invocable_r_v<tensor, A, tensor, tensor> && std::is_invocable_v<B, bool, bool>
    auto test_binary_bool_compare(std::int64_t lim, A&& a, B&& b) -> decltype(auto) {
        auto ctx = context{compute_device::cpu};
        ctx.stop_grad_recorder();
        for_all_shape_perms(lim, BROADCAST ? 2 : 1, [&](std::span<const std::int64_t> shape){
            tensor t_a{ctx, dtype::boolean, shape};  t_a.fill_int(true);
            tensor t_b{t_a.clone()};
            t_b.fill_int(false);
            std::vector<bool> d_a{t_a.to_bool_vector()};
            std::vector<bool> d_b{t_b.to_bool_vector()};
            tensor t_r{std::invoke(a, t_a, t_b)};
            ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
            ASSERT_EQ(t_r.dtype(), dtype::boolean);
            ASSERT_EQ(d_a.size(), d_b.size());
            ASSERT_EQ(d_a.size(), t_r.numel());
            std::vector<bool> d_r{t_r.to_bool_vector()};
            for (std::int64_t i = 0; i < d_r.size(); ++i)
                ASSERT_EQ(std::invoke(b, d_a[i], d_b[i]), d_r[i]) << d_a[i] << " ? " << d_b[i] << " = " << d_r[i];
        });
    }

    [[nodiscard]] inline auto make_random_view(tensor base) -> tensor {
        std::mt19937_64& rng {gen};
        if (base.rank() == 0) return base.view();
        bool all_one = true;
        for (auto s : base.shape())
            if (s > 1) { all_one = false; break; }
        if (all_one) return base.view();
        std::vector<std::int64_t> slicable;
        for (std::int64_t d {}; d < base.rank(); ++d)
            if (base.shape()[d] > 1) slicable.push_back(d);
        std::uniform_int_distribution<size_t> dim_dis(0, slicable.size() - 1);
        std::int64_t dim {slicable[dim_dis(rng)]};
        std::int64_t size {base.shape()[dim]};
        std::uniform_int_distribution<std::int64_t> step_dis {2, std::min<std::int64_t>(4, size)};
        std::int64_t step {step_dis(rng)};
        std::int64_t max_start {size - step};
        std::uniform_int_distribution<std::int64_t> start_dis {0, max_start};
        std::int64_t start {start_dis(rng)};
        std::int64_t max_len {(size - start + step - 1)/step};
        std::uniform_int_distribution<std::int64_t> len_dis {1, max_len};
        std::int64_t len {len_dis(rng)};
        return base.view_slice(dim, start, len, step);
    }

    template <bool BROADCAST, bool INPLACE, bool SUBVIEW, typename A, typename B>
        requires std::is_invocable_r_v<tensor, A, tensor> && std::is_invocable_v<B, e8m23_t>
    auto test_unary_operator(std::int64_t lim, e8m23_t eps, dtype ty, A&& a, B&& b, e8m23_t min = 0.0, e8m23_t max = 2.0) -> void {
        auto ctx = context{compute_device::cpu};
        for_all_shape_perms(lim, BROADCAST ? 2 : 1, [&](std::span<const std::int64_t> shape) {
            tensor base{ctx, ty, shape};
            base.fill_rand_uniform_float(min, max);
            tensor t_a = SUBVIEW ? make_random_view(base) : base;
            if constexpr (SUBVIEW)
                ASSERT_TRUE(t_a.is_view());
            std::vector<e8m23_t> d_a{t_a.to_float_vector()};
            tensor t_r = std::invoke(a, t_a);
            if constexpr (INPLACE)
                ASSERT_EQ(t_a.data_ptr(), t_r.data_ptr());
            else
                ASSERT_NE(t_a.data_ptr(), t_r.data_ptr());
            if constexpr (INPLACE)
                ASSERT_EQ(t_a.storage_base_ptr(), t_r.storage_base_ptr());
            std::vector<e8m23_t> d_r{t_r.to_float_vector()};
            ASSERT_EQ(d_a.size(), d_r.size());
            for (std::size_t i = 0; i < d_r.size(); ++i)
                ASSERT_NEAR(std::invoke(b, d_a[i]), d_r[i], eps);
        });
    }

    template <typename T>
    [[nodiscard]] auto compute_mean(std::span<const T> data) -> mag_e8m23_t {
        mag_e8m23_t sum {};
        for (const T x : data) sum += x;
        return sum / static_cast<mag_e8m23_t>(data.size());
    }

    template <typename T>
    [[nodiscard]] auto compute_mean(const mag_tensor_t* tensor) -> mag_e8m23_t {
        return compute_mean(std::span<const T>{reinterpret_cast<const T*>(mag_tensor_get_data_ptr(tensor)), static_cast<std::size_t>(tensor->numel)});
    }

    template <typename T>
    [[nodiscard]] auto compute_std(std::span<const T> data) -> mag_e11m52_t {
        mag_e8m23_t sum {};
        mag_e8m23_t mean {compute_mean(data)};
        for (const T x : data) {
            sum += std::pow(x-mean, 2.0f);
        }
        return std::sqrt(sum / static_cast<mag_e8m23_t>(data.size()));
    }

    template <typename T>
    [[nodiscard]] auto compute_std(const mag_tensor_t* tensor) -> mag_e11m52_t {
        return compute_std(std::span<const T>{reinterpret_cast<const T*>(mag_tensor_get_data_ptr(tensor)), static_cast<std::size_t>(tensor->numel)});
    }

    template <typename T>
    auto softmax(const T* input, T* output, std::size_t length) -> void {
        T max_input {input[0]};
        for (std::size_t i {1}; i < length; ++i) {
            if (input[i] > max_input) {
                max_input = input[i];
            }
        }
        T sum {};
        for (std::size_t i{}; i < length; ++i) {
            output[i] = std::exp(input[i] - max_input);
            sum += output[i];
        }
        for (std::size_t i{}; i < length; ++i) {
            output[i] /= sum;
        }
    }

    // Tiny module-like library for testing training and models
    namespace nn {
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
            explicit sgd(std::span<tensor> params, e8m23_t lr) : optimizer{params}, lr{lr} {}

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

            e8m23_t lr {};
        };

        // Linear/Dense layer
        class linear_layer final : public module {
        public:
            linear_layer(context& ctx, std::int64_t in_features, std::int64_t out_features, dtype type = dtype::e8m23, bool has_bias = true) {
                tensor weight {ctx, type, out_features, in_features};
                weight.fill_rand_normal(0.0f, 1.0f);
                weight = weight / static_cast<e8m23_t>(std::sqrt(in_features + out_features));
                register_param(weight);
                this->weight = weight;
                if (has_bias) {
                    tensor bias {ctx, type, out_features};
                    bias.fill_float(0.f);
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
}
