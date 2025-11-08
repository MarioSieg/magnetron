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

// Modern C++ API of a subset of the Magnetron C APi
// For easier testing and simpler C++ code

#pragma once

#include <magnetron/magnetron.h>

#include <algorithm>
#include <stdexcept>
#include <array>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <optional>
#include <span>
#include <vector>

namespace magnetron {
    /**
     * Thread scheduling priority for CPU compute, higher priority means more CPU time
     */
    enum class thread_sched_prio : std::underlying_type_t<mag_thread_prio_t> {
        normal = MAG_THREAD_PRIO_NORMAL,
        medium = MAG_THREAD_PRIO_MEDIUM,
        high = MAG_THREAD_PRIO_HIGH,
        realtime = MAG_THREAD_PRIO_REALTIME
    };

    /**
     * Desired color channels to load from image tensor
     */
    enum class color_channel : std::underlying_type_t<mag_color_channels_t> {
        automatic = MAG_COLOR_CHANNELS_AUTO,
        grayscale = MAG_COLOR_CHANNELS_GRAY,
        grayscale_alpha = MAG_COLOR_CHANNELS_GRAY_A,
        rgb = MAG_COLOR_CHANNELS_RGB,
        rgba = MAG_COLOR_CHANNELS_RGBA
    };

    /**
     * Enable or disable internal magnetron logging to stdout.
     * @param enable
     */
    inline auto enable_logging(bool enable) noexcept -> void {
        mag_set_log_mode(enable);
    }

    /**
     * The context owns all tensors and runtime data structures. It must kept alive as long as any tensor is used.
     */
    class context final {
    public:
        explicit context(const char *device_id = "cpu") noexcept {
            m_ctx = mag_ctx_create(device_id);
        }

        context(context&&) = default;
        context& operator=(context&&) = default;
        auto operator=(const context&) -> context& = delete;
        auto operator=(context&) -> context& = delete;

        ~context() {
            mag_ctx_destroy(m_ctx, false);
        }

        [[nodiscard]] auto operator *() noexcept -> mag_context_t& { return *m_ctx; }
        [[nodiscard]] auto operator *() const noexcept -> const mag_context_t& { return *m_ctx; }
        [[nodiscard]] auto device_name() const noexcept -> std::string_view { return mag_ctx_get_compute_device_name(m_ctx); }
        [[nodiscard]] auto os_name() const noexcept -> std::string_view { return mag_ctx_get_os_name(m_ctx); }
        [[nodiscard]] auto cpu_name() const noexcept -> std::string_view { return mag_ctx_get_cpu_name(m_ctx); }
        [[nodiscard]] auto cpu_virtual_cores() const noexcept -> std::uint32_t { return mag_ctx_get_cpu_virtual_cores(m_ctx); }
        [[nodiscard]] auto cpu_physical_cores() const noexcept -> std::uint32_t { return mag_ctx_get_cpu_physical_cores(m_ctx); }
        [[nodiscard]] auto cpu_sockets() const noexcept -> std::uint32_t { return mag_ctx_get_cpu_sockets(m_ctx); }
        [[nodiscard]] auto physical_memory_total() const noexcept -> std::uint64_t { return mag_ctx_get_physical_memory_total(m_ctx); }
        [[nodiscard]] auto physical_memory_free() const noexcept -> std::uint64_t { return mag_ctx_get_physical_memory_free(m_ctx); }
        [[nodiscard]] auto is_numa_system() const noexcept -> bool { return mag_ctx_is_numa_system(m_ctx); }
        [[nodiscard]] auto total_tensors_created() const noexcept -> std::size_t { return mag_ctx_get_total_tensors_created(m_ctx); }
        auto start_grad_recorder() noexcept -> void { mag_ctx_grad_recorder_start(m_ctx); }
        auto stop_grad_recorder() noexcept -> void { mag_ctx_grad_recorder_stop(m_ctx); }
        [[nodiscard]] auto is_recording_gradients() const noexcept -> bool { return mag_ctx_grad_recorder_is_running(m_ctx); }
        auto manual_seed(std::uint64_t seed) noexcept -> void { mag_ctx_manual_seed(m_ctx, seed); }

    private:
        mag_context_t* m_ctx {};
    };

    enum class dtype : std::underlying_type_t<mag_dtype_t> {
        e8m23 = MAG_DTYPE_E8M23,
        e5m10 = MAG_DTYPE_E5M10,
        boolean = MAG_DTYPE_BOOL,
        i32 = MAG_DTYPE_I32,
    };

    [[nodiscard]] inline auto dtype_size(dtype t) noexcept -> std::size_t {
        return mag_dtype_meta_of(static_cast<mag_dtype_t>(t))->size;
    }

    [[nodiscard]] inline auto dtype_name(dtype t) noexcept -> std::string_view {
        return mag_dtype_meta_of(static_cast<mag_dtype_t>(t))->name;
    }

    inline auto handle_error(mag_status_t status, mag_context_t *ctx = nullptr) -> void {
        if (status != MAG_STATUS_OK) [[unlikely]] {
            throw std::runtime_error {ctx ? mag_ctx_get_last_error(ctx)->message : mag_status_get_name(status)};
        }
    }

    /**
     * A 1-6 dimensional, reference counted tensor with a fixed size and data type.
     */
    class tensor final {
    public:
        tensor(context& ctx, dtype type, std::span<const std::int64_t> shape) {
            handle_error(mag_tensor_empty(&m_tensor, &*ctx, static_cast<mag_dtype_t>(type), shape.size(), shape.data()), &*ctx);
        }

        template <typename... S> requires std::is_integral_v<std::common_type_t<S...>>
        tensor(context& ctx, dtype type, S&&... shape) : tensor{ctx, type, std::array{static_cast<std::int64_t>(shape)...}} {}

        tensor(context& ctx, std::span<const std::int64_t> shape, std::span<const float> data) : tensor{ctx, dtype::e8m23, shape} {
            fill_from(data);
        }

        tensor(context& ctx, std::span<const std::int64_t> shape, std::span<const std::int32_t> data) : tensor{ctx, dtype::i32, shape} {
            fill_from(data);
        }

        tensor(const tensor& other) {
            mag_tensor_incref(other.m_tensor);
            m_tensor = other.m_tensor;
        }

        tensor(tensor&& other) {
            if (this != &other) {
                m_tensor = other.m_tensor;
                other.m_tensor = nullptr;
            }
        }

        auto operator = (const tensor& other) -> tensor& {
            if (this != &other) {
                mag_tensor_incref(other.m_tensor);
                mag_tensor_decref(m_tensor);
                m_tensor = other.m_tensor;
            }
            return *this;
        }

        auto operator = (tensor&& other) -> tensor& {
            if (this != &other) {
                mag_tensor_decref(m_tensor);
                m_tensor = other.m_tensor;
                other.m_tensor = nullptr;
            }
            return *this;
        }

        ~tensor() {
            if (m_tensor) {
                mag_tensor_decref(m_tensor);
            }
        }

        [[nodiscard]] auto operator * () noexcept -> mag_tensor_t& { return *m_tensor; }
        [[nodiscard]] auto operator * () const noexcept -> const mag_tensor_t& { return *m_tensor; }

        [[nodiscard]] auto clone() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_clone(&out, m_tensor));
            return tensor{out};
        }

        [[nodiscard]] auto view(std::initializer_list<std::int64_t> dims = {}) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_view(&out, m_tensor, std::empty(dims) ? nullptr : std::data(dims), std::size(dims)));
            return tensor{out};
        }

        [[nodiscard]] auto reshape(std::initializer_list<std::int64_t> dims = {}) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_reshape(&out, m_tensor, std::data(dims), std::size(dims)));
            return tensor{out};
        }

        [[nodiscard]] auto view_slice(std::int64_t dim, std::int64_t start, std::int64_t len, std::int64_t step) -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_view_slice(&out, m_tensor, dim, start, len, step));
            return tensor{out};
        }
        [[nodiscard]] auto T(std::int64_t dim1 = 0, std::int64_t dim2 = 1) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_transpose(&out, m_tensor, dim1, dim2));
            return tensor{out};
        }
        [[nodiscard]] auto transpose(std::int64_t dim1 = 0, std::int64_t dim2 = 1) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_transpose(&out, m_tensor, dim1, dim2));
            return tensor{out};
        }
        [[nodiscard]] auto permute(const std::array<std::int64_t, MAG_MAX_DIMS>& axes) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_permute(&out, m_tensor, axes.data(), axes.size()));
            return tensor{out};
        }
        [[nodiscard]] auto mean() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_mean(&out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto min() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_min(&out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto max() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_max(&out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto sum() const noexcept -> tensor {   mag_tensor_t *out = nullptr;
            handle_error(mag_sum(&out, m_tensor, nullptr, 0, false));
            return tensor{out}; }
        [[nodiscard]] auto argmin() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_argmin(&out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto argmax() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_argmax(&out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto abs() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_abs(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto abs_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_abs_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sgn() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sgn(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sgn_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sgn_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto neg() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_neg(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto neg_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_neg_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sqr() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sqr(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sqr_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sqr_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sqrt() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sqrt(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sqrt_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sqrt_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sin() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sin(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sin_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sin_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto cos() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cos(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto cos_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cos_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto step() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_step(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto step_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_step_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto exp() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_exp(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto exp_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_exp_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto floor() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_floor(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto floor_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_floor_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto ceil() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_ceil(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto ceil_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_ceil_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto round() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_round(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto round_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_round_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto softmax() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_softmax(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto softmax_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_softmax_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sigmoid() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sigmoid(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sigmoid_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sigmoid_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto hard_sigmoid() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_hard_sigmoid(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto hard_sigmoid_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_hard_sigmoid_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto silu() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_silu(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto silu_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_silu_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto tanh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_tanh(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto tanh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_tanh_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto relu() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_relu(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto relu_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_relu_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto gelu() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gelu(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto gelu_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gelu_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto gelu_approx() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gelu_approx(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto gelu_approx_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gelu_approx_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto add(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_add(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto add_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_add_(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto sub(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sub(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto sub_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sub_(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto mul(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_mul(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto mul_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_mul_(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto div(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_div(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto div_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_div_(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto matmul(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_matmul(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto add(float other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_tensor_scalar(&sca, mag_tensor_get_ctx(m_tensor), mag_tensor_get_dtype(m_tensor), other));
            return add(tensor{sca});
        }
        [[nodiscard]] auto sub(float other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_tensor_scalar(&sca, mag_tensor_get_ctx(m_tensor), mag_tensor_get_dtype(m_tensor), other));
            return sub(tensor{sca});
        }
        [[nodiscard]] auto mul(float other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_tensor_scalar(&sca, mag_tensor_get_ctx(m_tensor), mag_tensor_get_dtype(m_tensor), other));
            return mul(tensor{sca});
        }
        [[nodiscard]] auto div(float other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_tensor_scalar(&sca, mag_tensor_get_ctx(m_tensor), mag_tensor_get_dtype(m_tensor), other));
            return div(tensor{sca});
        }
        [[nodiscard]] auto band(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_and(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto band_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_and_(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bor(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_or(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bor_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_or_(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bxor(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_xor(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bxor_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_xor_(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bnot() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_not(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto bnot_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_not_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto bshl(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_shl(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bshl_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_shl_(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bshr(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_shr(&out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bshr_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_shr_(&out, m_tensor, &*other));
            return tensor{out};
        }

        [[nodiscard]] auto operator + (tensor other) const noexcept -> tensor { return add(other); }
        [[nodiscard]] auto operator + (float other) const noexcept -> tensor { return add(other); }
        auto operator += (tensor other) const noexcept -> tensor { return add_(other); }
        [[nodiscard]] auto operator - (tensor other) const noexcept -> tensor { return sub(other); }
        [[nodiscard]] auto operator - (float other) const noexcept -> tensor { return sub(other); }
        auto operator -= (tensor other) const noexcept -> tensor { return sub_(other); }
        [[nodiscard]] auto operator * (tensor other) const noexcept -> tensor { return mul(other); }
        [[nodiscard]] auto operator * (float other) const noexcept -> tensor { return mul(other); }
        auto operator *= (tensor other) const noexcept -> tensor { return mul_(other); }
        [[nodiscard]] auto operator / (tensor other) const noexcept -> tensor { return div(other); }
        [[nodiscard]] auto operator / (float other) const noexcept -> tensor { return div(other); }
        auto operator /= (tensor other) const noexcept -> tensor { return div_(other); }

        [[nodiscard]] auto operator % (tensor other) const noexcept -> tensor { return matmul(other); } // we use the % operator for matmul in C++, as @ is not allowed

        [[nodiscard]] auto operator & (tensor other) const noexcept -> tensor { return band(other); }
        auto operator &= (tensor other) const noexcept -> tensor { return band_(other); }
        [[nodiscard]] auto operator | (tensor other) const noexcept -> tensor { return bor(other); }
        auto operator |= (tensor other) const noexcept -> tensor { return bor_(other); }
        [[nodiscard]] auto operator ^ (tensor other) const noexcept -> tensor { return bxor(other); }
        auto operator ^= (tensor other) const noexcept -> tensor { return bxor_(other); }
        [[nodiscard]] auto operator ~ () const noexcept -> tensor { return bnot(); }
        [[nodiscard]] auto operator << (tensor other) const noexcept -> tensor { return bshl(other); }
        auto operator <<= (tensor other) const noexcept -> tensor { return bshl_(other); }
        [[nodiscard]] auto operator >> (tensor other) const noexcept -> tensor { return bshr(other); }
        auto operator >>= (tensor other) const noexcept -> tensor { return bshr_(other); }

        auto operator == (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_eq(&out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator != (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_ne(&out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator <= (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_le(&out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator >= (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_ge(&out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator < (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_lt(&out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator > (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gt(&out, m_tensor, &*other));
            return tensor{out};
        }

        auto fill_from(const void* buf, std::size_t nb) -> void {
            mag_tensor_fill_from_raw_bytes(m_tensor, buf, nb);
        }

        auto fill_from(std::span<const float> data) -> void {
            mag_tensor_fill_from_floats(m_tensor, data.data(), data.size());
        }

        auto fill_from(std::span<const bool> data) -> void {
            static_assert(sizeof(bool) == sizeof(std::uint8_t));
            mag_tensor_fill_from_raw_bytes(m_tensor, data.data(), data.size_bytes());
        }

        auto fill_from(const std::vector<bool>& data) -> void {
            static_assert(sizeof(bool) == sizeof(std::uint8_t));
            std::vector<std::uint8_t> unpacked {};
            unpacked.resize(data.size());
            std::ranges::copy(data, unpacked.begin());
            mag_tensor_fill_from_raw_bytes(m_tensor, unpacked.data(), unpacked.size());
        }

        auto fill_from(std::span<const std::int32_t> data) -> void {
            mag_tensor_fill_from_raw_bytes(m_tensor, data.data(), data.size_bytes());
        }

        template <typename T>
        auto fill(T val) -> void;

        template <typename T>
        auto masked_fill(tensor mask, T val) -> void;

        template <typename T>
        auto fill_rand_uniform(T min, T max) -> void;

        auto fill_rand_normal(float mean, float stddev) -> void {
            mag_tensor_fill_random_normal(m_tensor, mean, stddev);
        }

        auto fill_rand_bernoulli(float p = 0.5f) -> void {
            mag_tensor_fill_random_bernoulli(m_tensor, p);
        }

        auto fill_arange(float start, float step = 1.0f) -> void {
            mag_tensor_fill_arange(m_tensor, start, step);
        }

        [[nodiscard]] auto to_string(bool with_data = true, std::size_t from_start = 0, std::size_t from_end = 0) const -> std::string {
            char* fmt {mag_tensor_to_string(m_tensor, with_data, from_start, from_end)};
            std::string str {fmt};
            mag_tensor_to_string_free_data(fmt);
            return str;
        }
        [[nodiscard]] auto rank() const noexcept -> std::int64_t { return mag_tensor_get_rank(m_tensor); }
        [[nodiscard]] auto shape() const noexcept -> std::span<const std::int64_t> {
            return {mag_tensor_get_shape(m_tensor), static_cast<std::size_t>(rank())};
        }
        [[nodiscard]] auto strides() const noexcept -> std::span<const std::int64_t> {
            return {mag_tensor_get_strides(m_tensor), static_cast<std::size_t>(rank())};
        }
        [[nodiscard]] auto dtype() const noexcept -> dtype { return static_cast<enum dtype>(mag_tensor_get_dtype(m_tensor)); }
        [[nodiscard]] auto data_ptr() const noexcept -> void* { return mag_tensor_get_data_ptr(m_tensor); }
        [[nodiscard]] auto storage_base_ptr() const noexcept -> void* { return mag_tensor_get_storage_base_ptr(m_tensor); }

        template <typename T>
        [[nodiscard]] auto to_vector() const -> std::vector<T>;
        [[nodiscard]] auto data_size() const noexcept -> std::int64_t { return mag_tensor_get_data_size(m_tensor); }
        [[nodiscard]] auto numel() const noexcept -> std::int64_t { return mag_tensor_get_numel(m_tensor); }
        [[nodiscard]] auto is_shape_eq(tensor other) const noexcept -> bool { return mag_tensor_is_shape_eq(m_tensor, &*other); }
        [[nodiscard]] auto can_broadcast(tensor other) const noexcept -> bool { return mag_tensor_can_broadcast(m_tensor, &*other); }
        [[nodiscard]] auto is_transposed() const noexcept -> bool { return mag_tensor_is_transposed(m_tensor); }
        [[nodiscard]] auto is_permuted() const noexcept -> bool { return mag_tensor_is_permuted(m_tensor); }
        [[nodiscard]] auto is_contiguous() const noexcept -> bool { return mag_tensor_is_contiguous(m_tensor); }
        [[nodiscard]] auto is_view() const noexcept -> bool { return mag_tensor_is_view(m_tensor); }
        [[nodiscard]] auto is_floating_point_typed() const noexcept -> bool { return mag_tensor_is_floating_point_typed(m_tensor); }
        [[nodiscard]] auto is_integral_typed() const noexcept -> bool { return mag_tensor_is_integral_typed(m_tensor); }
        [[nodiscard]] auto is_integer_typed() const noexcept -> bool { return mag_tensor_is_integer_typed(m_tensor); }
        [[nodiscard]] auto is_numeric_typed() const noexcept -> bool { return mag_tensor_is_numeric_typed(m_tensor); }

        [[nodiscard]] auto grad() const noexcept -> std::optional<tensor> {
            mag_tensor_t *grad;
            mag_status_t stat = mag_tensor_get_grad(m_tensor, &grad);
            if (stat != MAG_STATUS_OK) return std::nullopt;
            return tensor{grad};
        }
        [[nodiscard]] auto requires_grad() const noexcept -> bool { return mag_tensor_requires_grad(m_tensor); }
        auto requires_grad(bool yes) noexcept -> void { mag_tensor_set_requires_grad(m_tensor, yes); }
        auto backward() -> void { mag_tensor_backward(m_tensor); }
        auto zero_grad() -> void { mag_tensor_zero_grad(m_tensor); }

        [[nodiscard]] auto operator ()(const std::array<std::int64_t, MAG_MAX_DIMS>& idx) const noexcept -> float {
            return mag_tensor_subscript_get_multi(m_tensor, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
        }
        auto operator ()(const std::array<std::int64_t, MAG_MAX_DIMS>& idx, float x) const noexcept -> void {
            mag_tensor_subscript_set_multi(m_tensor, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], x);
        }
        [[nodiscard]] auto operator ()(std::int64_t idx) const noexcept -> float {
            return mag_tensor_subscript_get_flattened(m_tensor, idx);
        }
        auto operator ()(std::int64_t idx, float x) const noexcept -> void {
            mag_tensor_subscript_set_flattened(m_tensor, idx, x);
        }

        explicit tensor(mag_tensor_t* ptr) noexcept : m_tensor{ptr} {}

    private:
        friend class storage_stream;

        mag_tensor_t* m_tensor {};
    };

    template <>
    inline auto tensor::fill(float val) -> void {
        mag_tensor_fill_float(m_tensor, val);
    }

    template <>
    inline auto tensor::fill(std::int32_t val) -> void {
        mag_tensor_fill_int(m_tensor, val);
    }

    template <>
    inline auto tensor::fill(bool val) -> void {
        mag_tensor_fill_int(m_tensor, val ? 1 : 0);
    }

    template <>
    inline auto tensor::masked_fill(tensor mask, float val) -> void {
        if (mask.dtype() != dtype::boolean)
            throw std::runtime_error {"mask must be bool tensor"};
        mag_tensor_masked_fill_float(&*mask, m_tensor, val);
    }

    template <>
    inline auto tensor::masked_fill(tensor mask, std::int32_t val) -> void {
        if (mask.dtype() != dtype::boolean)
            throw std::runtime_error {"mask must be bool tensor"};
        mag_tensor_masked_fill_int(&*mask, m_tensor, val);
    }

    template <>
    inline auto tensor::masked_fill(tensor mask, bool val) -> void {
        if (mask.dtype() != dtype::boolean)
            throw std::runtime_error {"mask must be bool tensor"};
        mag_tensor_masked_fill_int(&*mask, m_tensor, val ? 1 : 0);
    }

    template <>
    inline auto tensor::fill_rand_uniform(float min, float max) -> void {
        mag_tensor_fill_random_uniform_float(m_tensor, min, max);
    }

    template <>
    inline auto tensor::fill_rand_uniform(std::int32_t min, std::int32_t max) -> void {
        mag_tensor_fill_random_uniform_int(m_tensor, min, max);
    }

    template <>
    inline auto tensor::to_vector() const -> std::vector<float> {
        if (!is_floating_point_typed())
            throw std::runtime_error {"requires floating point dtype"};
        auto* data {mag_tensor_get_data_as_floats(m_tensor)};
        std::vector<float> result {};
        result.resize(numel());
        std::copy_n(data, numel(), result.begin());
        mag_tensor_get_data_as_floats_free(data);
        return result;
    }

    template <>
    inline auto tensor::to_vector() const -> std::vector<bool> {
        if (dtype() != dtype::boolean)
            throw std::runtime_error {"requires boolean dtype"};
        auto* data {static_cast<std::uint8_t*>(mag_tensor_get_raw_data_as_bytes(m_tensor))};
        std::vector<bool> result {};
        result.resize(numel());
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<bool>(data[i]);
        mag_tensor_get_raw_data_as_bytes_free(data);
        return result;
    }

    template <>
    inline auto tensor::to_vector() const -> std::vector<std::int32_t> {
        if (dtype() != dtype::i32)
            throw std::runtime_error {"requires int32 dtype"};
        auto* data {static_cast<std::int32_t*>(mag_tensor_get_raw_data_as_bytes(m_tensor))};
        std::vector<std::int32_t> result {};
        result.resize(numel());
        std::copy_n(data, numel(), result.begin());
        mag_tensor_get_raw_data_as_bytes_free(data);
        return result;
    }
}
