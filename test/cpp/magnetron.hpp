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

#include <core/mag_rc.h>
#include <core/mag_backend.h>
#include <core/mag_tensor.h>

#include <algorithm>
#include <stdexcept>
#include <optional>
#include <vector>

#include "extern/half/include/half.hpp"

namespace magnetron {
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
        [[nodiscard]] auto cpu_virtual_cores() const noexcept -> uint32_t { return mag_ctx_get_cpu_virtual_cores(m_ctx); }
        [[nodiscard]] auto cpu_physical_cores() const noexcept -> uint32_t { return mag_ctx_get_cpu_physical_cores(m_ctx); }
        [[nodiscard]] auto cpu_sockets() const noexcept -> uint32_t { return mag_ctx_get_cpu_sockets(m_ctx); }
        [[nodiscard]] auto physical_memory_total() const noexcept -> uint64_t { return mag_ctx_get_physical_memory_total(m_ctx); }
        [[nodiscard]] auto physical_memory_free() const noexcept -> uint64_t { return mag_ctx_get_physical_memory_free(m_ctx); }
        [[nodiscard]] auto is_numa_system() const noexcept -> bool { return mag_ctx_is_numa_system(m_ctx); }
        [[nodiscard]] auto total_tensors_created() const noexcept -> size_t { return mag_ctx_get_total_tensors_created(m_ctx); }
        auto start_grad_recorder() noexcept -> void { mag_ctx_grad_recorder_start(m_ctx); }
        auto stop_grad_recorder() noexcept -> void { mag_ctx_grad_recorder_stop(m_ctx); }
        [[nodiscard]] auto is_recording_gradients() const noexcept -> bool { return mag_ctx_grad_recorder_is_running(m_ctx); }
        auto manual_seed(uint64_t seed) noexcept -> void { mag_ctx_manual_seed(m_ctx, seed); }

    private:
        mag_context_t* m_ctx {};
    };

    enum class dtype : std::underlying_type_t<mag_dtype_t> {
        float32 = MAG_DTYPE_FLOAT32,
        float16 = MAG_DTYPE_FLOAT16,
        boolean = MAG_DTYPE_BOOLEAN,
        u8 = MAG_DTYPE_UINT8,
        i8 = MAG_DTYPE_INT8,
        u16 = MAG_DTYPE_UINT16,
        i16 = MAG_DTYPE_INT16,
        u32 = MAG_DTYPE_UINT32,
        i32 = MAG_DTYPE_INT32,
        u64 = MAG_DTYPE_UINT64,
        i64 = MAG_DTYPE_INT64,
    };

    [[nodiscard]] inline auto dtype_size(dtype t) noexcept -> size_t {
        return mag_type_trait(static_cast<mag_dtype_t>(t))->size;
    }

    [[nodiscard]] inline auto dtype_name(dtype t) noexcept -> std::string_view {
        return mag_type_trait(static_cast<mag_dtype_t>(t))->name;
    }

    inline auto handle_error(mag_status_t status, mag_context_t *ctx = nullptr) -> void {
        if (status != MAG_STATUS_OK) [[unlikely]] {
            std::printf("%s", ctx ? mag_ctx_get_last_error(ctx)->message : mag_status_get_name(status));
            std::fflush(stdout);
            std::abort();
        }
    }

    template <typename T>
    [[nodiscard]] constexpr std::optional<dtype> generic_to_dtype() {
        if constexpr (std::is_same_v<T, bool>) return dtype::boolean;
        if constexpr (std::is_same_v<T, uint8_t>) return dtype::u8;
        if constexpr (std::is_same_v<T, int8_t>) return dtype::i8;
        if constexpr (std::is_same_v<T, uint16_t>) return dtype::u16;
        if constexpr (std::is_same_v<T, int16_t>) return dtype::i16;
        if constexpr (std::is_same_v<T, uint32_t>) return dtype::u32;
        if constexpr (std::is_same_v<T, int32_t>) return dtype::i32;
        if constexpr (std::is_same_v<T, uint64_t>) return dtype::u64;
        if constexpr (std::is_same_v<T, int64_t>) return dtype::i64;
        if constexpr (std::is_same_v<T, float>) return dtype::float32;
        if constexpr (std::is_same_v<T, mag_float16_t>) return dtype::float16;
        if constexpr (std::is_same_v<T, half_float::half>) return dtype::float16;
        return std::nullopt;
    }

    /**
     * A 1-6 dimensional, reference counted tensor with a fixed size and data type.
     */
    class tensor final {
    public:
        tensor(context& ctx, dtype type, std::initializer_list<int64_t> shape) {
            if (shape.size() == 1 && *shape.begin() == 1) handle_error(mag_empty_scalar(&m_tensor, &*ctx, static_cast<mag_dtype_t>(type)), &*ctx);
            else handle_error(mag_empty(&m_tensor, &*ctx, static_cast<mag_dtype_t>(type), shape.size(), shape.begin()), &*ctx);
        }

        tensor(context& ctx, dtype type, const std::vector<int64_t>& shape) {
            if (shape.size() == 1 && *shape.begin() == 1) handle_error(mag_empty_scalar(&m_tensor, &*ctx, static_cast<mag_dtype_t>(type)), &*ctx);
            else handle_error(mag_empty(&m_tensor, &*ctx, static_cast<mag_dtype_t>(type), shape.size(), shape.data()), &*ctx);
        }

        template <typename... S, typename = std::enable_if_t<std::conjunction_v<std::is_integral<std::decay_t<S>>...>>>
        tensor(context& ctx, dtype type, S&&... shape) : tensor{ctx, type, {static_cast<int64_t>(shape)...}} {}

        tensor(context& ctx, std::initializer_list<int64_t> shape, const std::vector<float>& data) : tensor{ctx, dtype::float32, shape} {
            copy_(data);
        }

        tensor(context& ctx, std::initializer_list<int64_t> shape, const std::vector<int32_t>& data) : tensor{ctx, dtype::i32, shape} {
            copy_(data);
        }

        tensor(const tensor& other) {
            mag_rc_incref(other.m_tensor);
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
                mag_rc_incref(other.m_tensor);
                mag_rc_decref(m_tensor);
                m_tensor = other.m_tensor;
            }
            return *this;
        }

        auto operator = (tensor&& other) -> tensor& {
            if (this != &other) {
                mag_rc_decref(m_tensor);
                m_tensor = other.m_tensor;
                other.m_tensor = nullptr;
            }
            return *this;
        }

        ~tensor() {
            if (m_tensor) {
                mag_rc_decref(m_tensor);
            }
        }

        [[nodiscard]] auto operator * () noexcept -> mag_tensor_t& { return *m_tensor; }
        [[nodiscard]] auto operator * () const noexcept -> const mag_tensor_t& { return *m_tensor; }

        [[nodiscard]] auto clone() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_clone(&out, m_tensor));
            return tensor{out};
        }

        [[nodiscard]] auto cast(dtype dst_dtype) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cast(&out, m_tensor, static_cast<mag_dtype_t>(dst_dtype)));
            return tensor{out};
        }

        [[nodiscard]] auto view(const std::vector<int64_t>& dims = {}) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_view(&out, m_tensor, dims.empty() ? nullptr : dims.data(), dims.size()));
            return tensor{out};
        }

        [[nodiscard]] auto reshape(const std::vector<int64_t>& dims = {}) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_reshape(&out, m_tensor, dims.data(), dims.size()));
            return tensor{out};
        }

        [[nodiscard]] auto view_slice(int64_t dim, int64_t start, int64_t len, int64_t step) -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_view_slice(&out, m_tensor, dim, start, len, step));
            return tensor{out};
        }
        [[nodiscard]] auto T(int64_t dim1 = 0, int64_t dim2 = 1) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_transpose(&out, m_tensor, dim1, dim2));
            return tensor{out};
        }
        [[nodiscard]] auto transpose(int64_t dim1 = 0, int64_t dim2 = 1) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_transpose(&out, m_tensor, dim1, dim2));
            return tensor{out};
        }
        [[nodiscard]] auto permute(const std::vector<int64_t>& axes) const noexcept -> tensor {
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
        [[nodiscard]] auto log10() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log10(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log10_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log10_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log1p() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log1p(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log1p_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log1p_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log2() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log2(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log2_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log2_(&out, m_tensor));
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
        [[nodiscard]] auto rcp() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_rcp(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto rcp_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_rcp_(&out, m_tensor));
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
        [[nodiscard]] auto rsqrt() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_rsqrt(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto rsqrt_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_rsqrt_(&out, m_tensor));
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
        [[nodiscard]] auto tan() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_tan(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto tan_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_tan_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto asin() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_asin(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto asin_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sin_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto acos() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_acos(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto acos_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_acos_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto atan() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_atan(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto atan_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_atan_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sinh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sinh(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sinh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sinh_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto cosh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cosh(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto cosh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cosh_(&out, m_tensor));
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
        [[nodiscard]] auto asinh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_asinh(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto asinh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_asinh_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto acosh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_acosh(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto acosh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_acosh_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto atanh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_atanh(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto atanh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_atanh_(&out, m_tensor));
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
        [[nodiscard]] auto erf() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_erf(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto erf_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_erf_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto erfc() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_erfc(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto erfc_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_erfc_(&out, m_tensor));
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
        [[nodiscard]] auto exp2() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_exp2(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto exp2_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_exp2_(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto expm1() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_expm1(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto expm1_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_expm1_(&out, m_tensor));
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
        [[nodiscard]] auto trunc() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_trunc(&out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto trunc_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_trunc_(&out, m_tensor));
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
        [[nodiscard]] auto add(double other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_scalar(&sca, mag_tensor_context(m_tensor), mag_tensor_type(m_tensor), mag_scalar_float(other)));
            return add(tensor{sca});
        }
        [[nodiscard]] auto sub(double other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_scalar(&sca, mag_tensor_context(m_tensor), mag_tensor_type(m_tensor), mag_scalar_float(other)));
            return sub(tensor{sca});
        }
        [[nodiscard]] auto mul(double other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_scalar(&sca, mag_tensor_context(m_tensor), mag_tensor_type(m_tensor), mag_scalar_float(other)));
            return mul(tensor{sca});
        }
        [[nodiscard]] auto div(double other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_scalar(&sca, mag_tensor_context(m_tensor), mag_tensor_type(m_tensor), mag_scalar_float(other)));
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

        auto copy_(const void* buf, size_t nb) -> void {
            mag_copy_raw_(m_tensor, buf, nb);
        }

        template <typename T>
        auto copy_(const std::vector<T>& data) -> void {
            if (generic_to_dtype<T>() != dtype())
                throw std::runtime_error{"data type does not match tensor dtype"};
            mag_copy_raw_(m_tensor, data.data(), data.size()*sizeof(data[0]));
        }

        auto copy_(const std::vector<uint8_t>& data) -> void {
            std::vector<uint8_t> unpacked {};
            unpacked.resize(data.size());
            for (size_t i=0; i < unpacked.size(); ++i) unpacked[i] = data[i];
            mag_copy_raw_(m_tensor, unpacked.data(), unpacked.size()*sizeof(data[0]));
        }

        template <typename T>
        auto fill_(T val) -> void {
            static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, bool>);
            if constexpr (std::is_floating_point_v<T>) {
                handle_error(mag_fill_(m_tensor, mag_scalar_float(static_cast<double>(val))));
            } else if constexpr (std::is_integral_v<T>) {
                handle_error(mag_fill_(m_tensor, mag_scalar_int(static_cast<int64_t>(val))));
            } else {
                throw std::runtime_error{"unsupported type for fill_"};
            }
        }

        template <typename T>
        auto masked_fill_(tensor mask, T val) -> void;

        template <typename T>
        auto uniform_(T min, T max) -> void;

        auto normal_(float mean, float stddev) -> void {
            mag_normal_(m_tensor, mag_scalar_float(mean), mag_scalar_float(stddev));
        }

        auto bernoulli_(float p = 0.5f) -> void {
            mag_bernoulli_(m_tensor, mag_scalar_float(p));
        }

        [[nodiscard]] auto to_string(int64_t head = 3, int64_t tail = 3, int64_t threshold = 1000) const -> std::string {
            char* fmt {mag_tensor_to_string(m_tensor, head, tail, threshold)};
            std::string str {fmt};
            mag_tensor_to_string_free_data(fmt);
            return str;
        }
        [[nodiscard]] auto rank() const noexcept -> int64_t { return mag_tensor_rank(m_tensor); }
        [[nodiscard]] auto shape() const noexcept -> std::vector<int64_t> {
            const int64_t *p = mag_tensor_shape_ptr(m_tensor);
            return std::vector<int64_t>{p, p+rank()}; /* We also copy unused dims as they are checked in some tests */
        }
        [[nodiscard]] auto strides() const noexcept -> std::vector<int64_t> {
            const int64_t *p = mag_tensor_strides_ptr(m_tensor);
            return std::vector<int64_t>{p, p+rank()}; /* We also copy unused dims as they are checked in some tests */
        }
        [[nodiscard]] auto dtype() const noexcept -> dtype { return static_cast<enum dtype>(mag_tensor_type(m_tensor)); }
        [[nodiscard]] auto data_ptr() const noexcept -> void* { return reinterpret_cast<void *>(mag_tensor_data_ptr_mut(m_tensor)); }
        [[nodiscard]] auto storage_base_ptr() const noexcept -> void* { return reinterpret_cast<void *>(mag_tensor_data_storage_ptr_mut(m_tensor)); }

        template <typename T>
        [[nodiscard]] auto to_vector() const -> std::vector<T> {
            static_assert(!std::is_same_v<T, bool>); // use uint8_t for bool
            if (dtype() != generic_to_dtype<T>())
                throw std::runtime_error {"T and tensor dtype must match: " + std::string{typeid(std::decay_t<T>).name()} + " != " + std::string{mag_type_trait(m_tensor->dtype)->name}};
            auto* data {static_cast<T *>(mag_tensor_copy_data(m_tensor))};
            std::vector<T> result {};
            result.resize(numel());
            std::copy_n(data, numel(), result.begin());
            mag_tensor_copy_data_free(data);
            return result;
        }

        [[nodiscard]] auto data_size() const noexcept -> int64_t { return mag_tensor_numbytes(m_tensor); }
        [[nodiscard]] auto numel() const noexcept -> int64_t { return mag_tensor_numel(m_tensor); }
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
            mag_status_t stat = mag_tensor_grad(m_tensor, &grad);
            if (stat != MAG_STATUS_OK) return std::nullopt;
            return tensor{grad};
        }
        [[nodiscard]] auto requires_grad() const noexcept -> bool { return mag_tensor_requires_grad(m_tensor); }
        auto requires_grad(bool yes) noexcept -> void { mag_tensor_set_requires_grad(m_tensor, yes); }
        auto backward() -> void { mag_tensor_backward(m_tensor); }
        auto zero_grad() -> void { mag_tensor_zero_grad(m_tensor); }

        explicit tensor(mag_tensor_t* ptr) noexcept : m_tensor{ptr} {}

    private:
        friend class storage_stream;

        mag_tensor_t* m_tensor {};
    };

    template <>
    inline auto tensor::masked_fill_(tensor mask, float val) -> void {
        if (mask.dtype() != dtype::boolean)
            throw std::runtime_error {"mask must be bool tensor"};
        handle_error(mag_masked_fill_(m_tensor, &*mask, mag_scalar_float(val)));
    }

    template <>
    inline auto tensor::masked_fill_(tensor mask, int64_t val) -> void {
        if (mask.dtype() != dtype::boolean)
            throw std::runtime_error {"mask must be bool tensor"};
        handle_error(mag_masked_fill_(m_tensor, &*mask, mag_scalar_int(val)));
    }

    template <>
    inline auto tensor::masked_fill_(tensor mask, bool val) -> void {
        if (mask.dtype() != dtype::boolean)
            throw std::runtime_error {"mask must be bool tensor"};
        handle_error(mag_masked_fill_(m_tensor, &*mask, mag_scalar_int(val)));
    }

    template <>
    inline auto tensor::uniform_(float min, float max) -> void {
        handle_error(mag_uniform_(m_tensor, mag_scalar_float(min), mag_scalar_float(max)));
    }

    template <>
    inline auto tensor::uniform_(int64_t min, int64_t max) -> void {
        handle_error(mag_uniform_(m_tensor, mag_scalar_int(min), mag_scalar_int(max)));
    }
}
