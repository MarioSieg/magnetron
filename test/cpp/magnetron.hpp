/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
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
        context() noexcept {
            m_ctx = mag_ctx_create();
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
        bfloat16 = MAG_DTYPE_BFLOAT16,
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

    inline mag_error_t g_error; // Suboptimal, but only for tests for now. Can be improved but no time for this rn

    inline auto handle_error(mag_status_t status, mag_context_t * = nullptr) -> void {
        if (status != MAG_STATUS_OK) [[unlikely]] {
            std::printf("%s - %s", mag_status_get_name(status), g_error.message); // Not the best solution as we discord all info of mag_error_t, but it's for tests only so yeah
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

    class tensor final {
    public:
        tensor(context& ctx, dtype type, std::initializer_list<int64_t> shape) {
            if (shape.size() == 1 && *shape.begin() == 1) handle_error(mag_empty_scalar(&g_error, &m_tensor, &*ctx, static_cast<mag_dtype_t>(type), mag_device(CPU, 0)), &*ctx);
            else handle_error(mag_empty(&g_error, &m_tensor, &*ctx, static_cast<mag_dtype_t>(type), shape.size(), shape.begin(), mag_device(CPU, 0)), &*ctx);
        }

        tensor(context& ctx, dtype type, const std::vector<int64_t>& shape) {
            if (shape.size() == 1 && *shape.begin() == 1) handle_error(mag_empty_scalar(&g_error, &m_tensor, &*ctx, static_cast<mag_dtype_t>(type), mag_device(CPU, 0)), &*ctx);
            else handle_error(mag_empty(&g_error, &m_tensor, &*ctx, static_cast<mag_dtype_t>(type), shape.size(), shape.data(), mag_device(CPU, 0)), &*ctx);
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
            handle_error(mag_clone(&g_error,  &out, m_tensor));
            return tensor{out};
        }

        [[nodiscard]] auto cast(dtype dst_dtype) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cast(&g_error,  &out, m_tensor, static_cast<mag_dtype_t>(dst_dtype)));
            return tensor{out};
        }

        [[nodiscard]] auto view(const std::vector<int64_t>& dims = {}) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_view(&g_error,  &out, m_tensor, dims.empty() ? nullptr : dims.data(), dims.size()));
            return tensor{out};
        }

        [[nodiscard]] auto reshape(const std::vector<int64_t>& dims = {}) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_reshape(&g_error,  &out, m_tensor, dims.data(), dims.size()));
            return tensor{out};
        }

        [[nodiscard]] auto view_slice(int64_t dim, int64_t start, int64_t len, int64_t step) -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_view_slice(&g_error,  &out, m_tensor, dim, start, len, step));
            return tensor{out};
        }
        [[nodiscard]] auto T(int64_t dim1 = 0, int64_t dim2 = 1) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_transpose(&g_error,  &out, m_tensor, dim1, dim2));
            return tensor{out};
        }
        [[nodiscard]] auto transpose(int64_t dim1 = 0, int64_t dim2 = 1) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_transpose(&g_error,  &out, m_tensor, dim1, dim2));
            return tensor{out};
        }
        [[nodiscard]] auto permute(const std::vector<int64_t>& axes) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_permute(&g_error,  &out, m_tensor, axes.data(), axes.size()));
            return tensor{out};
        }
        [[nodiscard]] auto mean() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_mean(&g_error,  &out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto min() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_min(&g_error,  &out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto max() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_max(&g_error,  &out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto sum() const noexcept -> tensor {   mag_tensor_t *out = nullptr;
            handle_error(mag_sum(&g_error,  &out, m_tensor, nullptr, 0, false));
            return tensor{out}; }
        [[nodiscard]] auto argmin() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_argmin(&g_error,  &out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto argmax() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_argmax(&g_error,  &out, m_tensor, nullptr, 0, false));
            return tensor{out};
        }
        [[nodiscard]] auto abs() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_abs(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto abs_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_abs_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sgn() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sgn(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sgn_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sgn_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto neg() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_neg(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto neg_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_neg_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log10() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log10(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log10_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log10_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log1p() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log1p(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log1p_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log1p_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log2() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log2(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto log2_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_log2_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sqr() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sqr(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sqr_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sqr_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto rcp() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_rcp(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto rcp_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_rcp_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sqrt() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sqrt(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sqrt_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sqrt_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto rsqrt() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_rsqrt(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto rsqrt_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_rsqrt_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sin() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sin(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sin_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sin_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto cos() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cos(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto cos_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cos_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto tan() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_tan(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto tan_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_tan_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto asin() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_asin(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto asin_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sin_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto acos() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_acos(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto acos_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_acos_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto atan() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_atan(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto atan_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_atan_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sinh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sinh(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sinh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sinh_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto cosh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cosh(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto cosh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_cosh_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto tanh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_tanh(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto tanh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_tanh_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto asinh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_asinh(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto asinh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_asinh_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto acosh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_acosh(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto acosh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_acosh_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto atanh() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_atanh(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto atanh_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_atanh_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto step() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_step(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto step_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_step_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto erf() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_erf(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto erf_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_erf_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto erfc() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_erfc(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto erfc_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_erfc_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto exp() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_exp(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto exp_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_exp_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto exp2() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_exp2(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto exp2_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_exp2_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto expm1() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_expm1(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto expm1_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_expm1_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto floor() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_floor(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto floor_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_floor_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto ceil() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_ceil(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto ceil_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_ceil_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto round() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_round(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto round_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_round_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto trunc() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_trunc(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto trunc_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_trunc_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto softmax() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_softmax(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto softmax_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_softmax_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sigmoid() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sigmoid(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto sigmoid_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sigmoid_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto hard_sigmoid() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_hard_sigmoid(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto hard_sigmoid_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_hard_sigmoid_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto silu() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_silu(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto silu_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_silu_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto relu() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_relu(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto relu_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_relu_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto gelu() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gelu(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto gelu_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gelu_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto gelu_approx() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gelu_approx(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto gelu_approx_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gelu_approx_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto add(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_add(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto add_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_add_(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto sub(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sub(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto sub_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_sub_(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto mul(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_mul(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto mul_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_mul_(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto div(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_div(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto div_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_div_(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto matmul(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_matmul(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto add(double other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_scalar(&g_error, &sca, mag_tensor_context(m_tensor), mag_tensor_type(m_tensor), mag_scalar_from_f64(other), mag_device(CPU, 0)));
            return add(tensor{sca});
        }
        [[nodiscard]] auto sub(double other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_scalar(&g_error, &sca, mag_tensor_context(m_tensor), mag_tensor_type(m_tensor), mag_scalar_from_f64(other), mag_device(CPU, 0)));
            return sub(tensor{sca});
        }
        [[nodiscard]] auto mul(double other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_scalar(&g_error, &sca, mag_tensor_context(m_tensor), mag_tensor_type(m_tensor), mag_scalar_from_f64(other), mag_device(CPU, 0)));
            return mul(tensor{sca});
        }
        [[nodiscard]] auto div(double other) const noexcept -> tensor {
            mag_tensor_t *sca = nullptr;
            handle_error(mag_scalar(&g_error, &sca, mag_tensor_context(m_tensor), mag_tensor_type(m_tensor), mag_scalar_from_f64(other), mag_device(CPU, 0)));
            return div(tensor{sca});
        }
        [[nodiscard]] auto band(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_and(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto band_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_and_(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bor(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_or(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bor_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_or_(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bxor(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_xor(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bxor_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_xor_(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bnot() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_not(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto bnot_() const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_not_(&g_error,  &out, m_tensor));
            return tensor{out};
        }
        [[nodiscard]] auto bshl(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_shl(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bshl_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_shl_(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bshr(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_shr(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        [[nodiscard]] auto bshr_(tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_shr_(&g_error,  &out, m_tensor, &*other));
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
            handle_error(mag_eq(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator != (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_ne(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator <= (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_le(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator >= (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_ge(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator < (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_lt(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }
        auto operator > (tensor other) const noexcept -> tensor {
            mag_tensor_t *out = nullptr;
            handle_error(mag_gt(&g_error,  &out, m_tensor, &*other));
            return tensor{out};
        }

        [[nodiscard]] auto all() const noexcept -> bool {
            mag_tensor_t *result = nullptr;
            handle_error(mag_all(&g_error, &result, m_tensor, nullptr, 0, false));
            mag_scalar_t scalar;
            handle_error(mag_tensor_item(&g_error, result, &scalar));
            mag_tensor_decref(result);
            return !!mag_scalar_as_u64(scalar);
        }

        [[nodiscard]] auto any() const noexcept -> bool {
            mag_tensor_t *result = nullptr;
            handle_error(mag_any(&g_error, &result, m_tensor, nullptr, 0, false));
            mag_scalar_t scalar;
            handle_error(mag_tensor_item(&g_error, result, &scalar));
            mag_tensor_decref(result);
            return !!mag_scalar_as_u64(scalar);
        }

        auto copy_(const void* buf, size_t nb) -> void {
            mag_copy_raw_(&g_error, m_tensor, buf, nb);
        }

        template <typename T>
        auto copy_(const std::vector<T>& data) -> void {
            if (generic_to_dtype<T>() != dtype())
                throw std::runtime_error{"data type does not match tensor dtype"};
            mag_copy_raw_(&g_error, m_tensor, data.data(), data.size()*sizeof(data[0]));
        }

        auto copy_(const std::vector<uint8_t>& data) -> void {
            std::vector<uint8_t> unpacked {};
            unpacked.resize(data.size());
            for (size_t i=0; i < unpacked.size(); ++i) unpacked[i] = data[i];
            mag_copy_raw_(&g_error, m_tensor, unpacked.data(), unpacked.size()*sizeof(data[0]));
        }

        template <typename T>
        auto fill_(T val) -> void {
            static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, bool>);
            if constexpr (std::is_floating_point_v<T>) {
                handle_error(mag_fill_(&g_error, m_tensor, mag_scalar_from_f64(static_cast<double>(val))));
            } else if constexpr (std::is_integral_v<T>) {
                handle_error(mag_fill_(&g_error, m_tensor, mag_scalar_from_i64(static_cast<int64_t>(val))));
            } else {
                throw std::runtime_error{"unsupported type for fill_"};
            }
        }

        template <typename T>
        auto masked_fill_(tensor mask, T val) -> void;

        template <typename T>
        auto uniform_(T min, T max) -> void;

        auto normal_(float mean, float stddev) -> void {
            mag_normal_(&g_error, m_tensor, mag_scalar_from_f64(mean), mag_scalar_from_f64(stddev));
        }

        auto bernoulli_(float p = 0.5f) -> void {
            mag_bernoulli_(&g_error, m_tensor, mag_scalar_from_f64(p));
        }

        [[nodiscard]] auto to_string(int64_t head = 3, int64_t tail = 3, int64_t threshold = 1000) const -> std::string {
            const char* fmt {mag_tensor_to_string(m_tensor, head, tail, threshold)};
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
            void *data = nullptr;
            size_t nb = 0;
            if (mag_iserr(mag_tensor_copy_data(&g_error, m_tensor, &data, &nb)))
                throw std::runtime_error {"failed to copy data from tensor"};
            std::vector<T> result {};
            result.resize(numel());
            std::copy_n(static_cast<const T *>(data), numel(), result.begin());
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
            mag_status_t stat = mag_tensor_grad(nullptr, m_tensor, &grad);
            if (stat != MAG_STATUS_OK) return std::nullopt;
            return tensor{grad};
        }
        [[nodiscard]] auto requires_grad() const noexcept -> bool { return mag_tensor_requires_grad(m_tensor); }
        auto requires_grad(bool yes) noexcept -> void { handle_error(mag_tensor_set_requires_grad(nullptr, m_tensor, yes)); }
        auto backward() -> void { handle_error(mag_tensor_backward(nullptr, m_tensor)); }
        auto zero_grad() -> void { handle_error(mag_tensor_zero_grad(nullptr, m_tensor)); }

        explicit tensor(mag_tensor_t* ptr) noexcept : m_tensor{ptr} {}

    private:
        friend class storage_stream;

        mag_tensor_t* m_tensor {};
    };

    template <>
    inline auto tensor::masked_fill_(tensor mask, float val) -> void {
        if (mask.dtype() != dtype::boolean)
            throw std::runtime_error {"mask must be bool tensor"};
        handle_error(mag_masked_fill_(&g_error, m_tensor, &*mask, mag_scalar_from_f64(val)));
    }

    template <>
    inline auto tensor::masked_fill_(tensor mask, int64_t val) -> void {
        if (mask.dtype() != dtype::boolean)
            throw std::runtime_error {"mask must be bool tensor"};
        handle_error(mag_masked_fill_(&g_error, m_tensor, &*mask, mag_scalar_from_i64(val)));
    }

    template <>
    inline auto tensor::masked_fill_(tensor mask, bool val) -> void {
        if (mask.dtype() != dtype::boolean)
            throw std::runtime_error {"mask must be bool tensor"};
        handle_error(mag_masked_fill_(&g_error, m_tensor, &*mask, mag_scalar_from_i64(val)));
    }

    template <>
    inline auto tensor::uniform_(float min, float max) -> void {
        handle_error(mag_uniform_(&g_error, m_tensor, mag_scalar_from_f64(min), mag_scalar_from_f64(max)));
    }

    template <>
    inline auto tensor::uniform_(int min, int max) -> void {
        handle_error(mag_uniform_(&g_error, m_tensor, mag_scalar_from_i64(min), mag_scalar_from_i64(max)));
    }

    template <>
    inline auto tensor::uniform_(int64_t min, int64_t max) -> void {
        handle_error(mag_uniform_(&g_error, m_tensor, mag_scalar_from_i64(min), mag_scalar_from_i64(max)));
    }
}
