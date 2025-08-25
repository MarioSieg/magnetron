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

// Magnetron Machine Learning Framework, public C++ 20 API.

#pragma once

#include "magnetron.h"

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
    constexpr std::size_t k_max_dims {MAG_MAX_DIMS};
    constexpr std::size_t k_max_input_tensors {MAG_MAX_OP_INPUTS};
    constexpr std::size_t k_max_op_params {MAG_MAX_OP_PARAMS};
    constexpr std::uint16_t k_version {MAG_VERSION};
    constexpr std::uint8_t k_version_major {mag_version_major(MAG_VERSION)};
    constexpr std::uint8_t k_version_minor {mag_version_minor(MAG_VERSION)};

    /**
     * Enumerates compute device types.
     */
    enum class compute_device : std::underlying_type_t<mag_device_type_t> {
        cpu = MAG_DEVICE_TYPE_CPU,
        gpu_cuda = MAG_DEVICE_TYPE_GPU_CUDA
    };

    /**
     * Get enumerator name of compute device.
     * @param device Compute device.
     * @return Enum name of the compute device.
     */
    [[nodiscard]] inline auto compute_device_name(compute_device device) noexcept -> std::string_view {
        return mag_device_type_get_name(static_cast<mag_device_type_t>(device));
    }

    /**
     * Pseudo-random number generator (PRNG) algorithm
     */
    enum class prng_algorithm : std::underlying_type_t<mag_prng_algo_t> {
        /**
         * Merseene Twister 64
         */
        mersenne_twister = MAG_PRNG_MERSENNE_TWISTER,

        /**
         * Permuted Congruential Generator
         */
        pcg = MAG_PRNG_PCG
    };

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

    [[nodiscard]] inline auto pack_color_u8(std::uint8_t r, std::uint8_t g, std::uint8_t b) noexcept -> std::uint32_t {
        return mag_pack_color_u8(r, g, b);
    }

    [[nodiscard]] inline auto pack_color_f32(float r, float g, float b) noexcept -> std::uint32_t {
        return mag_pack_color_f32(r, g, b);
    }

    /**
     * CPU compute device creation info.
     */
    struct cpu_device final {
        /**
         * Amount of threads to use for CPU parallel processing.
         */
        std::uint32_t thread_count {std::max(1u, std::thread::hardware_concurrency())};
    };

    /**
     * CUDA GPU device creation info.
     */
    struct cuda_device final {
        /**
         * GPU index to use. 0 = first GPU, 1 = second etc.
         */
        std::uint32_t device_index {};
    };

    using device_descriptor = std::variant<cpu_device, cuda_device>;

    /**
     * The context owns all tensors and runtime data structures. It must kept alive as long as any tensor is used.
     */
    class context final {
    public:
        explicit context(compute_device dvc) noexcept {
            m_ctx = mag_ctx_create(static_cast<mag_device_type_t>(dvc));
        }

        explicit context(device_descriptor device) {
            mag_device_desc_t desc {};
            if (std::holds_alternative<cpu_device>(device)) {
                const auto& cpu = std::get<cpu_device>(device);
                desc.type = MAG_DEVICE_TYPE_CPU;
                desc.cpu_thread_count = cpu.thread_count;
            } else if (std::holds_alternative<cuda_device>(device)) {
                const auto& cuda = std::get<cuda_device>(device);
                desc.type = MAG_DEVICE_TYPE_GPU_CUDA;
                desc.cuda_device_id = cuda.device_index;
            } else {
                throw std::invalid_argument("Invalid device type");
            }
            m_ctx = mag_ctx_create2(&desc);
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
        [[nodiscard]] auto prng_algorithm() const noexcept -> prng_algorithm { return static_cast<enum prng_algorithm>(mag_ctx_get_prng_algorithm(m_ctx)); }
        auto prng_algorithm(enum prng_algorithm algorithm, std::uint64_t seed) noexcept -> void { mag_ctx_set_prng_algorithm(m_ctx, static_cast<mag_prng_algo_t>(algorithm), seed); }
        [[nodiscard]] auto device_type() const noexcept -> compute_device { return static_cast<compute_device>(mag_ctx_get_compute_device_type(m_ctx)); }
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

    /**
     * A 1-6 dimensional, reference counted tensor with a fixed size and data type.
     */
    class tensor final {
    public:
        tensor(context& ctx, dtype type, std::span<const std::int64_t> shape) {
            m_tensor = mag_tensor_empty(&*ctx, static_cast<mag_dtype_t>(type), shape.size(), shape.data());
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

        [[nodiscard]] auto clone() const noexcept -> tensor { return tensor{mag_clone(m_tensor)}; }
        [[nodiscard]] auto view(std::initializer_list<std::int64_t> dims = {}) const noexcept -> tensor {
            return tensor{mag_view(m_tensor, std::empty(dims) ? nullptr : std::data(dims), std::size(dims))};
        }
        [[nodiscard]] auto reshape(std::initializer_list<std::int64_t> dims = {}) const noexcept -> tensor {
            return tensor{mag_reshape(m_tensor, std::data(dims), std::size(dims))};
        }
        [[nodiscard]] auto view_slice(std::int64_t dim, std::int64_t start, std::int64_t len, std::int64_t step) -> tensor { return tensor {mag_view_slice(m_tensor, dim, start, len, step)}; }
        [[nodiscard]] auto T(std::int64_t dim1 = 0, std::int64_t dim2 = 1) const noexcept -> tensor { return tensor{mag_transpose(m_tensor, dim1, dim2)}; }
        [[nodiscard]] auto transpose(std::int64_t dim1 = 0, std::int64_t dim2 = 1) const noexcept -> tensor { return tensor{mag_transpose(m_tensor, dim1, dim2)}; }
        [[nodiscard]] auto permute(const std::array<std::int64_t, k_max_dims>& axes) const noexcept -> tensor { return tensor{mag_permute(m_tensor, axes.data(), axes.size())}; }
        [[nodiscard]] auto mean() const noexcept -> tensor { return tensor{mag_mean(m_tensor, nullptr, 0, false)}; }
        [[nodiscard]] auto min() const noexcept -> tensor { return tensor{mag_min(m_tensor, nullptr, 0, false)}; }
        [[nodiscard]] auto max() const noexcept -> tensor { return tensor{mag_max(m_tensor, nullptr, 0, false)}; }
        [[nodiscard]] auto sum() const noexcept -> tensor { return tensor{mag_sum(m_tensor, nullptr, 0, false)}; }
        [[nodiscard]] auto argmin() const noexcept -> tensor { return tensor{mag_argmin(m_tensor, nullptr, 0, false)}; }
        [[nodiscard]] auto argmax() const noexcept -> tensor { return tensor{mag_argmax(m_tensor, nullptr, 0, false)}; }
        [[nodiscard]] auto abs() const noexcept -> tensor { return tensor{mag_abs(m_tensor)}; }
        [[nodiscard]] auto abs_() const noexcept -> tensor { return tensor{mag_abs_(m_tensor)}; }
        [[nodiscard]] auto sgn() const noexcept -> tensor { return tensor{mag_sgn(m_tensor)}; }
        [[nodiscard]] auto sgn_() const noexcept -> tensor { return tensor{mag_sgn_(m_tensor)}; }
        [[nodiscard]] auto neg() const noexcept -> tensor { return tensor{mag_neg(m_tensor)}; }
        [[nodiscard]] auto neg_() const noexcept -> tensor { return tensor{mag_neg_(m_tensor)}; }
        [[nodiscard]] auto log() const noexcept -> tensor { return tensor{mag_log(m_tensor)}; }
        [[nodiscard]] auto log_() const noexcept -> tensor { return tensor{mag_log_(m_tensor)}; }
        [[nodiscard]] auto sqr() const noexcept -> tensor { return tensor{mag_sqr(m_tensor)}; }
        [[nodiscard]] auto sqr_() const noexcept -> tensor { return tensor{mag_sqr_(m_tensor)}; }
        [[nodiscard]] auto sqrt() const noexcept -> tensor { return tensor{mag_sqrt(m_tensor)}; }
        [[nodiscard]] auto sqrt_() const noexcept -> tensor { return tensor{mag_sqrt_(m_tensor)}; }
        [[nodiscard]] auto sin() const noexcept -> tensor { return tensor{mag_sin(m_tensor)}; }
        [[nodiscard]] auto sin_() const noexcept -> tensor { return tensor{mag_sin_(m_tensor)}; }
        [[nodiscard]] auto cos() const noexcept -> tensor { return tensor{mag_cos(m_tensor)}; }
        [[nodiscard]] auto cos_() const noexcept -> tensor { return tensor{mag_cos_(m_tensor)}; }
        [[nodiscard]] auto step() const noexcept -> tensor { return tensor{mag_step(m_tensor)}; }
        [[nodiscard]] auto step_() const noexcept -> tensor { return tensor{mag_step_(m_tensor)}; }
        [[nodiscard]] auto exp() const noexcept -> tensor { return tensor{mag_exp(m_tensor)}; }
        [[nodiscard]] auto exp_() const noexcept -> tensor { return tensor{mag_exp_(m_tensor)}; }
        [[nodiscard]] auto floor() const noexcept -> tensor { return tensor{mag_floor(m_tensor)}; }
        [[nodiscard]] auto floor_() const noexcept -> tensor { return tensor{mag_floor_(m_tensor)}; }
        [[nodiscard]] auto ceil() const noexcept -> tensor { return tensor{mag_ceil(m_tensor)}; }
        [[nodiscard]] auto ceil_() const noexcept -> tensor { return tensor{mag_ceil_(m_tensor)}; }
        [[nodiscard]] auto round() const noexcept -> tensor { return tensor{mag_round(m_tensor)}; }
        [[nodiscard]] auto round_() const noexcept -> tensor { return tensor{mag_round_(m_tensor)}; }
        [[nodiscard]] auto softmax() const noexcept -> tensor { return tensor{mag_softmax(m_tensor)}; }
        [[nodiscard]] auto softmax_() const noexcept -> tensor { return tensor{mag_softmax_(m_tensor)}; }
        [[nodiscard]] auto sigmoid() const noexcept -> tensor { return tensor{mag_sigmoid(m_tensor)}; }
        [[nodiscard]] auto sigmoid_() const noexcept -> tensor { return tensor{mag_sigmoid_(m_tensor)}; }
        [[nodiscard]] auto hard_sigmoid() const noexcept -> tensor { return tensor{mag_hard_sigmoid(m_tensor)}; }
        [[nodiscard]] auto hard_sigmoid_() const noexcept -> tensor { return tensor{mag_hard_sigmoid_(m_tensor)}; }
        [[nodiscard]] auto silu() const noexcept -> tensor { return tensor{mag_silu(m_tensor)}; }
        [[nodiscard]] auto silu_() const noexcept -> tensor { return tensor{mag_silu_(m_tensor)}; }
        [[nodiscard]] auto tanh() const noexcept -> tensor { return tensor{mag_tanh(m_tensor)}; }
        [[nodiscard]] auto tanh_() const noexcept -> tensor { return tensor{mag_tanh_(m_tensor)}; }
        [[nodiscard]] auto relu() const noexcept -> tensor { return tensor{mag_relu(m_tensor)}; }
        [[nodiscard]] auto relu_() const noexcept -> tensor { return tensor{mag_relu_(m_tensor)}; }
        [[nodiscard]] auto gelu() const noexcept -> tensor { return tensor{mag_gelu(m_tensor)}; }
        [[nodiscard]] auto gelu_() const noexcept -> tensor { return tensor{mag_gelu_(m_tensor)}; }
        [[nodiscard]] auto gelu_approx() const noexcept -> tensor { return tensor{mag_gelu_approx(m_tensor)}; }
        [[nodiscard]] auto gelu_approx_() const noexcept -> tensor { return tensor{mag_gelu_approx_(m_tensor)}; }
        [[nodiscard]] auto add(tensor other) const noexcept -> tensor {return tensor{mag_add(m_tensor, &*other)}; }
        [[nodiscard]] auto add_(tensor other) const noexcept -> tensor { return tensor{mag_add_(m_tensor, &*other)}; }
        [[nodiscard]] auto sub(tensor other) const noexcept -> tensor { return tensor{mag_sub(m_tensor, &*other)}; }
        [[nodiscard]] auto sub_(tensor other) const noexcept -> tensor { return tensor{mag_sub_(m_tensor, &*other)}; }
        [[nodiscard]] auto mul(tensor other) const noexcept -> tensor { return tensor{mag_mul(m_tensor, &*other)}; }
        [[nodiscard]] auto mul_(tensor other) const noexcept -> tensor { return tensor{mag_mul_(m_tensor, &*other)}; }
        [[nodiscard]] auto div(tensor other) const noexcept -> tensor { return tensor{mag_div(m_tensor, &*other)}; }
        [[nodiscard]] auto div_(tensor other) const noexcept -> tensor { return tensor{mag_div_(m_tensor, &*other)}; }
        [[nodiscard]] auto matmul(tensor other) const noexcept -> tensor { return tensor{mag_matmul(m_tensor, &*other)}; }
        [[nodiscard]] auto add(float other) const noexcept -> tensor {
            tensor sca {mag_tensor_scalar(mag_tensor_get_ctx(m_tensor), mag_tensor_get_dtype(m_tensor), other)};
            return add(sca);
        }
        [[nodiscard]] auto sub(float other) const noexcept -> tensor {
            tensor sca {mag_tensor_scalar(mag_tensor_get_ctx(m_tensor), mag_tensor_get_dtype(m_tensor), other)};
            return sub(sca);
        }
        [[nodiscard]] auto mul(float other) const noexcept -> tensor {
            tensor sca {mag_tensor_scalar(mag_tensor_get_ctx(m_tensor), mag_tensor_get_dtype(m_tensor), other)};
            return mul(sca);
        }
        [[nodiscard]] auto div(float other) const noexcept -> tensor {
            tensor sca {mag_tensor_scalar(mag_tensor_get_ctx(m_tensor), mag_tensor_get_dtype(m_tensor), other)};
            return div(sca);
        }
        [[nodiscard]] auto band(tensor other) const noexcept -> tensor {return tensor{mag_and(m_tensor, &*other)}; }
        [[nodiscard]] auto band_(tensor other) const noexcept -> tensor { return tensor{mag_and_(m_tensor, &*other)}; }
        [[nodiscard]] auto bor(tensor other) const noexcept -> tensor {return tensor{mag_or(m_tensor, &*other)}; }
        [[nodiscard]] auto bor_(tensor other) const noexcept -> tensor { return tensor{mag_or_(m_tensor, &*other)}; }
        [[nodiscard]] auto bxor(tensor other) const noexcept -> tensor {return tensor{mag_xor(m_tensor, &*other)}; }
        [[nodiscard]] auto bxor_(tensor other) const noexcept -> tensor { return tensor{mag_xor_(m_tensor, &*other)}; }
        [[nodiscard]] auto bnot() const noexcept -> tensor { return tensor{mag_not(m_tensor)}; }
        [[nodiscard]] auto bnot_() const noexcept -> tensor { return tensor{mag_not_(m_tensor)}; }
        [[nodiscard]] auto bshl(tensor other) const noexcept -> tensor {return tensor{mag_shl(m_tensor, &*other)}; }
        [[nodiscard]] auto bshl_(tensor other) const noexcept -> tensor { return tensor{mag_shl_(m_tensor, &*other)}; }
        [[nodiscard]] auto bshr(tensor other) const noexcept -> tensor {return tensor{mag_shr(m_tensor, &*other)}; }
        [[nodiscard]] auto bshr_(tensor other) const noexcept -> tensor { return tensor{mag_shr_(m_tensor, &*other)}; }

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
            return tensor{mag_eq(m_tensor, &*other)};
        }
        auto operator != (tensor other) const noexcept -> tensor {
            return tensor{mag_ne(m_tensor, &*other)};
        }
        auto operator <= (tensor other) const noexcept -> tensor {
            return tensor{mag_le(m_tensor, &*other)};
        }
        auto operator >= (tensor other) const noexcept -> tensor {
            return tensor{mag_ge(m_tensor, &*other)};
        }
        auto operator < (tensor other) const noexcept -> tensor {
            return tensor{mag_lt(m_tensor, &*other)};
        }
        auto operator > (tensor other) const noexcept -> tensor {
            return tensor{mag_gt(m_tensor, &*other)};
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

        auto fill_float(float val) -> void {
            mag_tensor_fill_float(m_tensor, val);
        }

        auto fill_int(std::int32_t val) -> void {
            mag_tensor_fill_int(m_tensor, val);
        }

        auto fill_rand_uniform_float(float min, float max) -> void {
            mag_tensor_fill_random_uniform_float(m_tensor, min, max);
        }

        auto fill_rand_uniform_int(std::int32_t min, std::int32_t max) -> void {
            mag_tensor_fill_random_uniform_int(m_tensor, min, max);
        }

        auto fill_rand_normal(float mean, float stddev) -> void {
            mag_tensor_fill_random_normal(m_tensor, mean, stddev);
        }

        auto fill_rand_bernoulli(float p = 0.5f) -> void {
            mag_tensor_fill_random_bernoulli(m_tensor, p);
        }

        [[nodiscard]] auto refcount() const noexcept -> std::uint64_t { return mag_tensor_get_refcount(m_tensor); }
        [[nodiscard]] auto memory_usage() const noexcept -> std::size_t { return mag_tensor_get_memory_usage(m_tensor); }
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
        [[nodiscard]] auto to_float_vector() const -> std::vector<float> {
            if (!is_floating_point_typed())
                throw std::runtime_error {"requires floating point dtype"};
            auto* data {mag_tensor_get_data_as_floats(m_tensor)};
            std::vector<float> result {};
            result.resize(numel());
            std::copy_n(data, numel(), result.begin());
            mag_tensor_get_data_as_floats_free(data);
            return result;
        }
        [[nodiscard]] auto to_bool_vector() const -> std::vector<bool> {
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
        [[nodiscard]] auto to_int_vector() const -> std::vector<std::int32_t> {
            if (dtype() != dtype::i32)
                throw std::runtime_error {"requires int32 dtype"};
            auto* data {static_cast<std::int32_t*>(mag_tensor_get_raw_data_as_bytes(m_tensor))};
            std::vector<std::int32_t> result {};
            result.resize(numel());
            std::memcpy(result.data(), data, data_size());
            mag_tensor_get_raw_data_as_bytes_free(data);
            return result;
        }
        [[nodiscard]] auto data_size() const noexcept -> std::int64_t { return mag_tensor_get_data_size(m_tensor); }
        [[nodiscard]] auto numel() const noexcept -> std::int64_t { return mag_tensor_get_numel(m_tensor); }
        [[nodiscard]] auto is_shape_eq(tensor other) const noexcept -> bool { return mag_tensor_is_shape_eq(m_tensor, &*other); }
        [[nodiscard]] auto can_broadcast(tensor other) const noexcept -> bool { return mag_tensor_can_broadcast(m_tensor, &*other); }
        [[nodiscard]] auto is_transposed() const noexcept -> bool { return mag_tensor_is_transposed(m_tensor); }
        [[nodiscard]] auto is_permuted() const noexcept -> bool { return mag_tensor_is_permuted(m_tensor); }
        [[nodiscard]] auto is_contiguous() const noexcept -> bool { return mag_tensor_is_contiguous(m_tensor); }
        [[nodiscard]] auto is_view() const noexcept -> bool { return mag_tensor_is_view(m_tensor); }
        [[nodiscard]] auto is_floating_point_typed() const noexcept -> bool { return mag_tensor_is_floating_point_typed(m_tensor); }
        [[nodiscard]] auto is_integral_point_typed() const noexcept -> bool { return mag_tensor_is_integral_typed(m_tensor); }

        [[nodiscard]] auto grad() const noexcept -> std::optional<tensor> {
            auto* grad {mag_tensor_get_grad(m_tensor)};
            if (!grad) return std::nullopt;
            return tensor{grad};
        }
        [[nodiscard]] auto requires_grad() const noexcept -> bool { return mag_tensor_requires_grad(m_tensor); }
        auto requires_grad(bool yes) noexcept -> void { mag_tensor_set_requires_grad(m_tensor, yes); }
        auto backward() -> void { mag_tensor_backward(m_tensor); }
        auto zero_grad() -> void { mag_tensor_zero_grad(m_tensor); }
        auto export_graphviz_forward(const std::string& filename) const -> void {
            mag_tensor_export_forward_graph_graphviz(m_tensor, filename.c_str());
        }
        auto export_graphviz_backward(const std::string& filename) const -> void {
            mag_tensor_export_backward_graph_graphviz(m_tensor, filename.c_str());
        }

        [[nodiscard]] auto operator ()(const std::array<std::int64_t, k_max_dims>& idx) const noexcept -> float {
            return mag_tensor_subscript_get_multi(m_tensor, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
        }
        auto operator ()(const std::array<std::int64_t, k_max_dims>& idx, float x) const noexcept -> void {
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
}
