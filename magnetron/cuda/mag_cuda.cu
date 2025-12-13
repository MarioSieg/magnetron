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

#include "mag_cuda.cuh"
#include "mag_cuda_unary.cuh"
#include "mag_cuda_binary.cuh"
#include "mag_cuda_fill.cuh"
#include "mag_cuda_reduction.cuh"

#include "cpu/mag_cpu.h"

#include <core/mag_alloc.h>

#include <array>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mag {
    #define mag_cuda_check(expr) \
        do { \
            if (auto result = (expr); mag_unlikely(result != cudaSuccess)) { \
                mag_panic("%s:%d CUDA error: " #expr " <- %s", __FILE__, __LINE__, cudaGetErrorString(result)); \
            } \
        } while (0)

    struct physical_device final {
        int id = 0;
        std::array<char, 256> name = {};
        size_t vram = 0;
        uint32_t cl = 0;
        uint32_t nsm = 0;
        uint32_t ntpb = 0;
        size_t smpb = 0;
        size_t smpb_opt = 0;
        bool has_vmm = false;
        size_t vmm_granularity = 0;

        [[nodiscard]] static std::optional<physical_device> query_from_idx(int idx) {
            CUdevice cu_dvc = 0;
            if (cuDeviceGet(&cu_dvc, idx) != CUDA_SUCCESS)
                return std::nullopt;
            int vmm_support = 0;
            if (cuDeviceGetAttribute(&vmm_support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_dvc) != CUDA_SUCCESS)
                return std::nullopt;
            size_t vmm_granularity = 0;
            if (vmm_support) {
                CUmemAllocationProp alloc_props {};
                alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                alloc_props.location.id = idx;
                if (cuMemGetAllocationGranularity(&vmm_granularity, &alloc_props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS)
                    return std::nullopt;
            }
            cudaDeviceProp props = {};
            if (cudaGetDeviceProperties(&props, idx) != cudaSuccess)
                return std::nullopt;
            physical_device device = {
                .id = idx,
                .name = {},
                .vram = props.totalGlobalMem,
                .cl = static_cast<uint32_t>(100*props.major + 10*props.minor),
                .nsm = static_cast<uint32_t>(props.multiProcessorCount),
                .ntpb = static_cast<uint32_t>(props.maxThreadsPerBlock),
                .smpb = props.sharedMemPerBlock,
                .smpb_opt = props.sharedMemPerBlockOptin,
                .has_vmm = !!vmm_support,
                .vmm_granularity = vmm_granularity,
            };
            std::snprintf(device.name.data(), device.name.size(), "%s", props.name);
            return device;
        }
    };

    struct backend_impl final {
        [[nodiscard]] const physical_device &active_device() const { return m_phys_devices.at(m_active_dvc); }
        [[nodiscard]] const std::vector<physical_device> &devices() const noexcept { return m_phys_devices; }

        explicit backend_impl(int ngpus) {
            m_phys_devices.reserve(ngpus);
            for (int i=0; i < ngpus; ++i) {
                if (std::optional<physical_device> dvc = physical_device::query_from_idx(i); dvc) {
                    m_phys_devices.emplace_back(*dvc);
                    mag_log_info("Found device %d: %s (CL %u, %.01f GiB VRAM)", i, dvc->name.data(), dvc->cl, static_cast<double>(dvc->vram)/static_cast<double>(1ull<<30));
                }
            }
        }

    private:
        size_t m_active_dvc = 0;
        size_t m_best_dvc = 0;
        std::vector<physical_device> m_phys_devices = {};
    };

    static void manual_seed(mag_device_t *dvc, uint64_t seed) {
        global_seed.store(seed, std::memory_order_relaxed);
    }

    using kernel_fn = void (const mag_command_t *);

    static void op_nop(const mag_command_t *) { }

    static void submit(mag_device_t *dvc, const mag_command_t *cmd) {
        static constexpr kernel_fn *dispatch_table[] = {
            [MAG_OP_NOP] = &op_nop,
            [MAG_OP_FILL] = &fill_op_fill,
            [MAG_OP_MASKED_FILL] = &fill_op_masked_fill,
            [MAG_OP_RAND_UNIFORM] = &fill_op_fill_rand_uniform,
            [MAG_OP_RAND_NORMAL] = &fill_op_fill_rand_normal,
            [MAG_OP_RAND_BERNOULLI] = nullptr,
            [MAG_OP_RAND_PERM] = nullptr,
            [MAG_OP_ARANGE] = nullptr,
            [MAG_OP_ONE_HOT] = nullptr,
            [MAG_OP_CLONE] = &unary_op_clone,
            [MAG_OP_CAST] = &unary_op_cast,
            [MAG_OP_VIEW] = &op_nop,
            [MAG_OP_TRANSPOSE] = &op_nop,
            [MAG_OP_PERMUTE] = &op_nop,
            [MAG_OP_MEAN] = &reduce_op_mean,
            [MAG_OP_MIN] = &reduce_op_min,
            [MAG_OP_MAX] = &reduce_op_max,
            [MAG_OP_ARGMIN] = nullptr,
            [MAG_OP_ARGMAX] = nullptr,
            [MAG_OP_SUM] = &reduce_op_sum,
            [MAG_OP_PROD] = &reduce_op_prod,
            [MAG_OP_ALL] = nullptr,
            [MAG_OP_ANY] = nullptr,
            [MAG_OP_TOPK] = nullptr,
            [MAG_OP_ABS] = &unary_op_abs,
            [MAG_OP_SGN] = &unary_op_sgn,
            [MAG_OP_NEG] = &unary_op_neg,
            [MAG_OP_LOG] = &unary_op_log,
            [MAG_OP_LOG10] = &unary_op_log10,
            [MAG_OP_LOG1P] = &unary_op_log1p,
            [MAG_OP_LOG2] = &unary_op_log2,
            [MAG_OP_SQR] = &unary_op_sqr,
            [MAG_OP_RCP] = &unary_op_rcp,
            [MAG_OP_SQRT] = &unary_op_sqrt,
            [MAG_OP_RSQRT] = &unary_op_rsqrt,
            [MAG_OP_SIN] = &unary_op_sin,
            [MAG_OP_COS] = &unary_op_cos,
            [MAG_OP_TAN] = &unary_op_tan,
            [MAG_OP_SINH] = &unary_op_sinh,
            [MAG_OP_COSH] = &unary_op_cosh,
            [MAG_OP_TANH] = &unary_op_tanh,
            [MAG_OP_ASIN] = &unary_op_asin,
            [MAG_OP_ACOS] = &unary_op_acos,
            [MAG_OP_ATAN] = &unary_op_atan,
            [MAG_OP_ASINH] = &unary_op_asinh,
            [MAG_OP_ACOSH] = &unary_op_acosh,
            [MAG_OP_ATANH] = &unary_op_atanh,
            [MAG_OP_STEP] = &unary_op_step,
            [MAG_OP_ERF] = &unary_op_erf,
            [MAG_OP_ERFC] = &unary_op_erfc,
            [MAG_OP_EXP] = &unary_op_exp,
            [MAG_OP_EXP2] = &unary_op_exp2,
            [MAG_OP_EXPM1] = &unary_op_expm1,
            [MAG_OP_FLOOR] = &unary_op_floor,
            [MAG_OP_CEIL] = &unary_op_ceil,
            [MAG_OP_ROUND] = &unary_op_round,
            [MAG_OP_TRUNC] = &unary_op_trunc,
            [MAG_OP_SOFTMAX] = &unary_op_softmax,
            [MAG_OP_SOFTMAX_DV] = &unary_op_softmax_dv,
            [MAG_OP_SIGMOID] = &unary_op_sigmoid,
            [MAG_OP_SIGMOID_DV] = &unary_op_sigmoid_dv,
            [MAG_OP_HARD_SIGMOID] = &unary_op_hard_sigmoid,
            [MAG_OP_SILU] = &unary_op_silu,
            [MAG_OP_SILU_DV] = &unary_op_silu_dv,
            [MAG_OP_TANH_DV] = &unary_op_tanh_dv,
            [MAG_OP_RELU] = &unary_op_relu,
            [MAG_OP_RELU_DV] = &unary_op_relu_dv,
            [MAG_OP_GELU] = &unary_op_gelu,
            [MAG_OP_GELU_APPROX] = &unary_op_gelu,
            [MAG_OP_GELU_DV] = &unary_op_gelu_dv,
            [MAG_OP_TRIL] = nullptr,
            [MAG_OP_TRIU] = nullptr,
            [MAG_OP_MULTINOMIAL] = nullptr,
            [MAG_OP_CAT] = nullptr,
            [MAG_OP_ADD] = &binary_op_add,
            [MAG_OP_SUB] = &binary_op_sub,
            [MAG_OP_MUL] = &binary_op_mul,
            [MAG_OP_DIV] = &binary_op_div,
            [MAG_OP_FLOORDIV] = nullptr,
            [MAG_OP_MOD] = &binary_op_mod,
            [MAG_OP_MATMUL] = nullptr,
            [MAG_OP_REPEAT_BACK] = nullptr,
            [MAG_OP_GATHER] = nullptr,
            [MAG_OP_AND] = &binary_op_and,
            [MAG_OP_OR] = &binary_op_or,
            [MAG_OP_XOR] = &binary_op_xor,
            [MAG_OP_NOT] = nullptr,
            [MAG_OP_SHL] = &binary_op_shl,
            [MAG_OP_SHR] = &binary_op_shr,
            [MAG_OP_EQ] = &binary_op_eq,
            [MAG_OP_NE] = &binary_op_ne,
            [MAG_OP_LE] = &binary_op_le,
            [MAG_OP_GE] = &binary_op_ge,
            [MAG_OP_LT] = &binary_op_lt,
            [MAG_OP_GT] = &binary_op_gt
        };
        static_assert(std::size(dispatch_table) == MAG_OP__NUM, "Dispatch table size mismatch");
        kernel_fn *kern = dispatch_table[cmd->op];
        mag_assert(kern != nullptr, "Operator %s not implemented in CUDA backend", mag_op_traits(cmd->op)->mnemonic);
        (*kern)(cmd);
    }

    static void alloc_storage_buffer(mag_device_t *device, mag_storage_buffer_t **out, size_t size, mag_dtype_t dtype) {
        mag_context_t *ctx = device->ctx;
        uintptr_t base;
        mag_cuda_check(cudaMalloc(reinterpret_cast<void **>(&base), size));
        *out = static_cast<mag_storage_buffer_t*>(mag_fixed_pool_alloc_block(&ctx->storage_pool));
        new (*out) mag_storage_buffer_t {
            .__rcb = {},
            .ctx = ctx,
            .aux = {},
            .flags = MAG_STORAGE_FLAG_ACCESS_W,
            .base = base,
            .size = size,
            .alignment = 256, // cudaMalloc guarantees this
            .granularity = mag_type_trait(dtype)->size,
            .dtype = dtype,
            .device = device,
        };
        static constexpr auto *dealloc_callback = +[](void *self) {
            auto *buffer = static_cast<mag_storage_buffer_t *>(self);
            mag_context_t *ctx = buffer->ctx;
            mag_cuda_check(cudaFree(reinterpret_cast<void *>(buffer->base)));
            mag_fixed_pool_free_block(&ctx->storage_pool, buffer);
        };
        mag_rc_init_object(*out, dealloc_callback);
    }

    mag_device_t *mag_cuda_backend_init_device(mag_backend_t *bck, mag_context_t *ctx, uint32_t idx) {
        auto *impl = static_cast<backend_impl *>(bck->impl);
        if (idx >= impl->devices().size()) {
            mag_log_error("Invalid device index %u (max %zu)", idx, impl->devices().size()-1);
            return nullptr;
        }
        const physical_device &phys_device = impl->devices()[idx];
        auto *device = new mag_device_t {
            .ctx = ctx,
            .impl = nullptr,
            .is_async = false,
            .physical_device_name = "",
            .id = "",
            .submit = &submit,
            .alloc_storage = &alloc_storage_buffer,
            .manual_seed = &manual_seed
        };
        std::snprintf(device->id, sizeof(device->id), "cuda:%u", idx);
        std::snprintf(device->physical_device_name, sizeof(device->physical_device_name), "%s", phys_device.name.data());
        return device;
    }
    void mag_cuda_backend_destroy_device(mag_backend_t *bck, mag_device_t *dvc) {
        delete dvc;
    }

    [[nodiscard]] static mag_backend_t *backend_create(int ngpus) {
        return new mag_backend_t {
            .backend_version = +[](mag_backend_t *bck) noexcept -> uint32_t { return MAG_CUDA_BACKEND_VERSION; },
            .runtime_version = +[](mag_backend_t *bck) noexcept -> uint32_t { return MAG_VERSION; },
            .score = +[](mag_backend_t *bck) noexcept -> uint32_t { return 0; }, // TODO: fix score to reflect actual performance
            .id = +[](mag_backend_t *bck) noexcept -> const char* { return "cuda"; },
            .num_devices = +[](mag_backend_t *bck) noexcept -> uint32_t { return static_cast<backend_impl *>(bck->impl)->devices().size(); },
            .best_device_idx = +[](mag_backend_t *bck) noexcept -> uint32_t { return 0; },
            .init_device = &mag_cuda_backend_init_device,
            .destroy_device = &mag_cuda_backend_destroy_device,
            .impl = new backend_impl{ngpus},
        };
    }

    static void backend_destroy(mag_backend_t *backend) {
        delete static_cast<backend_impl *>(backend->impl);
        delete backend;
    }
}

uint32_t MAG_BACKEND_SYM_ABI_COOKIE(){
    return mag_pack_abi_cookie('M', 'A', 'G', MAG_BACKEND_MODULE_ABI_VER);
}

mag_backend_t *MAG_BACKEND_SYM_INIT(mag_context_t *ctx)
try {
    int ngpus = 0;
    if (cudaGetDeviceCount(&ngpus) != cudaSuccess || ngpus <= 0) { // No GPUs found, backend cannot be used
        mag_log_error("No CUDA-capable devices found.");
        return nullptr;
    }
    return mag::backend_create(ngpus);
} catch (const std::exception &e) {
    mag_log_error("Error during backend initialization: %s", e.what());
    return nullptr;
} catch (...) {
    mag_log_error("Unknown error during backend initialization.");
    return nullptr;
}

void MAG_BACKEND_SYM_SHUTDOWN(mag_backend_t *backend)
try {
    mag::backend_destroy(backend);
} catch (const std::exception &e) {
    mag_log_error("Error during backend shutdown: %s", e.what());
} catch (...) {
    mag_log_error("Unknown error during backend shutdown.");
}
