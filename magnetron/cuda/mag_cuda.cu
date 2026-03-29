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

#include "mag_cuda.cuh"
#include "mag_cuda_unary.cuh"
#include "mag_cuda_binary.cuh"
#include "mag_cuda_fill.cuh"
#include "mag_cuda_reduction.cuh"

#include "cpu/mag_cpu.h"

#include <core/mag_alloc.h>

#include <array>
#include <cstdio>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mag {
    #define mag_cuda_check(expr) \
        do { \
            if (auto result = (expr); mag_unlikely(result != cudaSuccess)) { \
                mag_panic("%s:%d CUDA error: " #expr " <- %s", __FILE__, __LINE__, cudaGetErrorString(result)); \
            } \
        } while (0)

    class cuda_backend_error : public std::runtime_error {
    private:
        [[nodiscard]] static std::string fmt_error_message(CUresult code, const char *what) {
            std::stringstream ss;
            ss << "CUDA error 0x" << std::hex << code << " (" << code << ") - " << what;
            return ss.str();
        }

        [[nodiscard]] static std::string fmt_error_message(cudaError_t code, const char *what) {
            std::stringstream ss;
            ss << "CUDA error " << cudaGetErrorString(code) << " (" << code << ") - " << what;
            return ss.str();
        }

    public:
        explicit cuda_backend_error(CUresult code, const char *what) : std::runtime_error {fmt_error_message(code, what)} { }
        explicit cuda_backend_error(cudaError_t code, const char *what) : std::runtime_error {fmt_error_message(code, what)} { }
    };

    static void manual_seed(mag_device_t *dvc, uint64_t seed) {
        global_seed.store(seed, std::memory_order_relaxed);
    }

    [[nodiscard]] static mag_status_t cuda_transfer(mag_device_t *dvc, mag_error_t *err, mag_transfer_dir_t dir, mag_tensor_t *src, mag_tensor_t *dst) {
        const size_t nb = mag_tensor_numbytes(src);
        mag_contract(err, ERR_INVALID_PARAM, {}, nb == mag_tensor_numbytes(dst), "Transfer: source and destination byte sizes differ.");
        mag_contract(err, ERR_INVALID_PARAM, {}, mag_tensor_is_contiguous(src) && mag_tensor_is_contiguous(dst), "Transfer requires contiguous tensors.");

        const int my_dev = static_cast<int>(dvc->id.device_ordinal);

        auto report_cuda = [err](cudaError_t ce, const char *what) -> mag_status_t {
            if (err && err->code == MAG_STATUS_OK) {
                *err = mag_error_t{
                    .code = MAG_STATUS_ERR_UNKNOWN,
                    .message = "",
                    .file = __FILE__,
                    .line = __LINE__,
                    .func = __func__,
                };
                snprintf(err->message, sizeof err->message, "%s: %s", what, cudaGetErrorString(ce));
            }
            return MAG_STATUS_ERR_UNKNOWN;
        };

        switch (dir) {
        case MAG_TRANSFER_DIR_H2D: {
            mag_contract(err, ERR_INVALID_PARAM, {}, src->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE, "H2D: source storage must be host-visible.");
            mag_contract(err, ERR_INVALID_PARAM, {}, !(dst->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE), "H2D: destination storage must be device-local.");
            mag_contract(err, ERR_INVALID_PARAM, {}, dst->storage->device == dvc, "H2D: destination device mismatch.");
            cudaError_t ce = cudaSetDevice(my_dev);
            if (mag_unlikely(ce != cudaSuccess))
                return report_cuda(ce, "cudaSetDevice (H2D)");
            ce = cudaMemcpy(reinterpret_cast<void *>(mag_tensor_data_ptr_mut(dst)), reinterpret_cast<const void *>(mag_tensor_data_ptr(src)), nb, cudaMemcpyHostToDevice);
            if (mag_unlikely(ce != cudaSuccess))
                return report_cuda(ce, "cudaMemcpy H2D");
            return MAG_STATUS_OK;
        }
        case MAG_TRANSFER_DIR_D2H: {
            mag_contract(err, ERR_INVALID_PARAM, {}, !(src->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE), "D2H: source storage must be device-local.");
            mag_contract(err, ERR_INVALID_PARAM, {}, src->storage->device == dvc, "D2H: source device mismatch.");
            mag_contract(err, ERR_INVALID_PARAM, {}, dst->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE, "D2H: destination storage must be host-visible.");
            cudaError_t ce = cudaSetDevice(my_dev);
            if (mag_unlikely(ce != cudaSuccess))
                return report_cuda(ce, "cudaSetDevice (D2H)");
            ce = cudaMemcpy(reinterpret_cast<void *>(mag_tensor_data_ptr_mut(dst)), reinterpret_cast<const void *>(mag_tensor_data_ptr(src)), nb, cudaMemcpyDeviceToHost);
            if (mag_unlikely(ce != cudaSuccess))
                return report_cuda(ce, "cudaMemcpy D2H");
            return MAG_STATUS_OK;
        }
        case MAG_TRANSFER_DIR_D2D: {
            mag_contract(err, ERR_INVALID_PARAM, {}, !(src->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE) && !(dst->storage->flags & MAG_STORAGE_FLAG_HOST_VISIBLE), "D2D: both storages must be device-local.");
            const int src_ord = static_cast<int>(src->storage->device->id.device_ordinal);
            const int dst_ord = static_cast<int>(dst->storage->device->id.device_ordinal);
            mag_contract(err, ERR_INVALID_PARAM, {}, dst->storage->device == dvc, "D2D: destination device mismatch.");
            cudaError_t ce = cudaSetDevice(dst_ord);
            if (mag_unlikely(ce != cudaSuccess))
                return report_cuda(ce, "cudaSetDevice (D2D)");
            void *dp = reinterpret_cast<void *>(mag_tensor_data_ptr_mut(dst));
            const void *sp = reinterpret_cast<const void *>(mag_tensor_data_ptr(src));
            ce = cudaMemcpyPeer(dp, dst_ord, sp, src_ord, nb);
            if (mag_unlikely(ce != cudaSuccess))
                return report_cuda(ce, "cudaMemcpyPeer");
            return MAG_STATUS_OK;
        }
        default:
            mag_contract(err, ERR_INVALID_PARAM, {}, false, "Invalid transfer direction.");
        }
    }

    [[nodiscard]] static mag_status_t submit(mag_device_t *dvc, const mag_command_t *cmd) {
        static constexpr auto *op_nop = +[](const mag_command_t &) -> void { };

        static constexpr void(*dispatch_table[])(const mag_command_t &) = {
            [MAG_OP_NOP] = op_nop,
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
            [MAG_OP_VIEW] = op_nop,
            [MAG_OP_TRANSPOSE] = op_nop,
            [MAG_OP_PERMUTE] = op_nop,
            [MAG_OP_MEAN] = &reduce_op_mean,
            [MAG_OP_MIN] = &reduce_op_min,
            [MAG_OP_MAX] = &reduce_op_max,
            [MAG_OP_ARGMIN] = nullptr,
            [MAG_OP_ARGMAX] = nullptr,
            [MAG_OP_SUM] = &reduce_op_sum,
            [MAG_OP_PROD] = &reduce_op_prod,
            [MAG_OP_ALL] = &reduce_op_all,
            [MAG_OP_ANY] = &reduce_op_any,
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
            [MAG_OP_FLOORDIV] = &binary_op_floordiv,
            [MAG_OP_MOD] = &binary_op_mod,
            [MAG_OP_POW] = &binary_op_pow,
            [MAG_OP_MATMUL] = nullptr,
            [MAG_OP_REPEAT_BACK] = nullptr,
            [MAG_OP_GATHER] = nullptr,
            [MAG_OP_AND] = &binary_op_and,
            [MAG_OP_OR] = &binary_op_or,
            [MAG_OP_XOR] = &binary_op_xor,
            [MAG_OP_NOT] = &unary_op_not,
            [MAG_OP_SHL] = &binary_op_shl,
            [MAG_OP_SHR] = &binary_op_shr,
            [MAG_OP_EQ] = &binary_op_eq,
            [MAG_OP_NE] = &binary_op_ne,
            [MAG_OP_LE] = &binary_op_le,
            [MAG_OP_GE] = &binary_op_ge,
            [MAG_OP_LT] = &binary_op_lt,
            [MAG_OP_GT] = &binary_op_gt,
            [MAG_OP_WHERE] = nullptr
        };
        static_assert(std::size(dispatch_table) == MAG_OP__NUM, "Dispatch table size mismatch");
        //static_assert([] -> bool {
        //    for (auto *fn : dispatch_table)
        //        if (!fn) return false;
        //    return true;
        //}());
        auto *kernel = dispatch_table[cmd->op];
        mag_assert(kernel != nullptr, "Operator %s not implemented in CUDA backend", mag_op_traits(cmd->op)->mnemonic);
        (*kernel)(*cmd);
        cudaDeviceSynchronize();
        return MAG_STATUS_OK;
    }

    [[nodiscard]] static mag_status_t alloc_storage_buffer(mag_device_t *dvc, mag_storage_buffer_t **out, size_t size, mag_dtype_t dtype) {
        mag_context_t *ctx = dvc->ctx;
        uintptr_t base;
        if (cudaMalloc(reinterpret_cast<void **>(&base), size) != cudaSuccess)
            return MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED;
        *out = static_cast<mag_storage_buffer_t *>(mag_slab_alloc(&ctx->storage_slab));
        new (*out) mag_storage_buffer_t {
            .__rcb = {},
            .ctx = ctx,
            .flags = MAG_STORAGE_FLAG_ACCESS_W,
            .alignment = 256, // cudaMalloc guarantees this
            .base = base,
            .size = size,
            .device = dvc,
            .aux = {},
        };
        mag_rc_init_object(*out, +[](void *self) -> mag_status_t {
            auto *buffer = static_cast<mag_storage_buffer_t *>(self);
            mag_context_t *ctx = buffer->ctx;
            auto *base = reinterpret_cast<void *>(buffer->base);
            mag_slab_free(&ctx->storage_slab, buffer);
            return cudaFree(base) == cudaSuccess ? MAG_STATUS_OK : MAG_STATUS_ERR_MEMORY_DEALLOCATION_FAILED;
        });
        return MAG_STATUS_OK;
    }

    class physical_device final : public mag_device_t {
    public:
        physical_device(mag_context_t *ctx, uint32_t idx) : mag_device_t{} {
            // Init interface of superclass first
            impl = this;
            this->ctx = ctx;
            id = {.type=MAG_BACKEND_TYPE_CUDA, .device_ordinal=idx};
            is_async = false;
            submit = &::mag::submit;
            alloc_storage = &::mag::alloc_storage_buffer;
            manual_seed = &::mag::manual_seed;
            transfer = &::mag::cuda_transfer;

            CUdevice cu_dvc = 0;
            CUresult res = CUDA_SUCCESS;
            cudaError_t res2 = cudaSuccess;
            int ord = static_cast<int>(idx);
            if ((res = cuDeviceGet(&cu_dvc, ord)) != CUDA_SUCCESS)
                throw cuda_backend_error {res, "Failed to get device"};
            int vmm_support = 0;
            if ((res = cuDeviceGetAttribute(&vmm_support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_dvc)) != CUDA_SUCCESS)
                throw cuda_backend_error {res, "Failed to query VMM support"};
            size_t vmm_gran = 0;
            if (vmm_support) {
                CUmemAllocationProp alloc_props {};
                alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                alloc_props.location.id = ord;
                if ((res = cuMemGetAllocationGranularity(&vmm_gran, &alloc_props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)) != CUDA_SUCCESS)
                    throw cuda_backend_error {res, "Failed to query VMM granularity"};
            }
            cudaDeviceProp props = {};
            if ((res2 = cudaGetDeviceProperties(&props, ord)) != cudaSuccess)
                throw cuda_backend_error {res2, "Failed to get device properties"};
            m_vram = props.totalGlobalMem;
            m_cl = static_cast<uint32_t>(100*props.major + 10*props.minor);
            m_nsm = static_cast<uint32_t>(props.multiProcessorCount);
            m_ntpb = static_cast<uint32_t>(props.maxThreadsPerBlock);
            m_smpb = props.sharedMemPerBlock;
            m_smpb_opt = props.sharedMemPerBlockOptin;
            m_has_vmm = !!vmm_support;
            m_vmm_granularity = vmm_gran;
            std::snprintf(physical_device_name, std::size(physical_device_name), "%s", props.name);
        }

        [[nodiscard]] size_t vram() const noexcept { return m_vram; }
        [[nodiscard]] uint32_t compute_capability() const noexcept { return m_cl; }
        [[nodiscard]] uint32_t num_sms() const noexcept { return m_nsm; }
        [[nodiscard]] uint32_t max_threads_per_block() const noexcept { return m_ntpb; }
        [[nodiscard]] size_t shared_mem_per_block() const noexcept { return m_smpb; }
        [[nodiscard]] size_t shared_mem_per_block_optin() const noexcept { return m_smpb_opt; }
        [[nodiscard]] bool supports_vmm() const noexcept { return m_has_vmm; }
        [[nodiscard]] size_t vmm_granularity() const noexcept { return m_vmm_granularity; }

    private:
        size_t m_vram = 0;
        uint32_t m_cl = 0;
        uint32_t m_nsm = 0;
        uint32_t m_ntpb = 0;
        size_t m_smpb = 0;
        size_t m_smpb_opt = 0;
        bool m_has_vmm = false;
        size_t m_vmm_granularity = 0;
    };

    class cuda_backend final : public mag_backend_t {
    public:
        cuda_backend() : mag_backend_t{} {
            // Only init the interface of superclass here, the actual device initialization is deferred to the init callback
            this->impl = this;
            this->init = +[](mag_backend_t *bck, mag_context_t *ctx) -> bool {
                return static_cast<cuda_backend *>(bck->impl)->initialize(ctx);
            };
            this->shutdown = +[](mag_backend_t *bck) -> bool {
                return static_cast<cuda_backend *>(bck->impl)->destroy();
            };
            backend_version = +[](mag_backend_t *bck) noexcept -> uint32_t { return MAG_CUDA_BACKEND_VERSION; };
            runtime_version = +[](mag_backend_t *bck) noexcept -> uint32_t { return MAG_VERSION; };
            id = +[](mag_backend_t *bck) noexcept -> const char* { return "cuda"; };
            num_devices = +[](mag_backend_t *bck) noexcept -> uint32_t { return static_cast<cuda_backend *>(bck->impl)->device_count(); };
            best_device_id = +[](mag_backend_t *bck) noexcept -> uint32_t { return 0; };
            get_device = +[](mag_backend_t *bck, uint32_t idx) -> mag_device_t* {
                auto &dvc = static_cast<cuda_backend *>(bck->impl)->devices();
                if (idx >= dvc.size()) {
                    mag_log_error("Invalid device index %u (max %zu)", idx, dvc.size()-1);
                    return nullptr;
                }
                return &*dvc[idx];
            };
        }

        [[nodiscard]] uint32_t device_count() const noexcept { return static_cast<uint32_t>(m_devices.size()); }
        [[nodiscard]] uint32_t active_device_idx() const noexcept { return m_active_device_idx; }
        [[nodiscard]] uint32_t best_device_idx() const noexcept { return m_best_device_idx; }
        [[nodiscard]] const physical_device &active_device() const noexcept { return *m_devices.at(m_active_device_idx); }
        [[nodiscard]] const physical_device &best_device() const noexcept { return *m_devices.at(m_best_device_idx); }
        [[nodiscard]] const std::vector<std::unique_ptr<physical_device>> &devices() const noexcept { return m_devices; }

    private:
        [[nodiscard]] bool initialize(mag_context_t *ctx) {
            int ndvc = 0;
            if (cudaGetDeviceCount(&ndvc) != cudaSuccess || ndvc <= 0) { // No GPUs found, backend cannot be used
                mag_log_error("No CUDA-capable devices found.");
                return false;
            }
            m_devices.reserve(ndvc);
            for (int device_ordinal=0; device_ordinal < ndvc; ++device_ordinal) {
                try {
                    m_devices.emplace_back(std::make_unique<physical_device>(ctx, device_ordinal));
                } catch (const cuda_backend_error &e) {
                    mag_log_error("Failed to initialize CUDA device %d: %s", device_ordinal, e.what());
                } catch (...) {
                    mag_log_error("Unknown error while initializing CUDA device %d", device_ordinal);
                }
            }
            return true;
        }

        [[nodiscard]] bool destroy() {
            m_devices.clear();
            return true;
        }

        uint32_t m_active_device_idx = 0;
        uint32_t m_best_device_idx = 0;
        std::vector<std::unique_ptr<physical_device>> m_devices = {};
    };
}

uint32_t MAG_BACKEND_SYM_ABI_COOKIE(){
    return mag_pack_abi_cookie('M', 'A', 'G', MAG_BACKEND_MODULE_ABI_VER);
}

mag_backend_t *MAG_BACKEND_SYM_INIT([[maybe_unused]] mag_context_t *ctx)
try {
    return new mag::cuda_backend {};
} catch (const std::exception &e) {
    mag_log_error("Error during backend initialization: %s", e.what());
    return nullptr;
} catch (...) {
    mag_log_error("Unknown error during backend initialization.");
    return nullptr;
}

void MAG_BACKEND_SYM_SHUTDOWN(mag_backend_t *backend)
try {
    delete static_cast<mag::cuda_backend *>(backend);
} catch (const std::exception &e) {
    mag_log_error("Error during backend shutdown: %s", e.what());
} catch (...) {
    mag_log_error("Unknown error during backend shutdown.");
}
