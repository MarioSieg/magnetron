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

#include <mag_context.h>
#include <mag_tensor.h>

#include <cuda.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

constexpr int UNARY_BLOCK_SIZE = 256;
constexpr mag_e8m23_t INVSQRT2 = 0.707106781186547524400844362104849039284835937f /* 1/√2 */;

[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_abs(mag_e8m23_t x) { return fabsf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_sgn(mag_e8m23_t x) { return x > 0.f ? 1.f : x < 0.f ? -1.f : 0.f; }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_neg(mag_e8m23_t x) { return -x; }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_log(mag_e8m23_t x) { return logf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_sqr(mag_e8m23_t x) { return x*x; }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_sqrt(mag_e8m23_t x) { return sqrtf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_sin(mag_e8m23_t x) { return sinf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_cos(mag_e8m23_t x) { return cosf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_step(mag_e8m23_t x) { return x > 0.f; }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_exp(mag_e8m23_t x) { return expf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_floor(mag_e8m23_t x) { return floorf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_ceil(mag_e8m23_t x) { return ceilf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_round(mag_e8m23_t x) { return rintf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_softmax(mag_e8m23_t x) { return exp(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_softmax_dv(mag_e8m23_t x) { return exp(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_sigmoid(mag_e8m23_t x) { return 1.f/(1.f + expf(-x)); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_sigmoid_dv(mag_e8m23_t x) { mag_e8m23_t sig = 1.f/(1.f + expf(-x)); return sig*(1.f-sig); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_hard_sigmoid(mag_e8m23_t x) { return fminf(1.f, fmaxf(0.0f, (x + 3.0f)/6.0f)); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_silu(mag_e8m23_t x) { return x*(1.f/(1.f + expf(-x))); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_silu_dv(mag_e8m23_t x) { mag_e8m23_t sig = 1.f/(1.f + expf(-x)); return sig + x*sig; }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_tanh(mag_e8m23_t x) { return tanhf(x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_tanh_dv(mag_e8m23_t x) {  mag_e8m23_t th = tanhf(x); return 1.f - th*th; }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_relu(mag_e8m23_t x) { return fmax(0.f, x); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_relu_dv(mag_e8m23_t x) { return x > 0.f ? 1.f : 0.f; }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_gelu(mag_e8m23_t x) { return .5f*x*(1.f+erff(x*INVSQRT2)); }
[[nodiscard]] static __device__ __forceinline__ mag_e8m23_t fn_op_gelu_dv(mag_e8m23_t x) { mag_e8m23_t th = tanhf(x); return .5f*(1.f + th) + .5f*x*(1.f - th*th); }

template <mag_e8m23_t (&op)(mag_e8m23_t), typename T>
static __global__ void unary_op_kernel(int n, T *o, const T *x) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= n) return;
    o[i] = static_cast<T>(op(static_cast<mag_e8m23_t>(x[i])));
}

template <mag_e8m23_t (&op)(mag_e8m23_t)>
static void impl_unary_op(mag_tensor_t *o, mag_tensor_t *x) {
    mag_assert2(o->numel == x->numel);
    mag_assert2(mag_tensor_is_contiguous(o));
    mag_assert2(mag_tensor_is_contiguous(x));
    mag_assert2(mag_tensor_is_floating_point_typed(o));
    mag_assert2(mag_tensor_is_floating_point_typed(x));
    mag_assert2(o->dtype == x->dtype);
    int n = static_cast<int>(o->numel);
    int blocks = (n+UNARY_BLOCK_SIZE-1)/UNARY_BLOCK_SIZE;
    switch (o->dtype) {
        case MAG_DTYPE_E8M23: {
            auto *xo = static_cast<mag_e8m23_t*>(mag_tensor_get_data_ptr(o));
            const auto *xx = static_cast<const mag_e8m23_t*>(mag_tensor_get_data_ptr(x));
            unary_op_kernel<op><<<blocks, UNARY_BLOCK_SIZE, 0>>>(n, xo, xx);
        } break;
        case MAG_DTYPE_E5M10: {
            auto *xo = static_cast<__half*>(mag_tensor_get_data_ptr(o));
            const auto *xx = static_cast<const half*>(mag_tensor_get_data_ptr(x));
            unary_op_kernel<op><<<blocks, UNARY_BLOCK_SIZE, 0>>>(n, xo, xx);
        } break;
        default: mag_assert(false, "Unsupported dtype for unary op");
    }
}

#define mag_cuda_check(expr) \
    do { \
        if (auto rrr {(expr)}; rrr != cudaSuccess) { \
            mag_panic(#expr, __func__, __FILE__, __LINE__, cudaGetErrorString(rrr)); \
        } \
    } while (0)

struct cuda_exception : std::runtime_error {
    explicit cuda_exception(const char *msg) : std::runtime_error(msg) {}
};

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

}

static void submit(mag_device_t *dvc, const mag_command_t *cmd) {
    switch (cmd->op) {
        case MAG_OP_ABS: impl_unary_op<fn_op_abs>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SGN: impl_unary_op<fn_op_sgn>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_NEG: impl_unary_op<fn_op_neg>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_LOG: impl_unary_op<fn_op_log>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SQR: impl_unary_op<fn_op_sqr>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SQRT: impl_unary_op<fn_op_sqrt>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SIN: impl_unary_op<fn_op_sin>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_COS: impl_unary_op<fn_op_cos>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_STEP: impl_unary_op<fn_op_step>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_EXP: impl_unary_op<fn_op_exp>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_FLOOR: impl_unary_op<fn_op_floor>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_CEIL: impl_unary_op<fn_op_ceil>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_ROUND: impl_unary_op<fn_op_round>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SOFTMAX: impl_unary_op<fn_op_softmax>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SOFTMAX_DV: impl_unary_op<fn_op_softmax_dv>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SIGMOID: impl_unary_op<fn_op_sigmoid>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SIGMOID_DV: impl_unary_op<fn_op_sigmoid_dv>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_HARD_SIGMOID: impl_unary_op<fn_op_hard_sigmoid>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SILU: impl_unary_op<fn_op_silu>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_SILU_DV: impl_unary_op<fn_op_silu_dv>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_TANH: impl_unary_op<fn_op_tanh>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_TANH_DV: impl_unary_op<fn_op_tanh_dv>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_RELU: impl_unary_op<fn_op_relu>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_RELU_DV: impl_unary_op<fn_op_relu_dv>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_GELU: impl_unary_op<fn_op_gelu>(cmd->out[0], cmd->in[0]); break;
        case MAG_OP_GELU_DV: impl_unary_op<fn_op_gelu_dv>(cmd->out[0], cmd->in[0]); break;
        default: mag_assert(false, "Unsupported operation in CUDA backend: %s", mag_op_meta_of(cmd->op)->mnemonic); break;
    }
}

static void dealloc_storage_buffer(void *self) {
    auto *buffer = static_cast<mag_storage_buffer_t *>(self);
    mag_context_t *ctx = buffer->ctx;
    mag_cuda_check(cudaFree(reinterpret_cast<void *>(buffer->base)));
    mag_fixed_pool_free_block(&ctx->storage_pool, buffer);
}

static void broadcast(mag_storage_buffer_t *sto, size_t offs, const void *x, size_t stride) {

}

static void transfer(mag_storage_buffer_t *sto, mag_transfer_dir_t dir, size_t offs, void *inout, size_t size) {

}

static void convert(mag_storage_buffer_t *sto, mag_transfer_dir_t dir, size_t offs, void *host, size_t size, mag_dtype_t hdt) {

}

static void alloc_storage_buffer(mag_device_t *device, mag_storage_buffer_t **out, size_t size, mag_dtype_t dtype) {
    mag_context_t *ctx = device->ctx;
    void *block = nullptr;
    mag_cuda_check(cudaMalloc(&block, size));
    *out = static_cast<mag_storage_buffer_t*>(mag_fixed_pool_alloc_block(&ctx->storage_pool));
    new (*out) mag_storage_buffer_t {
        .ctx = ctx,
        .rc_control = mag_rc_control_init(*out, &dealloc_storage_buffer),
        .base = reinterpret_cast<uintptr_t>(block),
        .size = size,
        .alignment = 256, // cudaMalloc guarantees this
        .granularity = mag_dtype_meta_of(dtype)->size,
        .dtype = dtype,
        .host = device,
        .broadcast = &broadcast,
        .transfer = &transfer,
        .convert = &convert
    };
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
} catch (const cuda_exception &e) {
    mag_log_error("CUDA error during backend initialization: %s", e.what());
    return nullptr;
} catch (const std::exception &e) {
    mag_log_error("Error during backend initialization: %s", e.what());
    return nullptr;
} catch (...) {
    mag_log_error("Unknown error during backend initialization.");
    return nullptr;
}

void MAG_BACKEND_SYM_SHUTDOWN(mag_backend_t *backend)
try {
    delete static_cast<backend_impl *>(backend->impl);
    delete backend;
} catch (const std::exception &e) {
    mag_log_error("Error during backend shutdown: %s", e.what());
} catch (...) {
    mag_log_error("Unknown error during backend shutdown.");
}
