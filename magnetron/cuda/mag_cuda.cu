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

#include <array>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

namespace kernels {
    template <typename T, typename F>
    static __global__ void unary_op(int64_t numel, T* o, const T* x, F &&f) {
        uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < numel)
            o[idx] = f(x[idx]);
    }
}

namespace ops {
    template <typename T, typename F>
    static void unary_op_base(mag_tensor_t* r, mag_tensor_t* x, F &&f) {
        auto *br = static_cast<T *>(mag_tensor_get_data_ptr(r));
        auto *bx = static_cast<T *>(mag_tensor_get_data_ptr(x));
        int64_t numel = r->numel;
        int64_t threads = 256;
        int64_t blocks = (numel + threads - 1)/threads;
        kernels::unary_op<T, true><<<blocks, threads>>>(numel, br, bx, f);
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
