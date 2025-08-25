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

#include <array>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <optional>

#include <cuda.h>

#include  "magnetron_internal.h"

/* Runtime result check. */
#define mag_cuda_check(expr) \
    do { \
        if (auto rrr {(expr)}; rrr != cudaSuccess) { \
            mag_panic(#expr, __func__, __FILE__, __LINE__, cudaGetErrorString(rrr)); \
        } \
    } while (0)

[[nodiscard]] static std::pair<mag_e11m52_t, const char*> humanize_vram_size_size(size_t nb) {
    if (nb < 1<<10) return {static_cast<mag_e11m52_t>(nb), "B"};
    if (nb < 1<<20) return {static_cast<mag_e11m52_t>(nb)/static_cast<mag_e11m52_t>(1<<10), "KiB"};
    if (nb < 1<<30) return {static_cast<mag_e11m52_t>(nb)/static_cast<mag_e11m52_t>(1<<20), "MiB"};
    return {static_cast<mag_e11m52_t>(nb)/static_cast<mag_e11m52_t>(1<<30), "GiB"};
}

namespace kernels {
    template <typename T>
    static __global__ void broadcast(T* dst, T value, size_t n) {
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < n) dst[i] = value;
    }

    static __global__ void broadcast_generic(uint8_t* dst, const uint8_t* pattern, size_t stride, size_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const uint8_t* src = pattern;
        uint8_t* out = dst + i*stride;
        #pragma unroll
        for (size_t j=0; j < stride; ++j)
            out[j] = src[j];
    }
}

namespace storage {
    static void dealloc(void* self) {
        auto* buffer = static_cast<mag_istorage_t*>(self);
        mag_context_t* ctx = buffer->ctx;
        mag_assert(ctx->num_storages > 0, "double freed storage");
        --ctx->num_storages;
        mag_cuda_check(cudaFree(reinterpret_cast<void*>(buffer->base)));
        mag_fixed_pool_free(&ctx->storage_pool, buffer);
    }

    static void alloc(mag_idevice_t* host, mag_istorage_t** out, size_t size, mag_dtype_t dtype) {
        mag_context_t* ctx = host->ctx;
        void* block = nullptr;
        mag_cuda_check(cudaMalloc(&block, size));
        auto* buffer = static_cast<mag_istorage_t*>(mag_fixed_pool_malloc(&ctx->storage_pool));
        new (buffer) mag_istorage_t {
            .ctx = ctx,
            .rc_control = mag_rc_control_init(buffer, &dealloc),
            .base = reinterpret_cast<uintptr_t>(buffer),
            .size = size,
            .alignment = 256, // cudaMalloc guarantees this
            .granularity = mag_dtype_meta_of(dtype)->size,
            .dtype = dtype,
            .host = host,
            .broadcast = nullptr, // TODO
            .transfer = nullptr // TODO
        };
    }
}

constexpr size_t MAX_DEVICES = 32;
constexpr int DEFAULT_DEVICE_ID = 0;

struct PhysicalDevice final {
    int id = 0;                         /* Device ID */
    std::array<char, 128> name = {};    /* Device name */
    size_t vram = 0;                    /* Video memory in bytes */
    uint32_t cl = 0;                    /* Compute capability */
    uint32_t nsm = 0;                   /* Number of SMs */
    uint32_t ntpb = 0;                  /* Number of threads per block */
    size_t smpb = 0;                    /* Shared memory per block */
    size_t smpb_opt = 0;                /* Shared memory per block opt-in */
    bool has_vmm = false;               /* Has virtual memory management */
    size_t vmm_granularity = 0;         /* Virtual memory management granularity */
};

extern "C" mag_idevice_t* mag_init_device_cuda(mag_context_t* ctx, const mag_device_desc_t* desc) {
    int requested_device_id = static_cast<int>(desc->cuda_device_id);
    int ngpus = 0;
    if (cudaGetDeviceCount(&ngpus) != cudaSuccess)
        return nullptr;
    if (ngpus < 1 || ngpus > MAX_DEVICES) // No devices found, return nullptr so magnetron falls back to the CPU backend
        return nullptr;

    auto fetch_device_info = [](int ordinal) -> std::optional<PhysicalDevice> {
        CUdevice cu_dvc = 0;
        if (cuDeviceGet(&cu_dvc, ordinal) != CUDA_SUCCESS)
            return std::nullopt;
        int vmm_support = 0;
        if (cuDeviceGetAttribute(&vmm_support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_dvc) != CUDA_SUCCESS)
            return std::nullopt;
        size_t vmm_granularity = 0;
        if (vmm_support) {
            CUmemAllocationProp alloc_props {};
            alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_props.location.id = ordinal;
            if (cuMemGetAllocationGranularity(&vmm_granularity, &alloc_props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS)
                return std::nullopt;
        }
        cudaDeviceProp props = {};
        if (cudaGetDeviceProperties(&props, ordinal) != cudaSuccess)
            return std::nullopt;
        PhysicalDevice device = {
            .id = ordinal,
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
    };

    std::vector<PhysicalDevice> device_list = {};
    device_list.reserve(ngpus);
    for (int id=0; id < ngpus; ++id) {
        std::optional<PhysicalDevice> device = fetch_device_info(id);
        if (device) device_list.emplace_back(*device);
    }
    if (device_list.empty()) { /* No devices available or initialization failed, let runtime fallback to other compute device. */
        mag_log_error("No CUDA devices with id %d available, using CPU processing", requested_device_id);
        return nullptr;
    }

    requested_device_id = requested_device_id < ngpus ? requested_device_id : DEFAULT_DEVICE_ID;
    auto* active_device = new PhysicalDevice{device_list[requested_device_id]};

    auto* compute_device = new mag_idevice_t {
        .name = "GPU",
        .impl = active_device,
        .is_async = true,
        .type = MAG_DEVICE_TYPE_GPU_CUDA,
        .eager_exec_init = nullptr,
        .eager_exec_fwd = nullptr,
        .alloc_storage = &storage::alloc,
    };
    auto [vram, unit] = humanize_vram_size_size(active_device->vram);
    std::snprintf(compute_device->name, sizeof(compute_device->name), "%s - %s - %.03f %s VRAM", mag_device_type_get_name(compute_device->type), active_device->name.data(), vram, unit);
    return compute_device;
}

extern "C" void mag_destroy_device_cuda(mag_idevice_t* dvc) {
    delete static_cast<PhysicalDevice*>(dvc->impl);
    delete dvc;
}
