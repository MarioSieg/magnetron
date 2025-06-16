/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
*/

#include <array>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <optional>

#include <cuda.h>

#include  "magnetron_internal.h"

/* Driver result check. */
#define mag_cu_chk_rdv(expr) \
    do { \
    if (auto rrr {(expr)}; rrr != CUDA_SUCCESS) { \
        const char* err_str = "?"; \
        cuGetErrorString(rrr, &err_str); \
        mag_panic(#expr, __func__, __FILE__, __LINE__, err_str); \
    } \
} while (0)

/* Runtime result check. */
#define mag_cu_chk_rt(expr) \
    do { \
        if (auto rrr {(expr)}; rrr != cudaSuccess) { \
            mag_panic(#expr, __func__, __FILE__, __LINE__, cudaGetErrorString(rrr)); \
        } \
    } while (0)

[[nodiscard]] static std::pair<mag_E11M52, const char*> humanize_vram_size_size(size_t nb) {
    if (nb < 1<<10) return {static_cast<mag_E11M52>(nb), "B"};
    if (nb < 1<<20) return {static_cast<mag_E11M52>(nb)/static_cast<mag_E11M52>(1<<10), "KiB"};
    if (nb < 1<<30) return {static_cast<mag_E11M52>(nb)/static_cast<mag_E11M52>(1<<20), "MiB"};
    return {static_cast<mag_E11M52>(nb)/static_cast<mag_E11M52>(1<<30), "GiB"};
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

extern "C" mag_IComputeDevice* mag_init_device_cuda(mag_Context* ctx, const mag_ComputeDeviceDesc* desc) {
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

    auto* compute_device = new mag_IComputeDevice {
        .name = "GPU",
        .impl = active_device,
        .is_async = true,
        .type = MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA,
        .eager_exec_init = nullptr,
        .eager_exec_fwd = nullptr,
        .alloc_storage = nullptr,
    };
    auto [vram, unit] = humanize_vram_size_size(active_device->vram);
    std::snprintf(compute_device->name, sizeof(compute_device->name), "%s - %s - %.03f %s VRAM", mag_device_type_get_name(compute_device->type), active_device->name.data(), vram, unit);
    return compute_device;
}

extern "C" void mag_destroy_device_cuda(mag_IComputeDevice* dvc) {
    delete static_cast<PhysicalDevice*>(dvc->impl);
    delete dvc;
}
