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

#include "mag_cuda_device.cuh"

#include <algorithm>

#include "mag_cuda_transfer.cuh"
#include "mag_cuda_storage.cuh"
#include "mag_cuda_exec.cuh"

#include <array>
#include <sstream>

namespace mag {
  std::underlying_type_t<device_features::$> device_features::from_compute_caps(uint32_t cl) noexcept {
    std::underlying_type_t<$> f = none;
    if (cl >= 700) f |= mma;
    if (cl >= 750) f |= ldmatrix;
    if (cl >= 800) f |= cp_async|mma_bf16;
    if (cl >= 900) f |= stmatrix|tma|clusters;
    if (cl >= 900 && cl < 1000) f |= wgmma; // wgmma is Hopper only, Blackwell gropped it for tcgen05 again
    if (cl == 1000 || cl == 1010 || cl == 1030) f |= tcgen05; // only datacenter Blackwell has tcgen05, consumer doesn't
#if CUDART_VERSION < 12080
    f &= ~static_cast<decltype(f)>(tcgen05);
#endif
#if CUDART_VERSION < 12000
    f &= ~static_cast<decltype(f)>(stmatrix|tma|clusters|wgmma);
#endif
    return f;
  }

  std::string device_features::to_string(std::underlying_type_t<$> f) {
    static constexpr std::array<std::string_view, 11> table = {
      "virt_mem",
      "mem_pool",
      "mma",
      "ldmatrix",
      "cp_async",
      "mma_bf16",
      "stmatrix",
      "tma",
      "clusters",
      "wgmma",
      "tcgen05",
    };
    std::stringstream ss {};
    for (size_t i=0; i < table.size(); ++i) {
      if (!(f&(1u<<i))) continue;
      auto name = std::string{table[i]};
      std::for_each(name.begin(), name.end(), [](char &c) -> void { c = static_cast<char>(std::toupper(c)); });
      ss << name;
      if (i != (table.size()-1)) ss << " ";
    }
    auto res = ss.str();
    return res.empty() ? "none" : res;
  }

  static void set_global_seed([[maybe_unused]] mag_error_t *err, [[maybe_unused]] mag_device_t *dvc, uint64_t seed) {
    global_seed.store(seed, std::memory_order_relaxed);
  }

  mag_status_t physical_device::create(
     mag_error_t *err,
     std::shared_ptr<physical_device> &out,
     mag_context_t *ctx,
     int ordinal
   ) {
    if (mag_unlikely(ordinal > static_cast<int>(MAG_DEVICE_ORDINAL_MAX)))
      return mag_set_error(err, MAG_ERR_PARAM, "cuda: device ordinal exceeds maximum %d must be <= %d.", ordinal, MAG_DEVICE_ORDINAL_MAX);

    struct proxy final : physical_device {};
    auto device = std::make_shared<proxy>();

    // Init magnetron interface base
    dynamic_cast<mag_device_t &>(*device) = mag_device_t {
      .impl = &*device,
      .ctx = ctx,
      .id = mag_device_id_t {
        .is_virtual = false,
        .device_ordinal = static_cast<uint32_t>(ordinal),
        .type = MAG_BACKEND_TYPE_CUDA
      },
      .is_async = true,
      .submit = &submit_op,
      .alloc_storage = &alloc_storage_buffer,
      .manual_seed = &set_global_seed,
      .transfer = &bidirectional_transfer,
      .physical_device_name = {}
    };

    CUdevice dvc {};
    mag_cu_check(err, cuDeviceGet(&dvc, ordinal), "failed to get device ordinal");
    size_t vmm_gran = 0;
    if (int vmm_support = 0; cuDeviceGetAttribute(&vmm_support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dvc) == CUDA_SUCCESS && !!vmm_support) {
      CUmemAllocationProp alloc_props {};
      alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      alloc_props.location.id = ordinal;
      CUresult res = cuMemGetAllocationGranularity(&vmm_gran, &alloc_props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
      if (mag_likely(res == CUDA_SUCCESS)) device->m_features |= device_features::virt_mem;
    }
    cudaDeviceProp props = {};
    mag_cu_rt_check(err, cudaGetDeviceProperties(&props, ordinal), "failed to query device properties");
    device->m_vram = props.totalGlobalMem;
    device->m_cl = static_cast<uint32_t>(100*props.major + 10*props.minor);
    device->m_nsm = static_cast<uint32_t>(props.multiProcessorCount);
    device->m_ntpb = static_cast<uint32_t>(props.maxThreadsPerBlock);
    device->m_smpb = props.sharedMemPerBlock;
    device->m_smpb_opt = props.sharedMemPerBlockOptin;
    device->m_vmm_granularity = vmm_gran;
    device->m_stream = nullptr;
    device->m_event = nullptr;
    std::snprintf(device->physical_device_name, std::size(device->physical_device_name), "%s", props.name);
    mag_cu_rt_check(err, cudaSetDevice(ordinal), "failed to select device for stream creation");
    mag_cu_rt_check(err, cudaStreamCreateWithFlags(&device->m_stream, cudaStreamNonBlocking), "failed to create non-blocking stream");
    mag_cu_rt_check(err, cudaEventCreateWithFlags(&device->m_event, cudaEventDisableTiming), "failed to create device event");

    device->m_features |= device_features::from_compute_caps(device->m_cl);
    if (device->m_features & device_features::clusters) {
      int cluster_support = 0;
      if (cudaDeviceGetAttribute(&cluster_support, cudaDevAttrClusterLaunch, ordinal) != cudaSuccess || !cluster_support) // Cluster launch requires additional check
        device->m_features &= ~static_cast<std::underlying_type_t<device_features::$>>(device_features::clusters);
    }
    if (int pool_support = 0; cudaDeviceGetAttribute(&pool_support, cudaDevAttrMemoryPoolsSupported, ordinal) == cudaSuccess && !!pool_support) {
      cudaMemPool_t pool = nullptr;
      if (mag_likely(cudaDeviceGetDefaultMemPool(&pool, ordinal) == cudaSuccess)) {
        uint64_t release_threshold = std::numeric_limits<uint64_t>::max();
        if (mag_likely(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold) == cudaSuccess))
          device->m_features |= device_features::mem_pool;
      }
    }
    out = device;
    return MAG_OK;
  }

  std::string physical_device::info_string() const {
    std::ostringstream ss {};
    ss << (*physical_device_name ? physical_device_name : "Unknown CUDA Device");
    double amount = 0.0;
    const char *unit = "";
    mag_humanize_memory_size(m_vram, &amount, &unit);
    ss << ", VRAM: " << amount << " " << unit;
    ss << ", CAPS: " << device_features::to_string(m_features);
    return ss.str();
  }

  physical_device::~physical_device() {
    if (m_stream || m_event) {
      cudaSetDevice(static_cast<int>(id.device_ordinal));
      if (m_stream) {
        cudaStreamSynchronize(m_stream); /* Storage buffers are freed on this stream so drain before deleting. */
        cudaStreamDestroy(m_stream);
      }
      if (m_event) cudaEventDestroy(m_event);
    }
  }
}
