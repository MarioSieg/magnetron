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

#pragma once

#include <memory>

#include "mag_cuda.cuh"

#include <string>
#include <type_traits>

#include <cuda_runtime_api.h> /* CUDART_VERSION: nvcc pre-includes this for .cu, spell it out so the toolkit checks below cannot silently evaluate to 0. */

namespace mag {
  // Device capability bits used to select kernel variants at runtime.
  struct device_features final {
    enum $ : uint32_t {
      none = 0,
      virt_mem = 1<<0,  // virtual memory support
      mem_pool = 1<<1,  // memory pool for cudaMallocAsync
      mma = 1<<2,       // mma.sync - base tensor cores, sm_70+
      ldmatrix = 1<<3,  // ldmatrix.sync.aligned, sm_75+
      cp_async = 1<<4,  // cp.async.ca/cg.shared.global, sm_80+
      mma_bf16 = 1<<5,  // mma.sync m16n8k16 bf16/tf32, sm_80+
      stmatrix = 1<<6,  // stmatrix.sync.aligned, sm_90+
      tma = 1<<7,       // cp.async.bulk.tensor and cuTensorMapEncode*, sm_90+
      clusters = 1<<8,  // thread block clusters and distributed shared memory, sm_90+
      wgmma = 1<<9,     // wgmma.mma_async, Hopper only
      tcgen05 = 1<<10,  // tcgen05.mma and tensor memory - datacenter Blackwell only
    };
    [[nodiscard]] static std::underlying_type_t<$> from_compute_caps(uint32_t cl) noexcept;
    [[nodiscard]] static std::string to_string(std::underlying_type_t<$> f);
  };

  class physical_device : public mag_device_t {
  public:
    physical_device(const physical_device &) = delete;
    physical_device(physical_device &&) = default;
    physical_device &operator=(const physical_device &) = delete;
    physical_device &operator=(physical_device &&) = default;
    virtual ~physical_device();

    [[nodiscard]] static mag_status_t create(
      mag_error_t *err,
      std::shared_ptr<physical_device> &out,
      mag_context_t *ctx,
      int ordinal
    );

    [[nodiscard]] size_t vram() const noexcept { return m_vram; }
    [[nodiscard]] uint32_t compute_capability() const noexcept { return m_cl; }
    [[nodiscard]] uint32_t num_sms() const noexcept { return m_nsm; }
    [[nodiscard]] uint32_t max_threads_per_block() const noexcept { return m_ntpb; }
    [[nodiscard]] size_t shared_mem_per_block() const noexcept { return m_smpb; }
    [[nodiscard]] size_t shared_mem_per_block_optin() const noexcept { return m_smpb_opt; }
    [[nodiscard]] size_t vmm_granularity() const noexcept { return m_vmm_granularity; }
    [[nodiscard]] std::string_view name() const noexcept { return physical_device_name; }
    [[nodiscard]] std::underlying_type_t<device_features::$> features() const noexcept { return m_features; }
    [[nodiscard]] bool has_features(std::underlying_type_t<device_features::$> mask) const noexcept { return (m_features & mask) == mask; }
    [[nodiscard]] cudaStream_t stream() const noexcept { return m_stream; }
    [[nodiscard]] cudaEvent_t event() const noexcept { return m_event; }
    [[nodiscard]] std::string info_string() const;

  protected:
    physical_device() = default;

  private:
    size_t m_vram = 0;
    uint32_t m_cl = 0;
    uint32_t m_nsm = 0;
    uint32_t m_ntpb = 0;
    size_t m_smpb = 0;
    size_t m_smpb_opt = 0;
    size_t m_vmm_granularity = 0;
    std::underlying_type_t<device_features::$> m_features = device_features::$::none;
    cudaStream_t m_stream = nullptr;
    cudaEvent_t m_event = nullptr;
  };
}
