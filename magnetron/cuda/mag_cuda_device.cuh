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

namespace mag {
  struct device_features final {
    enum $ : uint32_t {
      none = 0,
      virt_mem = 1<<0,
      mem_pool = 1<<1,
    };
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
    [[nodiscard]] cudaStream_t stream() const noexcept { return m_stream; }
    [[nodiscard]] cudaEvent_t event() const noexcept { return m_event; }

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
