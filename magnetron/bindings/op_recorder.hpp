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

#include <cstdint>
#include <chrono>
#include <string>
#include <unordered_map>
#include <functional>
#include <mutex>

#include "prelude.hpp"

namespace mag::bindings {
  struct matmul_profile {
    uint64_t calls = 0;
    std::chrono::nanoseconds total {};
    std::chrono::nanoseconds max {};
  };

  class op_recorder final {
  public:
    [[nodiscard]] static op_recorder &singleton();

    void dump_csv(const std::string& base_name = "magnetron_profile");
    void profile(mag_tensor_t *x, mag_tensor_t *y, std::function<void()>&& callback);

  private:
    void record_matmul_data(const std::string &shape, const std::string &dtype, const std::string &kind, std::chrono::nanoseconds ns);

    op_recorder() = default;
    ~op_recorder();

    struct key {
      std::string shape {};
      std::string dtype {};
      std::string kind {};
      bool operator == (const key &other) const;
    };
    struct key_hash {
      size_t operator()(const key &k) const;
    };

    std::mutex m_mtx {};
    std::unordered_map<key, matmul_profile, key_hash> m_profiles {};
  };
}
