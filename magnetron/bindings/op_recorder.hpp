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
#include <complex>
#include <string>
#include <unordered_map>
#include <sstream>
#include <functional>
#include <filesystem>
#include <mutex>

#include "prelude.hpp"

#include "core/mag_coords.h"
#include "core/mag_tensor.h"

namespace mag::bindings {
  struct profile_record final {
    uint64_t calls = 0;
    std::chrono::nanoseconds total {};
    std::chrono::nanoseconds max {};
  };

  class op_recorder final {
  public:
    [[nodiscard]] static op_recorder &singleton();

    void dump_csv(const std::filesystem::path &base_name = "magnetron_profile");

    void profile(mag_opcode_t opcode, std::function<void()> &&f, std::initializer_list<mag_tensor_t *> tensors);

  private:
    void record_op_entry(
      std::string &&opcode,
      std::string &&shape,
      std::string &&strides,
      std::string &&dtype,
      std::string &&kind,
      std::chrono::nanoseconds ns
    );

    op_recorder() = default;
    ~op_recorder();

    struct key {
      std::string opcode {};
      std::string shape {};
      std::string shapes {};
      std::string dtype {};
      std::string kind {};
      bool operator == (const key &other) const;
    };
    struct key_hash {
      size_t operator()(const key &k) const;
    };

    std::mutex m_mtx {};
    std::unordered_map<key, profile_record, key_hash> m_profiles {};
  };
}
