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

    void dump_csv(const std::string& base_name = "magnetron_profile");

    template <typename F, typename... Args>
    void profile(mag_opcode_t opcode, F &&f, Args &&...tensor_args) {
      static_assert(std::is_same_v<std::common_type_t<std::decay_t<Args>...>, mag_tensor_t *>);
      static_assert(std::is_invocable_v<F>);
      std::stringstream shape_ss {};
      std::array<const mag_tensor_t *, sizeof...(Args)> tensors {tensor_args...};
      for (size_t i=0; i < tensors.size(); ++i) {
        auto *tensor = tensors[i];
        char shape_buf[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&shape_buf, &tensor->coords.shape, tensor->coords.rank);
        shape_ss << shape_buf << (i < tensors.size() - 1 ? ", " : "");
      }
      std::string dtype = sizeof...(Args) ? std::string{mag_type_trait(tensors[0]->dtype)->short_name} : "?";
      std::string kind {};
      if (opcode == MAG_OP_MATMUL && 2 == sizeof...(Args)) { // Matmul type for matmul
        kind = mag_matmul_type_name(mag_matmul_type_detect(tensors[0], tensors[1]));
      } else { // Else contig or strided kernel invocation
        bool all_cont = mag_all_shapes_equal_and_contig(tensors.data(), tensors.size());
        kind = all_cont ? "contig" : "strided";
      }
      auto start = std::chrono::high_resolution_clock::now();
      std::invoke(f);
      auto end = std::chrono::high_resolution_clock::now();
      std::string opname = mag_op_trait(opcode)->mnemonic;
      record_op_entry(std::move(opname), shape_ss.str(), std::move(dtype), std::move(kind), end - start);
    }

  private:
    void record_op_entry(
      std::string &&opcode,
      std::string &&shape,
      std::string &&dtype,
      std::string &&kind,
      std::chrono::nanoseconds ns
    );

    op_recorder() = default;
    ~op_recorder();

    struct key {
      std::string opcode {};
      std::string shape {};
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
