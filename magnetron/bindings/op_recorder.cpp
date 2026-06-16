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

#include "op_recorder.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace mag::bindings {
  bool op_recorder::key::operator==(const key &other) const {
    return shape == other.shape && dtype == other.dtype && kind  == other.kind;
  }

  size_t op_recorder::key_hash::operator()(const key &k) const {
    size_t h = std::hash<std::string>{}(k.shape);
    h ^= std::hash<std::string>{}(k.dtype)<<1;
    h ^= std::hash<std::string>{}(k.kind)<<2;
    return h;
  }

  op_recorder &op_recorder::singleton() {
    static op_recorder recorder;
    return recorder;
  }

  void op_recorder::record_op_entry(
    std::string &&opcode,
    std::string &&shape,
    std::string &&strides,
    std::string &&dtype,
    std::string &&kind,
    std::chrono::nanoseconds ns
  ) {
    std::lock_guard<std::mutex> lock {m_mtx};
    profile_record &entry = m_profiles[{std::move(opcode), std::move(shape), std::move(strides), std::move(dtype), std::move(kind)}];
    ++entry.calls;
    entry.total += ns;
    entry.max = std::max(entry.max, ns);
  }

  op_recorder::~op_recorder() {
    dump_csv();
  }

  [[nodiscard]] static std::filesystem::path make_profile_filename(const std::filesystem::path &base_name) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    #ifdef _WIN32
        localtime_s(&tm, &tt);
    #else
        localtime_r(&tt, &tm);
    #endif
    std::ostringstream ss;
    ss << base_name.string() << "_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".csv";
    return ss.str();
  }

  void op_recorder::profile(mag_opcode_t opcode, std::function<void()> &&f, std::initializer_list<mag_tensor_t *> tensors) {
    std::stringstream shape_ss {};
    std::stringstream strides_ss {};
    for (size_t i=0; i < tensors.size(); ++i) {
      auto *tensor = tensors.begin()[i];
      auto fmt_shape_tuple = [&](std::stringstream &ss, const int64_t (&dims)[MAG_MAX_DIMS]) {
        char shape_buf[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&shape_buf, &dims, tensor->coords.rank);
        ss << shape_buf << (i < tensors.size()-1 ? ", " : "");
      };
      fmt_shape_tuple(shape_ss, tensor->coords.shape);
      fmt_shape_tuple(strides_ss, tensor->coords.strides);
    }
    std::string dtype = tensors.size() ? std::string{mag_type_trait((*tensors.begin())->dtype)->short_name} : "?";
    std::string kind {};
    if (opcode == MAG_OP_MATMUL && tensors.size() == 2) { // Matmul type for matmul
      auto mmt = mag_matmul_type_detect(tensors.begin()[0], tensors.begin()[1]);
      bool contig = mag_matmul_type_is_micro_kernel_contig(mmt, tensors.begin()[0], tensors.begin()[1]);
      kind = mag_matmul_type_name(mmt);
      kind += " ";
      kind += contig ? "C" : "S";
    } else { // Else contig or strided kernel invocation
      bool contig = mag_all_shapes_equal_and_contig(const_cast<const mag_tensor_t **>(&*tensors.begin()), tensors.size());
      kind = contig ? "C" : "S";
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::invoke(f);
    auto end = std::chrono::high_resolution_clock::now();
    std::string opname = mag_op_trait(opcode)->mnemonic;
    record_op_entry(std::move(opname), shape_ss.str(), strides_ss.str(), std::move(dtype), std::move(kind), end - start);
  }

  void op_recorder::dump_csv(const std::filesystem::path &base_name) {
    std::lock_guard<std::mutex> lock {m_mtx};
    if (m_profiles.empty())
      return;
    std::filesystem::path full_path = make_profile_filename(base_name);
    std::cout << "Writing OP Recorder CSV: " << full_path << std::endl;
    struct row {
      key k;
      profile_record p;
    };
    std::vector<row> rows {};
    rows.reserve(m_profiles.size());
    for (auto &&[k, v] : m_profiles)
      rows.emplace_back(row{k, v});
    std::sort(rows.begin(), rows.end(), [](const row &a, const row &b) noexcept -> bool { return a.p.total > b.p.total; });
    std::ofstream file {full_path};
    file << "calls,op,shapes,strides,kind,dtype,total_ms,avg_us,max_us\n";
    for (auto &&row : rows) {
      double total_ms = std::chrono::duration<double, std::milli>(row.p.total).count();
      double max_us = std::chrono::duration<double, std::micro>(row.p.max).count();
      std::chrono::nanoseconds avg{};
      if (row.p.calls) avg = row.p.total / row.p.calls;
      double avg_us = std::chrono::duration<double, std::micro>(avg).count();
      file
        << row.p.calls << ','
        << row.k.opcode << ','
        << '"' << row.k.shape << '"' << ','
        << '"' << row.k.shapes << '"' << ','
        << row.k.kind << ','
        << row.k.dtype << ','
        << total_ms << ','
        << avg_us << ','
        << max_us
        << '\n';
    }
  }
}
