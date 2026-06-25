py/*
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
    return opcode == other.opcode && shape == other.shape && dtype == other.dtype && kind == other.kind;
  }

  size_t op_recorder::key_hash::operator()(const key &k) const {
    size_t h = std::hash<std::string>{}(k.opcode);
    h ^= std::hash<std::string>{}(k.shape) << 1;
    h ^= std::hash<std::string>{}(k.dtype) << 2;
    h ^= std::hash<std::string>{}(k.kind) << 3;
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

  void op_recorder::do_profile(
    op_recorder &rec,
    mag_opcode_t opcode,
    std::function<void()> &&f,
    mag_tensor_t * const *data,
    size_t count
  ) {
    std::stringstream shape_ss {};
    std::stringstream strides_ss {};
    for (size_t i=0; i < count; ++i) {
      auto *tensor = data[i];
      auto fmt_shape_tuple = [&](std::stringstream &ss, const int64_t (&dims)[MAG_MAX_DIMS]) {
        char shape_buf[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&shape_buf, &dims, tensor->coords.rank);
        ss << shape_buf << (i < count-1 ? ", " : "");
      };
      fmt_shape_tuple(shape_ss, tensor->coords.shape);
      fmt_shape_tuple(strides_ss, tensor->coords.strides);
    }
    std::string dtype = count ? std::string{mag_type_trait(data[0]->dtype)->short_name} : "?";
    std::string kind {};
    if (opcode == MAG_OP_MATMUL && count == 2) {
      auto mmt = mag_matmul_type_detect(data[0], data[1]);
      bool contig = mag_matmul_type_is_micro_kernel_contig(mmt, data[0], data[1]);
      kind = mag_matmul_type_name(mmt);
      kind += contig ? " C" : " S";
    } else {
      bool contig = mag_all_shapes_equal_and_contig(const_cast<const mag_tensor_t **>(data), count);
      kind = contig ? "C" : "S";
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::invoke(f);
    auto end = std::chrono::high_resolution_clock::now();
    std::string opname = mag_op_trait(opcode)->mnemonic;
    rec.record_op_entry(std::move(opname), shape_ss.str(), strides_ss.str(), std::move(dtype), std::move(kind), end - start);
  }

  void op_recorder::profile(mag_opcode_t opcode, std::function<void()> &&f, std::initializer_list<mag_tensor_t *> tensors) {
    do_profile(*this, opcode, std::move(f), const_cast<mag_tensor_t * const *>(tensors.begin()), tensors.size());
  }

  void op_recorder::profile(mag_opcode_t opcode, std::function<void()> &&f, std::vector<mag_tensor_t *> const& tensors) {
    do_profile(*this, opcode, std::move(f), tensors.data(), tensors.size());
  }

  void op_recorder::dump_csv(const std::filesystem::path &base_name) {
    std::lock_guard<std::mutex> lock {m_mtx};
    if (m_profiles.empty())
      return;

    struct row {
      key k;
      profile_record p;
    };
    std::vector<row> rows {};
    rows.reserve(m_profiles.size());
    for (auto &&[k, v] : m_profiles)
      rows.emplace_back(row{k, v});
    std::sort(rows.begin(), rows.end(), [](const row &a, const row &b) noexcept -> bool {
      return a.p.total > b.p.total;
    });

    double grand_total_ms = 0.0;
    for (auto &&r : rows)
      grand_total_ms += std::chrono::duration<double, std::milli>(r.p.total).count();

    constexpr int top_n = 20;
    int shown = static_cast<int>(std::min(rows.size(), static_cast<size_t>(top_n)));

    std::cout << "\n[Magnetron Op Profile] " << rows.size() << " unique op signatures, "
              << std::fixed << std::setprecision(2) << grand_total_ms << " ms total\n";
    std::cout << std::left
              << std::setw(4)  << "#"
              << std::setw(16) << "op"
              << std::setw(7)  << "dtype"
              << std::setw(12) << "kind"
              << std::right
              << std::setw(8)  << "calls"
              << std::setw(12) << "total(ms)"
              << std::setw(11) << "avg(us)"
              << std::setw(11) << "max(us)"
              << std::setw(8)  << "%time"
              << '\n';
    std::cout << std::string(89, '-') << '\n';
    for (int i = 0; i < shown; ++i) {
      auto &&r = rows[static_cast<size_t>(i)];
      double total_ms = std::chrono::duration<double, std::milli>(r.p.total).count();
      double max_us = std::chrono::duration<double, std::micro>(r.p.max).count();
      std::chrono::nanoseconds avg {};
      if (r.p.calls) avg = r.p.total / r.p.calls;
      double avg_us = std::chrono::duration<double, std::micro>(avg).count();
      double pct = grand_total_ms > 0.0 ? 100.0*total_ms / grand_total_ms : 0.0;
      std::cout << std::left
                << std::setw(4)  << (std::to_string(i+1) + ".")
                << std::setw(16) << r.k.opcode
                << std::setw(7)  << r.k.dtype
                << std::setw(12) << r.k.kind
                << std::right << std::fixed << std::setprecision(2)
                << std::setw(8)  << r.p.calls
                << std::setw(12) << total_ms
                << std::setw(11) << avg_us
                << std::setw(11) << max_us
                << std::setw(7)  << pct << "%"
                << '\n';
    }
    if (static_cast<int>(rows.size()) > shown)
      std::cout << "  ... and " << rows.size() - shown << " more\n";
    std::cout << std::string(89, '-') << "\n\n";

    std::filesystem::path full_path = make_profile_filename(base_name);
    std::cout << "Writing OP Recorder CSV: " << full_path << '\n';
    std::ofstream file {full_path};
    file << "calls,op,shapes,strides,kind,dtype,total_ms,avg_us,max_us\n";
    for (auto &&r : rows) {
      double total_ms = std::chrono::duration<double, std::milli>(r.p.total).count();
      double max_us = std::chrono::duration<double, std::micro>(r.p.max).count();
      std::chrono::nanoseconds avg {};
      if (r.p.calls) avg = r.p.total / r.p.calls;
      double avg_us = std::chrono::duration<double, std::micro>(avg).count();
      file
        << r.p.calls << ','
        << r.k.opcode << ','
        << '"' << r.k.shape << '"' << ','
        << '"' << r.k.shapes << '"' << ','
        << r.k.kind << ','
        << r.k.dtype << ','
        << total_ms << ','
        << avg_us << ','
        << max_us
        << '\n';
    }
  }
}
