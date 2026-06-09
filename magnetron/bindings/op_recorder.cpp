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

#include "core/mag_coords.h"
#include "core/mag_tensor.h"

namespace mag::bindings {
  bool op_recorder::key::operator==(const key &other) const {
    return shape == other.shape && dtype == other.dtype && kind  == other.kind;
  }

  size_t op_recorder::key_hash::operator()(const key &k) const {
    size_t h = std::hash<std::string>{}(k.shape);
    h ^= std::hash<std::string>{}(k.dtype) << 1;
    h ^= std::hash<std::string>{}(k.kind)  << 2;
    return h;
  }

  op_recorder &op_recorder::singleton() {
    static op_recorder recorder;
    return recorder;
  }

  void op_recorder::profile(mag_tensor_t *x, mag_tensor_t *y, std::function<void()> &&callback) {
    char shape_x[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&shape_x, &x->coords.shape, x->coords.rank);
    char shape_y[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&shape_y, &y->coords.shape, y->coords.rank);
    std::string shape = std::string{shape_x} + " @ " + std::string{shape_y};
    std::string dtype = std::string{mag_type_trait(x->dtype)->short_name};
    std::string kind = mag_matmul_type_name(mag_matmul_type_detect(x, y));
    auto start = std::chrono::high_resolution_clock::now();
    std::invoke(callback);
    auto end = std::chrono::high_resolution_clock::now();
    record_matmul_data(shape, dtype, kind, end - start);
  }

  void op_recorder::record_matmul_data(
    const std::string &shape,
    const std::string &dtype,
    const std::string &kind,
    std::chrono::nanoseconds ns
  ) {

    std::lock_guard<std::mutex> lock {m_mtx};
    auto &entry = m_profiles[{shape, dtype, kind}];
    ++entry.calls;
    entry.total += ns;
    entry.max = std::max(entry.max, ns);
  }

  op_recorder::~op_recorder() {
    dump_csv();
  }

  [[nodiscard]] static std::string make_profile_filename(const std::string &base_name) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    #ifdef _WIN32
        localtime_s(&tm, &tt);
    #else
        localtime_r(&tt, &tm);
    #endif
    std::ostringstream ss;
    ss << base_name << "_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".csv";
    return ss.str();
  }

  void op_recorder::dump_csv(const std::string &base_name) {
    std::lock_guard<std::mutex> lock {m_mtx};
    if (m_profiles.empty())
      return;
    std::string full_path = make_profile_filename(base_name);
    std::cout << "Writing OP Recorder CSV: " << full_path << std::endl;
    struct row {
      key k;
      matmul_profile p;
    };
    std::vector<row> rows {};
    rows.reserve(m_profiles.size());
    for (auto &&[k, v] : m_profiles)
      rows.emplace_back(row{k, v});
    std::sort(rows.begin(), rows.end(), [](const row &a, const row &b) -> bool { return a.p.total > b.p.total; });
    std::ofstream file {full_path};
    file << "calls,shape,kind,dtype,total_ms,avg_us,max_us\n";
    for (const auto &row : rows) {
      double total_ms = std::chrono::duration<double, std::milli>(row.p.total).count();
      double max_us = std::chrono::duration<double, std::micro>(row.p.max).count();
      std::chrono::nanoseconds avg{};
      if (row.p.calls) avg = row.p.total / row.p.calls;
      double avg_us = std::chrono::duration<double, std::micro>(avg).count();
      file
        << row.p.calls << ','
        << '"' << row.k.shape << '"' << ','
        << row.k.kind << ','
        << row.k.dtype << ','
        << total_ms << ','
        << avg_us << ','
        << max_us
        << '\n';
    }
  }
}
