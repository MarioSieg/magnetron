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

#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <magnetron/magnetron.h>
#include <core/mag_bfloat16.h>
#include <core/mag_float16.h>
#include <core/mag_float8_e4m3fn.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace mag::bindings {
  namespace nb = nanobind;
  using namespace nb::literals;

  // Lazy init the context, destruction is handled by the module destructor.
  [[nodiscard]] extern mag_context_t *get_ctx();
  [[nodiscard]] extern std::mutex &get_global_mutex();
  [[nodiscard]] extern const std::string &get_default_device_unlocked();
  [[nodiscard]] extern std::string get_default_device();

  // Set to 1 to enable record and profile all executed operators and export them to a CSV
  constexpr bool enable_op_recorder = false;

  struct dtype_wrapper final {
    mag_dtype_t v;

    constexpr mag_dtype_t operator *() const noexcept { return v; }
  };

  template <typename F>
  class on_scope_exit final {
  public:
    static_assert(std::is_invocable_v<F &>, "on_scope_exit requires a callable that can be invoked with no arguments.");

    explicit on_scope_exit(F &&f) noexcept(std::is_nothrow_move_constructible_v<F>) : m_callback{std::forward<F>(f)}, m_active{true} {}
    explicit on_scope_exit(const F &f) noexcept(std::is_nothrow_copy_constructible_v<F>) : m_callback{f}, m_active{true} {}
    on_scope_exit(const on_scope_exit &) = delete;
    on_scope_exit& operator=(const on_scope_exit &) = delete;
    on_scope_exit(on_scope_exit &&other) noexcept(std::is_nothrow_move_constructible_v<F>) : m_callback{std::move(other.m_callback)}, m_active{other.m_active} {
      other.m_active = false;
    }
    on_scope_exit& operator=(on_scope_exit&&) = delete;
    ~on_scope_exit() {
      if (!m_active) return;
      if constexpr (std::is_nothrow_invocable_v<F&>) std::invoke(m_callback);
      else try {  std::invoke(m_callback); } catch (...) { std::terminate(); }
    }
    void dismiss() noexcept { m_active = false; }
    [[nodiscard]] bool active() const noexcept { return m_active; }

  private:
    F m_callback {};
    bool m_active {};
  };

  struct tensor_wrapper final {
    mag_tensor_t *p = nullptr;

    constexpr tensor_wrapper() noexcept = default;
    explicit tensor_wrapper(mag_tensor_t *ptr) noexcept : p{ptr} {}
    tensor_wrapper(const tensor_wrapper &other) noexcept : p{other.p} { if (p) mag_tensor_incref(p); }
    constexpr tensor_wrapper(tensor_wrapper &&other) noexcept : p{other.p} { other.p = nullptr; }
    tensor_wrapper &operator=(const tensor_wrapper &other) noexcept {
      if (this != &other) {
        if (other.p) mag_tensor_incref(other.p);
        if (p) mag_tensor_decref(p);
        p = other.p;
      }
      return *this;
    }
    tensor_wrapper &operator=(tensor_wrapper &&other) noexcept {
      if (this != &other) {
        if (p) mag_tensor_decref(p);
        p = other.p;
        other.p = nullptr;
      }
      return *this;
    }
    ~tensor_wrapper() {
      if (p) mag_tensor_decref(p);
    }
    explicit constexpr operator bool() const noexcept { return p != nullptr; }
    constexpr mag_tensor_t *operator * () const noexcept { return p; }
  };
  static_assert(sizeof(tensor_wrapper) == sizeof(mag_tensor_t *), "tensor_wrapper should have the same size as a raw pointer.");

  struct reduction_axes final {
    std::vector<int64_t> storage {};
    const int64_t *ptr = nullptr;
    int64_t rank = 0;
  };

  [[nodiscard]] extern nb::tuple tuple_from_i64_span(const int64_t *p, Py_ssize_t n);
  [[nodiscard]] extern tensor_wrapper tensor_from_py_scalar(nb::handle obj, mag_dtype_t dt, mag_device_id_t device);
  [[nodiscard]] extern tensor_wrapper normalize_rhs_to_tensor(const tensor_wrapper &lhs, nb::handle rhs);
  [[nodiscard]] extern std::vector<int64_t> parse_shape_from_args(const nb::args &args);
  [[nodiscard]] extern std::vector<int64_t> parse_i64_dims(const nb::args &args, const char *what);
  [[nodiscard]] extern std::vector<int64_t> parse_i64_list_handle(nb::handle h, const char *what);
  [[nodiscard]] extern reduction_axes parse_reduction_axes(nb::handle dim_h);
  [[nodiscard]] extern mag_scalar_t scalar_from_py_number(nb::handle h);
  [[nodiscard]] extern nb::object py_scalar_from_mag_scalar(const mag_scalar_t &scalar);
  [[nodiscard]] extern dtype_wrapper deduce_dtype_from_py_scalar(nb::handle h);
  [[nodiscard]] extern std::string format_error_msg(const mag_error_t &err);
  [[nodiscard]] extern std::optional<mag_device_id_t> parse_device_id_str(std::string &&str);
  extern void validate_shape(const std::vector<int64_t> &shape);
  extern void validate_shape(const std::vector<int64_t> &shape);
  extern void validate_shape_infer_one(const std::vector<int64_t> &shape, const char *op);;

  inline void throw_if_error(mag_status_t st, const mag_error_t &err) {
    if (st == MAG_STATUS_OK) return;
    std::string msg = format_error_msg(err);
    throw std::runtime_error {msg.c_str()};
  }
}

namespace nanobind::detail { // Specialization for magnetron's half precision types
  template <> struct dtype_traits<mag_float16_t> {
    static constexpr dlpack::dtype value {
      static_cast<uint8_t>(dlpack::dtype_code::Float),
      16,
      1
    };
    static constexpr auto name = const_name("float16");
  };
  template <> struct dtype_traits<mag_bfloat16_t> {
    static constexpr dlpack::dtype value {
      static_cast<uint8_t>(dlpack::dtype_code::Bfloat),
      16,
      1
    };
    static constexpr auto name = const_name("bfloat16");
  };
  template <> struct dtype_traits<mag_float8_e4m3fn_t> {
    static constexpr dlpack::dtype value {
      static_cast<uint8_t>(dlpack::dtype_code::Float8_E4M3FN),
      8,
      1
    };
    static constexpr auto name = const_name("float8_e4m3fn");
  };
}
