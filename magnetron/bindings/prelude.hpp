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

#include <algorithm>
#include <array>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <magnetron/magnetron.h>
#include <core/mag_bfloat16.h>
#include <core/mag_float16.h>
#include <core/mag_float8_e4m3fn.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

namespace mag::bindings {
  namespace nb = nanobind;
  using namespace nb::literals;

  // Lazy init the context, destruction is handled by the module destructor.
  [[nodiscard]] extern mag_context_t *get_ctx();
  extern void release_ctx_if_pending() noexcept;
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

  class tensor_wrapper final {
  public:
    constexpr tensor_wrapper() noexcept = default;
    explicit tensor_wrapper(mag_tensor_t *ptr) noexcept : m_tensor{ptr} {}
    tensor_wrapper(const tensor_wrapper &other) noexcept : m_tensor{other.m_tensor} { if (m_tensor) mag_tensor_incref(m_tensor); }
    constexpr tensor_wrapper(tensor_wrapper &&other) noexcept : m_tensor{other.m_tensor} { other.m_tensor = nullptr; }
    tensor_wrapper &operator=(const tensor_wrapper &other) noexcept {
      if (this != &other) {
        if (other.m_tensor) mag_tensor_incref(other.m_tensor);
        release(m_tensor);
        m_tensor = other.m_tensor;
      }
      return *this;
    }
    tensor_wrapper &operator=(tensor_wrapper &&other) noexcept {
      if (this != &other) {
        release(m_tensor);
        m_tensor = other.m_tensor;
        other.m_tensor = nullptr;
      }
      return *this;
    }
    ~tensor_wrapper() {
      release(m_tensor);
    }
    void reset_owned(mag_tensor_t *next) noexcept {
      release(__atomic_exchange_n(&m_tensor, next, __ATOMIC_ACQ_REL));
    }
    void replace_shared(const tensor_wrapper &other) noexcept {
      mag_tensor_t *next = __atomic_load_n(&other.m_tensor, __ATOMIC_ACQUIRE);
      if (next) mag_tensor_incref(next);
      reset_owned(next);
    }

    explicit constexpr operator bool() const noexcept { return m_tensor != nullptr; }
    constexpr mag_tensor_t *operator * () const noexcept { return m_tensor; }
    constexpr mag_tensor_t *&operator * () noexcept { return m_tensor; }

  private:
    static void release(mag_tensor_t *t) noexcept {
      if (!t) return;
      mag_tensor_decref(t);
      release_ctx_if_pending();
    }

    mag_tensor_t *m_tensor = nullptr;
  };
  static_assert(sizeof(tensor_wrapper) == sizeof(mag_tensor_t *), "tensor_wrapper should have the same size as a raw pointer.");

  struct fixed_dim_vec final {
    std::array<int64_t, MAG_MAX_DIMS> dims {};
    int64_t numel = 0;

    constexpr fixed_dim_vec() noexcept = default;
    explicit fixed_dim_vec(size_t count, int64_t value = 0) {
      if (mag_unlikely(count > MAG_MAX_DIMS))
        throw nb::value_error(("too many dimensions, must be <= " + std::to_string(MAG_MAX_DIMS)).c_str());
      std::fill_n(dims.begin(), count, value);
      numel = static_cast<int64_t>(count);
    }
    fixed_dim_vec(std::initializer_list<int64_t> init) {
      if (mag_unlikely(init.size() > MAG_MAX_DIMS))
        throw nb::value_error(("too many dimensions, must be <= " + std::to_string(MAG_MAX_DIMS)).c_str());
      std::copy(init.begin(), init.end(), dims.begin());
      numel = static_cast<int64_t>(init.size());
    }
    template <typename It, typename = std::enable_if_t<!std::is_integral_v<It>>> /* Guard so (count, value) never resolves here */
    fixed_dim_vec(It first, It last) {
      auto count = static_cast<size_t>(std::distance(first, last));
      if (mag_unlikely(count > MAG_MAX_DIMS))
        throw nb::value_error(("too many dimensions, must be <= " + std::to_string(MAG_MAX_DIMS)).c_str());
      std::copy(first, last, dims.begin());
      numel = static_cast<int64_t>(count);
    }

    [[nodiscard]] int64_t *data() noexcept { return dims.data(); }
    [[nodiscard]] const int64_t *data() const noexcept { return dims.data(); }
    [[nodiscard]] size_t size() const noexcept { return static_cast<size_t>(numel); }
    [[nodiscard]] int64_t rank() const noexcept { return numel; }
    [[nodiscard]] bool empty() const noexcept { return numel == 0; }
    [[nodiscard]] int64_t *begin() noexcept { return dims.data(); }
    [[nodiscard]] int64_t *end() noexcept { return dims.data()+numel; }
    [[nodiscard]] const int64_t *begin() const noexcept { return dims.data(); }
    [[nodiscard]] const int64_t *end() const noexcept { return dims.data()+numel; }
    [[nodiscard]] int64_t &operator[](size_t i) noexcept { return dims[i]; }
    [[nodiscard]] int64_t operator[](size_t i) const noexcept { return dims[i]; }
    [[nodiscard]] int64_t &front() noexcept { return dims[0]; }
    [[nodiscard]] int64_t &back() noexcept { return dims[numel-1]; }
    [[nodiscard]] int64_t front() const noexcept { return dims[0]; }
    [[nodiscard]] int64_t back() const noexcept { return dims[numel-1]; }
    void clear() noexcept { numel = 0; }
    void pop_back() {
      if (mag_unlikely(!numel))
        throw nb::index_error("pop_back on empty dimension vector");
      --numel;
    }

    void emplace_back(int64_t x) {
      if (mag_unlikely(numel >= MAG_MAX_DIMS))
        throw nb::value_error(("too many dimensions, must be <= " + std::to_string(MAG_MAX_DIMS)).c_str());
      dims[numel++] = x;
    }

    template <typename It>
    int64_t *insert(const int64_t *pos, It first, It last) {
      auto at = static_cast<size_t>(pos - dims.data());
      auto count = static_cast<size_t>(std::distance(first, last));
      if (mag_unlikely(size() + count > MAG_MAX_DIMS))
        throw nb::value_error(("too many dimensions, must be <= " + std::to_string(MAG_MAX_DIMS)).c_str());
      std::array<int64_t, MAG_MAX_DIMS> src {}; /* Copy first so inserting a range of *this is safe */
      std::copy(first, last, src.begin());
      std::copy_backward(dims.begin()+at, dims.begin()+numel, dims.begin()+numel+count);
      std::copy_n(src.begin(), count, dims.begin()+at);
      numel += static_cast<int64_t>(count);
      return dims.data()+at;
    }

    void assign(const int64_t *p, int64_t count) {
      if (count > MAG_MAX_DIMS)
        throw nb::value_error(("too many dimensions, must be <= " + std::to_string(MAG_MAX_DIMS)).c_str());
      std::copy_n(p, count, dims.begin());
      numel = count;
    }

    template <typename T>
    [[nodiscard]] std::vector<T> to_vec() const {
      std::vector<T> out {};
      out.reserve(static_cast<size_t>(numel));
      for (int64_t i=0; i < numel; ++i) out.emplace_back(static_cast<T>(dims[i]));
      return out;
    }
  };

  struct reduction_axes final {
    std::vector<int64_t> storage {};
    const int64_t *ptr = nullptr;
    int64_t rank = 0;
  };

  [[nodiscard]] extern nb::tuple tuple_from_i64_span(const int64_t *p, Py_ssize_t n);
  [[nodiscard]] extern tensor_wrapper tensor_from_py_scalar(nb::handle obj, mag_dtype_t dt, mag_device_id_t device);
  [[nodiscard]] extern tensor_wrapper normalize_rhs_to_tensor(const tensor_wrapper &lhs, nb::handle rhs);
  extern void promote_ops(std::initializer_list<tensor_wrapper *> ops);
  [[nodiscard]] extern fixed_dim_vec parse_shape_from_args(const nb::args &args);
  [[nodiscard]] extern fixed_dim_vec parse_i64_dims(const nb::args &args, const char *what);
  [[nodiscard]] extern std::vector<int64_t> parse_i64_list_handle(nb::handle h, const char *what);
  [[nodiscard]] extern reduction_axes parse_reduction_axes(nb::handle dim_h);
  [[nodiscard]] extern mag_scalar_t scalar_from_py_number(nb::handle h);
  [[nodiscard]] extern nb::object py_scalar_from_mag_scalar(const mag_scalar_t &scalar);
  [[nodiscard]] extern dtype_wrapper deduce_dtype_from_py_scalar(nb::handle h);
  [[nodiscard]] extern std::string format_error_msg(const mag_error_t &err);
  [[nodiscard]] extern std::optional<mag_device_id_t> parse_device_id_str(const std::string &str, bool *out_has_ordinal = nullptr);
  [[nodiscard]] extern std::optional<mag_device_id_t> resolve_device_id_str(const std::string &str);
  extern void validate_shape(const fixed_dim_vec &shape);
  extern void validate_shape(const std::vector<int64_t> &shape);
  extern void validate_shape_infer_one(const fixed_dim_vec &shape, const char *op);

  constexpr int64_t gil_release_numel_threshold = 1000;

  template <typename F>
  [[nodiscard]] mag_status_t invoke_no_gil(F &&fn) noexcept(std::is_nothrow_invocable_r_v<mag_status_t, F>) {
    static_assert(std::is_invocable_r_v<mag_status_t, F>);
    nb::gil_scoped_release nogil {};
    return std::invoke(fn);
  }

  template <typename F>
  [[nodiscard]] mag_status_t invoke_auto_gil(const tensor_wrapper &x, F &&fn) {
    static_assert(std::is_invocable_r_v<mag_status_t, F>);
    if (mag_tensor_numel(*x) <= gil_release_numel_threshold) return std::invoke(fn);
    nb::gil_scoped_release nogil {};
    return std::invoke(fn);
  }

  template <typename F>
  [[nodiscard]] mag_status_t invoke_auto_gil(const tensor_wrapper &x, const tensor_wrapper &y, F &&fn) {
    static_assert(std::is_invocable_r_v<mag_status_t, F>);
    if (mag_tensor_numel(*x) <= gil_release_numel_threshold && mag_tensor_numel(*y) <= gil_release_numel_threshold) return std::invoke(fn);
    nb::gil_scoped_release nogil {};
    return std::invoke(fn);
  }

  inline void throw_if_error(mag_status_t st, const mag_error_t &err) {
    if (mag_likely(st == MAG_OK)) return;
    std::string msg = format_error_msg(err);
    throw std::runtime_error {msg.c_str()};
  }

  template <typename F, typename... Args>
  [[nodiscard]] tensor_wrapper call_op(F &&fn, Args &&...args) {
    static_assert(std::is_invocable_r_v<mag_status_t, F, mag_error_t *, mag_tensor_t **, Args...>);
    mag_tensor_t *out = nullptr;
    mag_error_t err {};
    throw_if_error(std::invoke(std::forward<F>(fn), &err, &out, std::forward<Args>(args)...), err);
    return tensor_wrapper{out};
  }

  template <typename F, typename... Args>
  [[nodiscard]] tensor_wrapper call_op_maybe_without_gil(const tensor_wrapper &x, F &&fn, Args &&...args) {
    static_assert(std::is_invocable_r_v<mag_status_t, F, mag_error_t *, mag_tensor_t **, Args...>);
    mag_tensor_t *out = nullptr;
    mag_error_t err {};
    throw_if_error(invoke_auto_gil(x, [&] { return std::invoke(fn, &err, &out, args...); }), err);
    return tensor_wrapper{out};
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
