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

#include <atomic>
#include <functional>
#include <exception>
#include <mutex>
#include <sstream>
#include <string>
#include <memory>
#include <vector>

#include <magnetron/magnetron.h>

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace mag::bindings {
    namespace nb = nanobind;
    using namespace nb::literals;

    // Lazy init the context, destruction is handled by the module destructor.
    [[nodiscard]] extern mag_context_t *get_ctx();
    [[nodiscard]] extern std::mutex &get_global_mutex();

    struct dtype_wrapper final { mag_dtype_t v; };

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

    struct reduction_axes final {
        std::vector<int64_t> storage {};
        const int64_t *ptr = nullptr;
        int64_t rank = 0;
    };

    [[nodiscard]] extern nb::tuple tuple_from_i64_span(const int64_t *p, Py_ssize_t n);
    [[nodiscard]] extern tensor_wrapper tensor_from_py_scalar(nb::handle obj, mag_dtype_t dt);
    [[nodiscard]] extern tensor_wrapper normalize_rhs_to_tensor(const tensor_wrapper &lhs, nb::handle rhs);
    [[nodiscard]] extern std::vector<int64_t> parse_shape_from_args(const nb::args &args);
    extern void validate_shape(const std::vector<int64_t> &shape);
    [[nodiscard]] extern std::vector<int64_t> parse_i64_dims(const nb::args &args, const char *what);
    [[nodiscard]] extern std::vector<int64_t> parse_i64_list_handle(nb::handle h, const char *what);
    [[nodiscard]] extern reduction_axes parse_reduction_axes(nb::handle dim_h);
    [[nodiscard]] extern mag_scalar_t scalar_from_py(nb::handle h);
    [[nodiscard]] extern dtype_wrapper deduce_dtype_from_py_scalar(nb::handle h);
    [[nodiscard]] extern std::string format_error_msg(const mag_error_t &err);

    inline void throw_if_error(mag_status_t st, const mag_error_t &err) {
        if (st == MAG_STATUS_OK) return;
        std::string msg = format_error_msg(err);
        throw std::runtime_error {msg.c_str()};
    }
}
