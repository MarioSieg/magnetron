/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
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
#include <string>
#include <memory>
#include <vector>

#include <magnetron/magnetron.h>

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

// Lazy init the context, destruction is handled by the module destructor. Im
[[nodiscard]] extern mag_context_t *get_ctx();

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

