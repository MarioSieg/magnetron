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
#include <mutex>
#include <string>
#include <memory>
#include <vector>

#include <magnetron/magnetron.h>
#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

// Lazy init the context, destruction is handled by the module destructor. Im
[[nodiscard]] extern mag_context_t *get_ctx();
