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

#include "prelude.hpp"

namespace mag::bindings {
    extern void init_bindings_context(nb::module_ &m);
    extern void init_bindings_dtype(nb::module_ &m);
    extern void init_bindings_tensor(nb::module_ &m);
    extern void init_bindings_snapshot(nb::module_ &m);
}

// Global module entry definition
NB_MODULE(_magnetron, m) {
    std::lock_guard lock {mag::bindings::get_global_mutex()};
    mag::bindings::init_bindings_context(m);
    mag::bindings::init_bindings_dtype(m);
    mag::bindings::init_bindings_tensor(m);
    mag::bindings::init_bindings_snapshot(m);
}

