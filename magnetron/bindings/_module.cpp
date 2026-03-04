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

#include "prelude.hpp"

extern void mag_init_bindings_context(nb::module_ &m);
extern void mag_init_bindings_dtype(nb::module_ &m);
extern void mag_init_bindings_tensor(nb::module_ &m);

// Global module entry defin
NB_MODULE(magnetron, m) {
    mag_init_bindings_context(m);
    mag_init_bindings_dtype(m);
    mag_init_bindings_tensor(m);
}

