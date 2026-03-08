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
NB_MODULE(_magnetron_bindings, m) {
    std::lock_guard lock {mag::bindings::get_global_mutex()};

    m.doc() = "A compact, bloat-free machine learning framework with CPU and CUDA acceleration.";

    // Export metadata
    std::array<char, 64> version_buf {};
    std::snprintf(version_buf.data(), version_buf.size(), "%d.%d.%d", mag_ver_major(MAG_VERSION), mag_ver_minor(MAG_VERSION), mag_ver_patch(MAG_VERSION));
    m.attr("__version__") = std::string{version_buf.data()};
    std::snprintf(version_buf.data(), version_buf.size(), "%d.%d.%d", mag_ver_major(MAG_SNAPSHOT_VERSION), mag_ver_minor(MAG_SNAPSHOT_VERSION), mag_ver_patch(MAG_SNAPSHOT_VERSION));
    m.attr("__snapshot_version__") = std::string{version_buf.data()};
    m.attr("__author__") = "Mario Sieg";
    m.attr("__email__") = "mario.sieg.64@gmail.com";
    m.attr("__author_email__") = "mario.sieg.64@gmail.com";
    m.attr("__license__") = "Apache 2.0";
    m.attr("__url__") = "https://github.com/MarioSieg/magnetron";

    // Export all submodules and bindings
    mag::bindings::init_bindings_context(m);
    mag::bindings::init_bindings_dtype(m);
    mag::bindings::init_bindings_tensor(m);
    mag::bindings::init_bindings_snapshot(m);
}

