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
    void init_bindings_dtype(nb::module_ &m) {
        auto dtype = m.def_submodule(
           "dtype",
           "Contains all data type definitions and related utilities."
        );

        nb::class_<dtype_wrapper>{dtype, "DType"}
            .def_prop_ro("id", [](const dtype_wrapper &self) noexcept -> int { return self.v; })
            .def_prop_ro("name", [](const dtype_wrapper &self) noexcept -> const char * { return mag_type_trait(self.v)->name; })
            .def_prop_ro("short_name", [](const dtype_wrapper &self) noexcept -> const char * { return mag_type_trait(self.v)->short_name; })
            .def_prop_ro("size", [](const dtype_wrapper &self) noexcept -> size_t { return mag_type_trait(self.v)->size; })
            .def_prop_ro("alignment", [](const dtype_wrapper &self) noexcept -> size_t { return mag_type_trait(self.v)->alignment; })
            .def("__repr__", [](const dtype_wrapper &self) -> nb::str { return nb::str{"magnetron.dtype.{}"}.format(mag_type_trait(self.v)->name); })
            .def("__int__", [](const dtype_wrapper &self) noexcept -> int { return self.v; })
            .def("__hash__", [](const dtype_wrapper &self) noexcept -> size_t { return self.v; })
            .def("__eq__", [](const dtype_wrapper &a, const dtype_wrapper &b) noexcept -> bool { return a.v == b.v; });

        // Bind all dtypes
        static_assert(MAG_DTYPE_FLOAT32 == 0);
        for (int dt=MAG_DTYPE_FLOAT32; dt < MAG_DTYPE__NUM; ++dt) {
            auto dte = static_cast<mag_dtype_t>(dt);
            dtype.attr(mag_type_trait(dte)->name) = nb::cast(dtype_wrapper{dte});
        }
    }
}
