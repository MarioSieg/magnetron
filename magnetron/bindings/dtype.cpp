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

        nb::class_<dtype_wrapper>{dtype, "DType", "Data type descriptor (e.g. float32, int64, boolean)."}
            .def_prop_ro("id", [](const dtype_wrapper &self) noexcept -> int { return self.v; }, "Internal type id.")
            .def_prop_ro("name", [](const dtype_wrapper &self) noexcept -> const char * { return mag_type_trait(self.v)->name; }, "Full name (e.g. float32).")
            .def_prop_ro("short_name", [](const dtype_wrapper &self) noexcept -> const char * { return mag_type_trait(self.v)->short_name; }, "Short name for display.")
            .def_prop_ro("size", [](const dtype_wrapper &self) noexcept -> size_t { return mag_type_trait(self.v)->size; }, "Size in bytes.")
            .def_prop_ro("alignment", [](const dtype_wrapper &self) noexcept -> size_t { return mag_type_trait(self.v)->alignment; }, "Alignment in bytes.")
            .def("__repr__", [](const dtype_wrapper &self) -> nb::str { return nb::str{"magnetron.dtype.{}"}.format(mag_type_trait(self.v)->name); })
            .def("is_floating_point", [](const dtype_wrapper &self) noexcept -> bool { return mag_type_category_is_floating_point(self.v); }, "True for float types.")
            .def("is_unsigned_integer", [](const dtype_wrapper &self) noexcept -> bool { return mag_type_category_is_unsigned_integer(self.v); }, "True for unsigned int types.")
            .def("is_signed_integer", [](const dtype_wrapper &self) noexcept -> bool { return mag_type_category_is_signed_integer(self.v); }, "True for signed int types.")
            .def("is_integer", [](const dtype_wrapper &self) noexcept -> bool { return mag_type_category_is_integer(self.v); }, "True for any integer type.")
            .def("is_integral", [](const dtype_wrapper &self) noexcept -> bool { return mag_type_category_is_integral(self.v); }, "True for int or bool.")
            .def("is_numeric", [](const dtype_wrapper &self) noexcept -> bool { return mag_type_category_is_numeric(self.v); }, "True for numeric dtypes.")
            .def("__int__", [](const dtype_wrapper &self) noexcept -> int { return self.v; })
            .def("__hash__", [](const dtype_wrapper &self) noexcept -> size_t { return self.v; })
            .def("__eq__", [](const dtype_wrapper &a, const dtype_wrapper &b) noexcept -> bool { return a.v == b.v; });

        // Bind all dtypes
        static_assert(MAG_DTYPE_FLOAT32 == 0);
        nb::set all_types {};
        for (int dt=MAG_DTYPE_FLOAT32; dt < MAG_DTYPE__NUM; ++dt) {
            auto dte = static_cast<mag_dtype_t>(dt);
            auto attr = dtype.attr(mag_type_trait(dte)->name);
            attr = nb::cast(dtype_wrapper{dte});
            all_types.add(attr);
        }
        dtype.attr("all_types") = all_types;
    }
}
