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
            .def_prop_ro("min", [](const dtype_wrapper &self) noexcept -> nb::object { return py_scalar_from_mag_scalar(mag_type_trait(self.v)->min_val); }, "Minimum representable value for this dtype.")
            .def_prop_ro("max", [](const dtype_wrapper &self) noexcept -> nb::object { return py_scalar_from_mag_scalar(mag_type_trait(self.v)->max_val); }, "Maximum representable value for this dtype.")
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

        for (std::underlying_type_t<mag_dtype_t> type=0; type < MAG_DTYPE__NUM; ++type) {
            auto dte = static_cast<mag_dtype_t>(type);
            dtype.attr(mag_type_trait(dte)->name) = nb::cast(dtype_wrapper{dte});
        }

        const auto bind_dtype_set = [&](const char *name, auto pred) -> void {
            nb::set s {};
            for (std::underlying_type_t<mag_dtype_t> type = 0; type < MAG_DTYPE__NUM; ++type) {
                auto dte = static_cast<mag_dtype_t>(type);
                if constexpr (!std::is_same_v<decltype(pred), std::nullptr_t>)
                    if (pred && !std::invoke(pred, dte)) continue;
                s.add(dtype.attr(mag_type_trait(dte)->name));
            }
            dtype.attr(name) = s;
        };

        bind_dtype_set("all", nullptr);
        bind_dtype_set("floating", [](mag_dtype_t dt) noexcept -> bool {
            return mag_type_category_is_floating_point(dt);
        });
        bind_dtype_set("unsigned", [](mag_dtype_t dt) noexcept -> bool {
            return mag_type_category_is_unsigned_integer(dt);
        });
        bind_dtype_set("signed", [](mag_dtype_t dt) noexcept -> bool {
            return mag_type_category_is_signed_integer(dt);
        });
        bind_dtype_set("integer", [](mag_dtype_t dt) noexcept -> bool {
            return mag_type_category_is_integer(dt);
        });
        bind_dtype_set("integral", [](mag_dtype_t dt) noexcept -> bool {
            return mag_type_category_is_integral(dt);
        });
        bind_dtype_set("numeric", [](mag_dtype_t dt) noexcept -> bool {
            return mag_type_category_is_numeric(dt);
        });
    }
}
