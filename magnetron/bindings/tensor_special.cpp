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

namespace mag::bindings {
    void init_tensor_special_methods(nb::class_<tensor_wrapper> &cls) {
        cls
       .def("__len__", [](const tensor_wrapper &self) -> int64_t {
           if (mag_tensor_rank(*self) == 0)
               throw nb::value_error("Tensor must have at least one dimension to use len()");
           return *mag_tensor_shape_ptr(*self);
       })
       .def("__str__", [](const tensor_wrapper &self) -> nb::str {
           const char *cstr = mag_tensor_to_string(*self, 3, 3, 1000);
           if (!cstr) throw std::runtime_error {"Failed to convert tensor to string"};
           on_scope_exit defer_free {[cstr] { mag_tensor_to_string_free_data(cstr); }};
           auto str = nb::str{cstr};
           return str;
       })
       .def("__repr__", [](const tensor_wrapper &self) -> nb::str {
           const char *cstr = mag_tensor_to_string(*self, 3, 3, 1000);
           if (!cstr) throw std::runtime_error {"Failed to convert tensor to string"};
           on_scope_exit defer_free {[cstr] { mag_tensor_to_string_free_data(cstr); }};
           auto str = nb::str {cstr};
           return str;
       })
       .def("__bool__", [](const tensor_wrapper &self) -> bool {
           if (mag_tensor_numel(*self) != 1) {
               throw nb::value_error(
                   "The truth value of a Tensor with more than one element is ambiguous. "
                   "Use .any() or .all() instead."
               );
           }
           mag_scalar_t s {};
           throw_if_error(mag_tensor_item(*self, &s));
           if (mag_scalar_is_f64(s)) return mag_scalar_as_f64(s) != 0.0;
           if (mag_scalar_is_i64(s)) return mag_scalar_as_i64(s) != 0;
           if (mag_scalar_is_u64(s)) return mag_scalar_as_u64(s) != 0;
           throw nb::type_error("Unsupported scalar type for __bool__()");
       });
    }
}
