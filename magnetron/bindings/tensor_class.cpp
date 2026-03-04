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
    static void init_tensor_class_base(nb::class_<tensor_wrapper> &cls) {
        cls
        .def_prop_ro("rank", [](const tensor_wrapper &self) -> int64_t {
            return mag_tensor_rank(*self);
        })
        .def_prop_ro("numel", [](const tensor_wrapper &self) -> int64_t {
           return mag_tensor_numel(*self);
        })
        .def_prop_ro("numbytes", [](const tensor_wrapper &self) -> size_t {
           return mag_tensor_numbytes(*self);
        })
        .def_prop_ro("dtype", [](const tensor_wrapper &self) -> dtype_wrapper {
            return dtype_wrapper{ mag_tensor_type(*self) };
        })
        .def_prop_ro("is_transposed", [](const tensor_wrapper &self) -> bool {
           return mag_tensor_is_transposed(*self);
        })
        .def_prop_ro("is_permuted", [](const tensor_wrapper &self) -> bool {
           return mag_tensor_is_permuted(*self);
        })
        .def_prop_ro("is_view", [](const tensor_wrapper &self) -> bool {
           return mag_tensor_is_view(*self);
        })
        .def_prop_ro("is_contiguous", [](const tensor_wrapper &self) -> bool {
           return mag_tensor_is_contiguous(*self);
        })
        .def_prop_ro("shape",
        [](const tensor_wrapper &self) -> nb::tuple {
            return tuple_from_i64_span(mag_tensor_shape_ptr(*self), mag_tensor_rank(*self));
        })
        .def_prop_ro("strides",
        [](const tensor_wrapper &self) -> nb::tuple {
            return tuple_from_i64_span(mag_tensor_strides_ptr(*self), mag_tensor_rank(*self));
        })
        .def_prop_ro("data_ptr", [](const tensor_wrapper &self) -> uintptr_t {
            return mag_tensor_data_ptr(*self);
        })
        .def_prop_ro("data_ptr_mut", [](const tensor_wrapper &self) -> uintptr_t {
            return mag_tensor_data_ptr_mut(*self);
        })
        .def_prop_ro("data_storage_ptr", [](const tensor_wrapper &self) -> uintptr_t {
           return mag_tensor_data_storage_ptr(*self);
        })
        .def_prop_ro("data_storage_ptr_mut", [](const tensor_wrapper &self) -> uintptr_t {
           return mag_tensor_data_storage_ptr_mut(*self);
        })
        .def("can_broadcast", [](const tensor_wrapper &self, const tensor_wrapper &rhs) -> bool {
           return mag_tensor_can_broadcast(*self, *rhs);
        })
        .def_prop_rw("requires_grad", [](const tensor_wrapper &self) -> bool {
            return mag_tensor_requires_grad(*self);
        }, [](const tensor_wrapper &self, bool req) {
            mag_tensor_set_requires_grad(*self, req);
        })
        .def_prop_ro("grad", [](const tensor_wrapper &self) -> nb::object {
            mag_tensor_t *grad = nullptr;
            throw_if_error(mag_tensor_grad(*self, &grad));
            if (!grad) return nb::none();
            return nb::cast(tensor_wrapper {grad});
        })
        .def("backward", [](const tensor_wrapper &self) -> void {
            throw_if_error(mag_tensor_backward(*self));
        })
        .def("zero_grad", [](const tensor_wrapper &self) -> void {
            throw_if_error(mag_tensor_zero_grad(*self));
        })
        .def("item", [](const tensor_wrapper &self) -> nb::object {
            if (mag_tensor_numel(*self) != 0)
                throw nb::value_error("Tensor must have exactly one element to retrieve an item");
            mag_scalar_t s {};
            throw_if_error(mag_tensor_item(*self, &s));
            if (mag_scalar_is_f64(s)) return nb::float_(mag_scalar_as_f64(s));
            if (mag_scalar_is_i64(s)) return nb::int_(mag_scalar_as_i64(s));
            if (mag_scalar_is_u64(s)) {
                uint64_t v = mag_scalar_as_u64(s);
                if (mag_tensor_type(*self) == MAG_DTYPE_BOOLEAN)
                    return nb::bool_{v != 0};
                return nb::int_{v};
            }
            throw nb::type_error("Unsupported scalar type for item()");
        })
        .def("detach", [](const tensor_wrapper &self) -> tensor_wrapper {
            return tensor_wrapper{mag_tensor_detach(*self)};
        });
    }

    extern void init_tensor_special_methods(nb::class_<tensor_wrapper> &cls);
    extern void init_tensor_class_factories(nb::class_<tensor_wrapper> &cls);
    extern void init_tensor_class_operators(nb::class_<tensor_wrapper> &cls);

    void init_bindings_tensor(nb::module_ &m) {
        auto cls = nb::class_<tensor_wrapper>{m, "Tensor"};
        init_tensor_class_base(cls);
        init_tensor_special_methods(cls);
        init_tensor_class_factories(cls);
        init_tensor_class_operators(cls);
    }
}
