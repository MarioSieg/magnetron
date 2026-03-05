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
#include "core/mag_tensor.h"

namespace mag::bindings {
    template <typename T, typename PT>
    [[nodiscard]] static nb::object build_list_recursive(
        const T *data,
        const int64_t *shape,
        const int64_t *strides,
        int64_t rank,
        int64_t offset,
        int64_t dim
    ) {
        if (dim == rank) return PT(data[offset]);
        int64_t size = shape[dim];
        PyObject *raw = PyList_New(size); // We use the raw Python API to preallocate the list's capacity
        if (!raw) throw std::runtime_error {"Failed to allocate list for tolist()"};
        for (int64_t i=0; i < size; ++i) {
            nb::object item = build_list_recursive<T, PT>(
                data,
                shape,
                strides,
                rank,
                offset + i*strides[dim],
                dim + 1
            );
            PyList_SET_ITEM(raw, i, item.release().ptr());
        }
        return nb::steal<nb::object>(raw);
    }

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
            if (mag_tensor_numel(*self) != 1)
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
        })
        .def("tolist", [](const tensor_wrapper &self) -> nb::object {
            if (!mag_tensor_numel(*self)) return nb::list();
            auto *tensor = *self;
            if (tensor->storage->device->id.type != MAG_BACKEND_TYPE_CPU)
                throw nb::value_error("tolist() only supports CPU tensors");
            enum class CastedTypeFamily { Float, Int, Bool };
            CastedTypeFamily casted_type {};
            mag_tensor_t *contig = nullptr;
            {
                mag_tensor_t *casted = nullptr;
                if (mag_tensor_is_floating_point_typed(tensor)) {
                   throw_if_error(mag_cast(&casted, *self, MAG_DTYPE_FLOAT32));
                   casted_type = CastedTypeFamily::Float;
                } else if (mag_tensor_is_integer_typed(tensor)) {
                   throw_if_error(mag_cast(&casted, *self, MAG_DTYPE_INT64));
                   casted_type = CastedTypeFamily::Int;
                } else if (mag_tensor_type(tensor) == MAG_DTYPE_BOOLEAN) {
                   throw_if_error(mag_cast(&casted, *self, MAG_DTYPE_UINT8));
                   casted_type = CastedTypeFamily::Bool;
                } else throw nb::type_error("Unsupported dtype for tolist()");
                on_scope_exit defer_decref {[casted] { mag_tensor_decref(casted); }};
                throw_if_error(mag_contiguous(&contig, casted));
            }
            on_scope_exit defer_decref2 {[contig] { mag_tensor_decref(contig); }};
            const auto *ptr = reinterpret_cast<const void *>(mag_tensor_data_ptr(contig));
            int64_t rank = mag_tensor_rank(contig);
            if (rank == 0) { // Handle scalars
                switch (casted_type) {
                    case CastedTypeFamily::Float: return nb::float_{*static_cast<const float *>(ptr)};
                    case CastedTypeFamily::Int: return nb::int_{*static_cast<const int64_t *>(ptr)};
                    case CastedTypeFamily::Bool: return nb::bool_{static_cast<bool>(*static_cast<const uint8_t *>(ptr))};
                }
            }
            const int64_t *shape = mag_tensor_shape_ptr(contig);
            std::vector<int64_t> strides(rank);
            strides[rank-1] = 1;
            for (int64_t i=rank-2; i >= 0; --i) // RowMaj strides
                strides[i] = strides[i+1]*shape[i+1];
            nb::object result {};
            switch (casted_type) {
                case CastedTypeFamily::Float: result = build_list_recursive<float, nb::float_>(static_cast<const float *>(ptr), shape, strides.data(), rank, 0, 0); break;
                case CastedTypeFamily::Int: result = build_list_recursive<int64_t, nb::int_>( static_cast<const int64_t *>(ptr), shape, strides.data(), rank, 0, 0); break;
                case CastedTypeFamily::Bool: result = build_list_recursive<uint8_t, nb::bool_>(static_cast<const uint8_t *>(ptr), shape, strides.data(), rank, 0, 0); break;
            }
            return result;
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
