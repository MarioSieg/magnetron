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
                   "The truth value of a Tensor with more than one element is ambiguous. Use .any() or .all() instead."
               );
           }
           mag_scalar_t s {};
           throw_if_error(mag_tensor_item(*self, &s));
           if (mag_scalar_is_f64(s)) return mag_scalar_as_f64(s) != 0.0;
           if (mag_scalar_is_i64(s)) return mag_scalar_as_i64(s) != 0;
           if (mag_scalar_is_u64(s)) return mag_scalar_as_u64(s) != 0;
           throw nb::type_error("Unsupported scalar type for __bool__()");
        })
       .def("__getitem__", [](const tensor_wrapper &self, nb::object index) -> tensor_wrapper {
            auto indices = nb::isinstance<nb::tuple>(index) ? nb::cast<nb::tuple>(index) : nb::make_tuple(index);
            tensor_wrapper curr = self;
            int64_t ax = 0;
            size_t n = nb::len(indices);
            bool has_ellipsis = false;
            for (size_t i=0; i < n; ++i) {
                const auto &idx = indices[i];
                if (idx.ptr() == Py_Ellipsis) {
                    if (has_ellipsis)
                        throw nb::value_error("Only one ellipsis allowed in indices");
                    has_ellipsis = true;
                    ax += mag_tensor_rank(*curr) - (n-i-1) - ax;;
                } else if (idx.is_none()) {
                    mag_tensor_t *out = nullptr;
                    throw_if_error(mag_unsqueeze(&out, *curr, ax));
                    curr = tensor_wrapper{out};
                    ++ax;
                } else if (nb::isinstance<nb::int_>(idx)) {
                    auto i = nb::cast<int64_t>(idx);
                    int64_t dim_size = mag_tensor_shape_ptr(*curr)[ax];
                    if (i < 0) i += dim_size;
                    if (i < 0 || i >= dim_size)
                        throw nb::index_error("Index out of bounds");
                    mag_tensor_t *tmp = nullptr;
                    throw_if_error(mag_view_slice(&tmp, *curr, ax, i, 1, 1));
                    on_scope_exit defer {[tmp] { mag_tensor_decref(tmp); }};
                    mag_tensor_t *out = nullptr;
                    throw_if_error(mag_squeeze_dim(&out, tmp, ax));
                    curr = tensor_wrapper{out};
                } else if (nb::isinstance<nb::slice>(idx)) {
                    auto s = nb::cast<nb::slice>(idx);
                    auto [start, stop, step, length] = s.compute(mag_tensor_shape_ptr(*curr)[ax]);
                    if (step <= 0)
                        throw nb::value_error("Non-positive slice steps are not supported");
                    if (length == 0)
                        throw nb::value_error("Zero-length slice not implemented");
                    mag_tensor_t *out = nullptr;
                    throw_if_error(mag_view_slice(&out, *curr, ax, start, length, step));
                    curr = tensor_wrapper{out};
                    ++ax;
                } else if (nb::isinstance<tensor_wrapper>(idx)) {
                    auto idx_tensor = nb::cast<tensor_wrapper>(idx);
                    mag_tensor_t *out = nullptr;
                    throw_if_error(mag_gather(&out, *curr, ax, *idx_tensor));
                    curr = tensor_wrapper{out};
                    ++ax;
                } else if (nb::isinstance<nb::sequence>(idx)) {
                    auto seq = nb::cast<nb::sequence>(idx);
                    std::vector<int64_t> data;
                    data.reserve(nb::len(seq));
                    for (auto &&v : seq)
                        data.emplace_back(nb::cast<int64_t>(v));
                    auto shape = static_cast<int64_t>(data.size());
                    mag_tensor_t *idx_tensor = nullptr;
                    throw_if_error(mag_empty(&idx_tensor, get_ctx(), MAG_DTYPE_INT64, 1, &shape));
                    throw_if_error(mag_copy_raw_(idx_tensor, data.data(), data.size() * sizeof(int64_t)));
                    mag_tensor_t *out = nullptr;
                    throw_if_error(mag_gather(&out, *curr, ax, idx_tensor));
                    curr = tensor_wrapper{out};
                    ++ax;
                } else {
                    throw nb::type_error("Invalid index component");
                }
            }
            return curr;
        });
    }
}
