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

    [[nodiscard]] static nb::list expand_ellipsis(nb::tuple idxs, int64_t rank) {
        int64_t consuming = 0;
        int64_t ellipsis_occ = 0;
        int64_t ellipsis_pos = -1;
        for (size_t i=0; i < idxs.size(); ++i) {
            const auto &x = idxs[i];
            if (x.ptr() == Py_Ellipsis) {
                ++ellipsis_occ;
                if (ellipsis_pos < 0) ellipsis_pos = static_cast<int64_t>(i);
            } else if (x.is_none()) {
                // does not consume
            } else {
                consuming++;
            }
        }
        if (ellipsis_occ > 1)
            throw nb::index_error("Only one Ellipsis (...) is allowed in the index tuple");
        nb::list out {};
        if (ellipsis_occ == 1) {
            int64_t to_insert = rank - consuming;
            if (to_insert < 0)
                throw nb::index_error(("Too many indices for a tensor of rank " + std::to_string(rank)).c_str());
            for (int64_t i=0; i < ellipsis_pos; ++i)
                out.append(idxs[static_cast<size_t>(i)]);
            for (int64_t k=0; k < to_insert; ++k)
                out.append(nb::slice {nb::none(), nb::none(), nb::none()});
            for (int64_t i=ellipsis_pos+1; i < static_cast<int64_t>(idxs.size()); ++i)
                out.append(idxs[static_cast<size_t>(i)]);
        } else {
            if (consuming > rank)
                throw nb::index_error(("Too many indices for a tensor of rank " + std::to_string(rank)).c_str());
            if (consuming < rank) {
                for (auto &&idx : idxs)
                    out.append(idx);
                for (int64_t k=0; k < rank - consuming; ++k)
                    out.append(nb::slice(nb::none(), nb::none(), nb::none()));
            } else {
                for (auto && idx : idxs)
                    out.append(idx);
            }
        }
        return out;
    }

    [[nodiscard]] static mag_tensor_t *index_by_scalar_per_axis(mag_tensor_t *base, int64_t ax, int64_t i) {
        mag_tensor_t *tmp = nullptr;
        throw_if_error(mag_view_slice(&tmp, base, ax, i, 1, 1));
        int64_t rank = mag_tensor_rank(tmp);
        const int64_t *shape = mag_tensor_shape_ptr(tmp);
        std::vector<int64_t> ns {};
        ns.reserve(rank > 0 ? static_cast<size_t>(rank - 1) : 1);
        for (int64_t d=0; d < rank; ++d) {
            if (d == ax) continue;
            ns.emplace_back(shape[d]);
        }
        if (ns.empty()) ns.emplace_back(1);
        mag_tensor_t *out = nullptr;
        throw_if_error(mag_view(&out, tmp, ns.data(), static_cast<int64_t>(ns.size())));
        mag_tensor_decref(tmp);
        return out;
    };

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
            nb::tuple idxs_in = nb::isinstance<nb::tuple>(index)
                ? nb::cast<nb::tuple>(index)
                : nb::make_tuple(index);
            tensor_wrapper curr = self;
            int64_t rank0 = mag_tensor_rank(*curr);
            nb::list indices = expand_ellipsis(idxs_in, rank0);
            int64_t ax = 0;
            for (auto &&idx : indices) {
                if (idx.is_none()) {
                    mag_tensor_t *out = nullptr;
                    throw_if_error(mag_unsqueeze(&out, *curr, ax));
                    curr = tensor_wrapper{out};
                    ++ax;
                    continue;
                }
                if (nb::isinstance<nb::int_>(idx)) {
                    int64_t i = nb::cast<int64_t>(idx);
                    int64_t dim_size = mag_tensor_shape_ptr(*curr)[ax];
                    if (i < 0) i += dim_size;
                    if (i < 0 || i >= dim_size)
                        throw nb::index_error("Index out of bounds");
                    mag_tensor_t *out = index_by_scalar_per_axis(*curr, ax, i);
                    curr = tensor_wrapper{out};
                    continue;
                }
                if (nb::isinstance<nb::slice>(idx)) {
                    auto s = nb::cast<nb::slice>(idx);
                    auto [start, stop, step, length] = s.compute(mag_tensor_shape_ptr(*curr)[ax]);
                    if (step <= 0) throw nb::value_error("Non-positive slice steps are not supported");
                    if (length == 0) throw nb::value_error("Zero-length slice not implemented");
                    mag_tensor_t *out = nullptr;
                    throw_if_error(mag_view_slice(&out, *curr, ax, start, (int64_t) length, (int64_t) step));
                    curr = tensor_wrapper{out};
                    ++ax;
                    continue;
                }
                if (nb::isinstance<tensor_wrapper>(idx)) {
                    auto idx_tw = nb::cast<tensor_wrapper>(idx);
                    if (mag_tensor_numel(*idx_tw) == 1) {
                        mag_dtype_t it = mag_tensor_type(*idx_tw);
                        if (mag_tensor_is_integer_typed(*idx_tw) || it == MAG_DTYPE_BOOLEAN) {
                            mag_scalar_t s {};
                            throw_if_error(mag_tensor_item(*idx_tw, &s));
                            int64_t i = mag_scalar_is_i64(s) ? mag_scalar_as_i64(s)
                                       : mag_scalar_is_u64(s) ? (int64_t) mag_scalar_as_u64(s)
                                       : (throw nb::type_error("Tensor index scalar must be integer or bool"), 0);
                            int64_t dim_size = mag_tensor_shape_ptr(*curr)[ax];
                            if (i < 0) i += dim_size;
                            if (i < 0 || i >= dim_size)
                                throw nb::index_error("Index out of bounds");
                            mag_tensor_t *out = index_by_scalar_per_axis(*curr, ax, i);
                            curr = tensor_wrapper{out};
                            continue;
                        }
                    }
                    mag_tensor_t *out = nullptr;
                    throw_if_error(mag_gather(&out, *curr, ax, *idx_tw));
                    curr = tensor_wrapper{out};
                    ++ax;
                    continue;
                }
                if (nb::isinstance<nb::sequence>(idx)) {
                    auto seq = nb::cast<nb::sequence>(idx);
                    std::vector<int64_t> data;
                    data.reserve(nb::len(seq));
                    for (auto &&v : seq)
                        data.emplace_back(nb::cast<int64_t>(v));
                    auto sh = static_cast<int64_t>(data.size());
                    mag_tensor_t *idx_tensor = nullptr;
                    throw_if_error(mag_empty(&idx_tensor, get_ctx(), MAG_DTYPE_INT64, 1, &sh));
                    on_scope_exit defer_idx([idx_tensor] { mag_tensor_decref(idx_tensor); });
                    throw_if_error(mag_copy_raw_(idx_tensor, data.data(), data.size() * sizeof(int64_t)));
                    mag_tensor_t *out = nullptr;
                    throw_if_error(mag_gather(&out, *curr, ax, idx_tensor));
                    curr = tensor_wrapper{out};
                    ++ax;
                    continue;
                }
                throw nb::type_error("Invalid index component");
            }

            return curr;
        });
    }
}
