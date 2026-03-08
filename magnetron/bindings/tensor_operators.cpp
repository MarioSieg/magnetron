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

#define bind_unary_pair(cls, name) \
    cls \
        .def(#name, [](const tensor_wrapper &self) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##name(&out, *self)); \
            return tensor_wrapper {out}; \
        }) \
        .def(#name "_", [](tensor_wrapper &self) -> tensor_wrapper& { \
            std::lock_guard lock {get_global_mutex()}; \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##name##_(&out, *self)); \
            if (self.p) mag_tensor_decref(self.p); \
            self.p = out; \
            return self; \
        }, nb::rv_policy::reference)

#define bind_binary_full_named(cls, dunder_name, c_name, named_name) \
    cls.def("__" #dunder_name "__", \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *a, *b)); \
            return tensor_wrapper{out}; \
        }, "rhs"_a); \
    cls.def("__r" #dunder_name "__", \
        [](const tensor_wrapper &a, nb::handle lhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper l = normalize_rhs_to_tensor(a, lhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *l, *a)); \
            return tensor_wrapper{out}; \
        }, "lhs"_a); \
    cls.def("__i" #dunder_name "__", \
        [](tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper& { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name##_(&out, *a, *b)); \
            if (a.p) mag_tensor_decref(a.p); \
            a.p = out; \
            return a; \
        }, "rhs"_a, nb::rv_policy::reference); \
    cls.def(#named_name, \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *a, *b)); \
            return tensor_wrapper{out}; \
        }, "rhs"_a); \
    cls.def(#named_name "_", \
        [](tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper& { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name##_(&out, *a, *b)); \
            if (a.p) mag_tensor_decref(a.p); \
            a.p = out; \
            return a; \
        }, "rhs"_a, nb::rv_policy::reference)

#define bind_compare(cls, dunder_name, c_name, named_name) \
    cls.def("__" #dunder_name "__", \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *a, *b)); \
            return tensor_wrapper{out}; \
        }, "rhs"_a); \
    cls.def(#named_name, \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *a, *b)); \
            return tensor_wrapper{out}; \
        }, "rhs"_a)

namespace mag::bindings {
    void init_tensor_class_operators(nb::class_<tensor_wrapper> &cls) {
        cls
        .def("fill_",
            [](tensor_wrapper &self, nb::handle value) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                throw_if_error(mag_fill_(*self, scalar_from_py(value)));
                return self;
            },
            "value"_a
        )
        .def("masked_fill_",
            [](tensor_wrapper &self, const tensor_wrapper &mask, nb::handle value) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                if (mag_tensor_type(*mask) != MAG_DTYPE_BOOLEAN)
                    throw nb::type_error("masked_fill_: mask must have dtype boolean");
                throw_if_error(mag_masked_fill_(*self, *mask, scalar_from_py(value)));
                return self;
            },
            "mask"_a, "value"_a
        )
        .def("uniform_",
            [](tensor_wrapper &self, nb::handle low_h = nb::none(), nb::handle high_h = nb::none()) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                mag_scalar_t low  = low_h.is_none()  ? mag_scalar_from_f64(0.0) : scalar_from_py(low_h);
                mag_scalar_t high = high_h.is_none() ? mag_scalar_from_f64(1.0) : scalar_from_py(high_h);
                throw_if_error(mag_uniform_(*self, low, high));
                return self;
            },
            "low"_a = nb::none(),
            "high"_a = nb::none()
        )
        .def("normal_",
            [](tensor_wrapper &self, nb::handle mean_h = nb::float_(0.0), nb::handle std_h  = nb::float_(1.0)) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                throw_if_error(mag_normal_(*self, scalar_from_py(mean_h), scalar_from_py(std_h)));
                return self;
            },
            "mean"_a = 0.0,
            "std"_a  = 1.0
        )
        .def("bernoulli_",
            [](tensor_wrapper &self, nb::handle p_h = nb::float_(0.5)) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                throw_if_error(mag_bernoulli_(*self, scalar_from_py(p_h)));
                return self;
            },
            "p"_a = 0.5
        )
        .def("clone", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_clone(&out, *self));
            return tensor_wrapper{out};
        })
        .def("cast", [](const tensor_wrapper &self, dtype_wrapper dt) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_cast(&out, *self, dt.v));
            return tensor_wrapper{out};
        }, "dtype"_a)
        .def("view", [](const tensor_wrapper &self, nb::args args) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            std::vector<int64_t> shape = parse_i64_dims(args, "view");
            validate_shape(shape);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_view(&out, *self, shape.data(), (int64_t)shape.size()));
            return tensor_wrapper{out};
        }, "shape"_a)
        .def("view_slice", [](const tensor_wrapper &self, int64_t dim, int64_t start, int64_t len, int64_t step) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_view_slice(&out, *self, dim, start, len, step));
            return tensor_wrapper{out};
        }, "dim"_a, "start"_a, "len"_a, "step"_a)
        .def("reshape",
            [](const tensor_wrapper &self, nb::args dims_args) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                std::vector<int64_t> dims = parse_i64_dims(dims_args, "reshape");
                int neg_ones = 0;
                for (int64_t d : dims) {
                    if (d == -1) ++neg_ones;
                    else if (d == 0) throw nb::value_error("reshape: dimension 0 is not allowed");
                }
                if (neg_ones > 1) throw nb::value_error("reshape: only one -1 is allowed");
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_reshape(&out, *self, dims.data(), static_cast<int64_t>(dims.size())));
                return tensor_wrapper{out};
            }
        )
        .def("transpose",
            [](const tensor_wrapper &self, int64_t dim0 = 0, int64_t dim1 = 1) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                if (dim0 == dim1)
                    throw nb::value_error("transpose: dim0 and dim1 must be different");
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_transpose(&out, *self, dim0, dim1));
                return tensor_wrapper{out};
            },
            "dim0"_a = 0, "dim1"_a = 1
        )
        .def_prop_ro("T", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_T(&out, *self));
            return tensor_wrapper{out};
        })
        .def("permute",
            [](const tensor_wrapper &self, nb::args dims_args) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                std::vector<int64_t> dims = parse_i64_dims(dims_args, "permute");
                int64_t r = mag_tensor_rank(*self);
                if ((int64_t) dims.size() != r)
                    throw nb::value_error("permute: number of dims must match tensor rank");

                mag_tensor_t *out = nullptr;
                throw_if_error(mag_permute(&out, *self, dims.data(), (int64_t) dims.size()));
                return tensor_wrapper{out};
            }
        )
        .def("contiguous",
            [](const tensor_wrapper &self) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_contiguous(&out, *self));
                return tensor_wrapper{out};
            }
        )
        .def("squeeze",
            [](const tensor_wrapper &self, nb::handle dim_h = nb::none()) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                if (dim_h.is_none()) {
                    throw_if_error(mag_squeeze_all(&out, *self));
                } else {
                    auto dim = nb::cast<int64_t>(dim_h);
                    throw_if_error(mag_squeeze_dim(&out, *self, dim));
                }
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none()
        )
        .def("unsqueeze",
            [](const tensor_wrapper &self, int64_t dim) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_unsqueeze(&out, *self, dim));
                return tensor_wrapper{out};
            },
            "dim"_a
        )
        .def("flatten",
            [](const tensor_wrapper &self, int64_t start_dim = 0, int64_t end_dim = -1) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_flatten(&out, *self, start_dim, end_dim));
                return tensor_wrapper{out};
            },
            "start_dim"_a = 0, "end_dim"_a = -1
        )
        .def("unflatten",
            [](const tensor_wrapper &self, int64_t dim, nb::handle sizes_h) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                std::vector<int64_t> sizes = parse_i64_list_handle(sizes_h, "unflatten(sizes)");
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_unflatten(&out, *self, dim, sizes.data(), static_cast<int64_t>(sizes.size())));
                return tensor_wrapper{out};
            },
            "dim"_a, "sizes"_a
        )
        .def("narrow",
            [](const tensor_wrapper &self, int64_t dim, int64_t start, int64_t length) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_narrow(&out, *self, dim, start, length));
                return tensor_wrapper{out};
            },
            "dim"_a, "start"_a, "length"_a
        )
        .def("movedim",
            [](const tensor_wrapper &self, int64_t src, int64_t dst) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_movedim(&out, *self, src, dst));
                return tensor_wrapper{out};
            },
            "src"_a, "dst"_a
        )
        .def("select",
            [](const tensor_wrapper &self, int64_t dim, int64_t index) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_select(&out, *self, dim, index));
                return tensor_wrapper{out};
            },
            "dim"_a, "index"_a
        )
        .def("split",
            [](const tensor_wrapper &self, int64_t split_size, int64_t dim = 0) -> nb::tuple {
                std::lock_guard lock {get_global_mutex()};
                if (split_size <= 0) throw nb::value_error("split: split_size must be > 0");
                int64_t rank = mag_tensor_rank(*self);
                if (rank == 0) throw std::runtime_error("split is not defined for 0-dim tensors");
                if (dim < 0) dim += rank;
                if (dim < 0 || dim >= rank) throw nb::index_error("split: dim out of range");
                int64_t size = mag_tensor_shape_ptr(*self)[dim];
                if (size == 0) return nb::tuple(); // empty tuple
                int64_t n_chunks = (size + split_size - 1) / split_size;
                std::vector<mag_tensor_t*> outs(static_cast<size_t>(n_chunks), nullptr);
                throw_if_error(mag_split(outs.data(), n_chunks, *self, split_size, dim));
                PyObject *t = PyTuple_New(n_chunks);
                if (!t) throw nb::python_error();
                for (int64_t i=0; i < n_chunks; ++i) {
                    tensor_wrapper tw{outs[static_cast<size_t>(i)]};
                    nb::object obj = nb::cast(tw);
                    PyTuple_SET_ITEM(t, i, obj.release().ptr());
                }
                return nb::steal<nb::tuple>(t);
            },
            "split_size"_a, "dim"_a = 0
        )
        .def("mean",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_mean(&out, *self, ax.ptr, ax.rank, keepdim));
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false
        )
        .def("min",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_min(&out, *self, ax.ptr, ax.rank, keepdim));
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false
        )
        .def("max",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_max(&out, *self, ax.ptr, ax.rank, keepdim));
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false
        )
        .def("argmin",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_argmin(&out, *self, ax.ptr, ax.rank, keepdim));
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false
        )
        .def("argmax",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_argmax(&out, *self, ax.ptr, ax.rank, keepdim));
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false
        )
        .def("sum",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_sum(&out, *self, ax.ptr, ax.rank, keepdim));
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false
        )
        .def("prod",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_prod(&out, *self, ax.ptr, ax.rank, keepdim));
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false
        )
        .def("all",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_all(&out, *self, ax.ptr, ax.rank, keepdim));
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false
        )
        .def("any",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_any(&out, *self, ax.ptr, ax.rank, keepdim));
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false
        )
        .def("topk",
            [](const tensor_wrapper &self, int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true) -> nb::tuple {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *values = nullptr;
                mag_tensor_t *indices = nullptr;
                throw_if_error(mag_topk(&values, &indices, *self, k, dim, largest, sorted));
                tensor_wrapper v_tw{values};
                tensor_wrapper i_tw{indices};
                PyObject *t = PyTuple_New(2);
                if (!t) throw nb::python_error();
                nb::object v = nb::cast(v_tw);
                nb::object i = nb::cast(i_tw);
                PyTuple_SET_ITEM(t, 0, v.release().ptr());
                PyTuple_SET_ITEM(t, 1, i.release().ptr());
                return nb::steal<nb::tuple>(t);
            },
            "k"_a, "dim"_a = -1, "largest"_a = true, "sorted"_a = true
        )
        .def("tril",
            [](const tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_tril(&out, *self, diagonal));
                return tensor_wrapper{out};
            },
            "diagonal"_a = 0
        )
        .def("tril_",
            [](tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_tril_(&out, *self, diagonal));
                return tensor_wrapper {out};
            },
            "diagonal"_a = 0
        )
        .def("triu",
            [](const tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_triu(&out, *self, diagonal));
                return tensor_wrapper{out};
            },
            "diagonal"_a = 0
        )
        .def("triu_",
            [](tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_triu_(&out, *self, diagonal));
                return tensor_wrapper {out};
            },
            "diagonal"_a = 0
        )
        .def("multinomial",
            [](const tensor_wrapper &self, int64_t num_samples = 1, bool replacement = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                if (num_samples <= 0)
                    throw nb::value_error("multinomial: num_samples must be > 0");
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_multinomial(&out, *self, num_samples, replacement));
                return tensor_wrapper{out};
            },
            "num_samples"_a = 1, "replacement"_a = false
        );

        cls.attr("cat") = nb::cpp_function(
            [](nb::handle tensors_h, int64_t dim = 0) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                if (!nb::isinstance<nb::sequence>(tensors_h))
                    throw nb::type_error("cat: 'tensors' must be a sequence of Tensor");
                auto seq = nb::cast<nb::sequence>(tensors_h);
                size_t n = nb::len(seq);
                if (n == 0)
                    throw nb::value_error("cat: at least one tensor is required");
                std::vector<tensor_wrapper> tensors;
                tensors.reserve(n);
                for (nb::handle h : seq) {
                    auto tw = nb::cast<tensor_wrapper>(h);
                    if (!tw) throw nb::value_error("cat: encountered a null Tensor");
                    tensors.emplace_back(tw);
                }
                int64_t rank = mag_tensor_rank(*tensors[0]);
                if (rank <= 0)
                    throw nb::value_error("cat: tensors must have rank > 0");
                if (dim < 0) dim += rank;
                if (dim < 0 || dim >= rank)
                    throw nb::index_error("cat: dim out of range");
                std::vector<tensor_wrapper> contig;
                contig.reserve(n);
                std::vector<mag_tensor_t*> ptrs;
                ptrs.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    mag_tensor_t *ci = nullptr;
                    throw_if_error(mag_contiguous(&ci, *tensors[i]));
                    contig.emplace_back(tensor_wrapper{ci}); // owns ci
                    ptrs.emplace_back(*contig.back());
                }
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_cat(&out, ptrs.data(), ptrs.size(), dim));
                return tensor_wrapper{out};
            },
            "tensors"_a, "dim"_a = 0
        );


        // Unary operators
        bind_unary_pair(cls, abs);
        bind_unary_pair(cls, sgn);
        bind_unary_pair(cls, neg);
        bind_unary_pair(cls, log);
        bind_unary_pair(cls, log10);
        bind_unary_pair(cls, log1p);
        bind_unary_pair(cls, log2);
        bind_unary_pair(cls, sqr);
        bind_unary_pair(cls, rcp);
        bind_unary_pair(cls, sqrt);
        bind_unary_pair(cls, rsqrt);
        bind_unary_pair(cls, sin);
        bind_unary_pair(cls, cos);
        bind_unary_pair(cls, tan);
        bind_unary_pair(cls, sinh);
        bind_unary_pair(cls, cosh);
        bind_unary_pair(cls, tanh);
        bind_unary_pair(cls, asin);
        bind_unary_pair(cls, acos);
        bind_unary_pair(cls, atan);
        bind_unary_pair(cls, asinh);
        bind_unary_pair(cls, acosh);
        bind_unary_pair(cls, atanh);
        bind_unary_pair(cls, step);
        bind_unary_pair(cls, erf);
        bind_unary_pair(cls, erfc);
        bind_unary_pair(cls, exp);
        bind_unary_pair(cls, expm1);
        bind_unary_pair(cls, floor);
        bind_unary_pair(cls, ceil);
        bind_unary_pair(cls, round);
        bind_unary_pair(cls, trunc);

        // Softmax has params and required a specialized binding
        cls.def("softmax",
            [](const tensor_wrapper &self, [[maybe_unused]] int64_t dim) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_softmax(&out, *self)); // TODO: respect dim
                return tensor_wrapper{out};
            },
            "dim"_a
        );
        cls.def("softmax_",
            [](tensor_wrapper &self, [[maybe_unused]] int64_t dim) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_softmax_(&out, *self)); // TODO: respect dim
                if (self.p) mag_tensor_decref(self.p);
                self.p = out;
                return self;
            },
            "dim"_a, nb::rv_policy::reference
        );

        bind_unary_pair(cls, softmax_dv);
        bind_unary_pair(cls, sigmoid);
        bind_unary_pair(cls, sigmoid_dv);
        bind_unary_pair(cls, hard_sigmoid);
        bind_unary_pair(cls, silu);
        bind_unary_pair(cls, silu_dv);
        bind_unary_pair(cls, tanh_dv);
        bind_unary_pair(cls, relu);
        bind_unary_pair(cls, relu_dv);
        bind_unary_pair(cls, gelu);
        bind_unary_pair(cls, gelu_approx);
        bind_unary_pair(cls, gelu_dv);
        cls
        .def("__neg__", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_neg(&out, *self));
            return tensor_wrapper{out};
        })
        .def("__pos__", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            return self;
        })
        .def("__abs__", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_abs(&out, *self));
            return tensor_wrapper{out};
        });

        // Binary operators
        bind_binary_full_named(cls, add, add, add);
        bind_binary_full_named(cls, sub, sub, sub);
        bind_binary_full_named(cls, mul, mul, mul);
        bind_binary_full_named(cls, mod, mod, mod);
        bind_binary_full_named(cls, truediv, div, truediv);
        bind_binary_full_named(cls, floordiv, floordiv, floordiv);
        bind_binary_full_named(cls, and, and, logical_and);
        bind_binary_full_named(cls, or,  or,  logical_or);
        bind_binary_full_named(cls, xor, xor, logical_xor);
        bind_binary_full_named(cls, lshift, shl, lshift);
        bind_binary_full_named(cls, rshift, shr, rshift);
        bind_compare(cls, lt, lt, lt);
        bind_compare(cls, le, le, le);
        bind_compare(cls, gt, gt, gt);
        bind_compare(cls, ge, ge, ge);
        bind_compare(cls, eq, eq, eq);
        bind_compare(cls, ne, ne, ne);

        // Matmul
        cls.def("__matmul__",
            [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                tensor_wrapper b = normalize_rhs_to_tensor(a, rhs);
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_matmul(&out, *a, *b));
                return tensor_wrapper{out};
            },
            "rhs"_a
        );
    }
}
