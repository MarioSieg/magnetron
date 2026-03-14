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

#include <algorithm>

#define bind_unary_pair(cls, name, doc) \
    cls \
        .def(#name, [](const tensor_wrapper &self) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            mag_tensor_t *out = nullptr; \
            mag_error_t err {}; \
            throw_if_error(mag_##name(&err, &out, *self), err); \
            return tensor_wrapper {out}; \
        }, doc) \
        .def(#name "_", [](tensor_wrapper &self) -> tensor_wrapper& { \
            std::lock_guard lock {get_global_mutex()}; \
            mag_tensor_t *out = nullptr; \
            mag_error_t err {}; \
            throw_if_error(mag_##name##_(&err, &out, *self), err); \
            if (self.p) mag_tensor_decref(self.p); \
            self.p = out; \
            return self; \
        }, "In-place version.", nb::rv_policy::reference)

#define bind_binary_full_named(cls, dunder_name, c_name, named_name, doc) \
    cls.def("__" #dunder_name "__", \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            mag_error_t err {}; \
            throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
            return tensor_wrapper{out}; \
        }, "rhs"_a, doc); \
    cls.def("__r" #dunder_name "__", \
        [](const tensor_wrapper &a, nb::handle lhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper l = normalize_rhs_to_tensor(a, lhs); \
            mag_tensor_t *out = nullptr; \
            mag_error_t err {}; \
            throw_if_error(mag_##c_name(&err, &out, *l, *a), err); \
            return tensor_wrapper{out}; \
        }, "lhs"_a, "Right-hand side of " #named_name " (reflected)."); \
    cls.def("__i" #dunder_name "__", \
        [](tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper& { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            mag_error_t err {}; \
            throw_if_error(mag_##c_name##_(&err, &out, *a, *b), err); \
            if (a.p) mag_tensor_decref(a.p); \
            a.p = out; \
            return a; \
        }, "rhs"_a, "In-place " #named_name ".", nb::rv_policy::reference); \
    cls.def(#named_name, \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            mag_error_t err {}; \
            throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
            return tensor_wrapper{out}; \
        }, "rhs"_a, doc); \
    cls.def(#named_name "_", \
        [](tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper& { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            mag_error_t err {}; \
            throw_if_error(mag_##c_name##_(&err, &out, *a, *b), err); \
            if (a.p) mag_tensor_decref(a.p); \
            a.p = out; \
            return a; \
        }, "rhs"_a, "In-place version.", nb::rv_policy::reference)

#define bind_compare(cls, dunder_name, c_name, named_name, doc) \
    cls.def("__" #dunder_name "__", \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            mag_error_t err {}; \
            throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
            return tensor_wrapper{out}; \
        }, "rhs"_a, doc); \
    cls.def(#named_name, \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            std::lock_guard lock {get_global_mutex()}; \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            mag_error_t err {}; \
            throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
            return tensor_wrapper{out}; \
        }, "rhs"_a, doc)

namespace mag::bindings {
    void init_tensor_class_operators(nb::class_<tensor_wrapper> &cls) {
        cls
        .def("fill_",
            [](tensor_wrapper &self, nb::handle value) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                mag_error_t err {};
                throw_if_error(mag_fill_(&err, *self, scalar_from_py(value)), err);
                return self;
            },
            "value"_a,
            "Fill the tensor with a scalar value."
        )
        .def("masked_fill_",
            [](tensor_wrapper &self, const tensor_wrapper &mask, nb::handle value) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                if (mag_tensor_type(*mask) != MAG_DTYPE_BOOLEAN)
                    throw nb::type_error("masked_fill_: mask must have dtype boolean");
                mag_error_t err {};
                throw_if_error(mag_masked_fill_(&err, *self, *mask, scalar_from_py(value)), err);
                return self;
            },
            "mask"_a, "value"_a,
            "Fill elements where mask is True with value."
        )
        .def("uniform_",
            [](tensor_wrapper &self, nb::handle low_h = nb::none(), nb::handle high_h = nb::none()) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                mag_scalar_t low  = low_h.is_none()  ? mag_scalar_from_f64(0.0) : scalar_from_py(low_h);
                mag_scalar_t high = high_h.is_none() ? mag_scalar_from_f64(1.0) : scalar_from_py(high_h);
                mag_error_t err {};
                throw_if_error(mag_uniform_(&err, *self, low, high), err);
                return self;
            },
            "low"_a = nb::none(),
            "high"_a = nb::none(),
            "Fill with samples from uniform(low, high). Default [0, 1)."
        )
        .def("normal_",
            [](tensor_wrapper &self, nb::handle mean_h = nb::float_(0.0), nb::handle std_h  = nb::float_(1.0)) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                mag_error_t err {};
                throw_if_error(mag_normal_(&err, *self, scalar_from_py(mean_h), scalar_from_py(std_h)), err);
                return self;
            },
            "mean"_a = 0.0,
            "std"_a  = 1.0,
            "Fill with samples from normal(mean, std)."
        )
        .def("bernoulli_",
            [](tensor_wrapper &self, nb::handle p_h = nb::float_(0.5)) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                mag_error_t err {};
                throw_if_error(mag_bernoulli_(&err, *self, scalar_from_py(p_h)), err);
                return self;
            },
            "p"_a = 0.5,
            "Fill with 0/1 from Bernoulli(p)."
        )
        .def("clone", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            mag_error_t err {};
            throw_if_error(mag_clone(&err, &out, *self), err);
            return tensor_wrapper{out};
        }, "Return a copy with the same data and dtype.")
        .def("cast", [](const tensor_wrapper &self, dtype_wrapper dt) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            mag_error_t err {};
            throw_if_error(mag_cast(&err, &out, *self, dt.v), err);
            return tensor_wrapper{out};
        }, "dtype"_a, "Return a copy with the given dtype.")
        .def("view", [](const tensor_wrapper &self, nb::args args) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            std::vector<int64_t> shape = parse_i64_dims(args, "view");
            validate_shape(shape);
            mag_tensor_t *out = nullptr;
            mag_error_t err {};
            throw_if_error(mag_view(&err, &out, *self, shape.data(), static_cast<int64_t>(shape.size())), err);
            return tensor_wrapper{out};
        }, "shape"_a, "View with new shape (same storage).")
        .def("view_slice", [](const tensor_wrapper &self, int64_t dim, int64_t start, int64_t len, int64_t step) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            mag_error_t err {};
            throw_if_error(mag_view_slice(&err, &out, *self, dim, start, len, step), err);
            return tensor_wrapper{out};
        }, "dim"_a, "start"_a, "len"_a, "step"_a, "View a slice along one dimension.")
        .def("reshape",
            [](const tensor_wrapper &self, nb::args dims_args) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                std::vector<int64_t> dims = parse_i64_dims(dims_args, "reshape");
                if (std::find(dims.begin(), dims.end(), 0) != dims.end())
                    throw nb::value_error("reshape: dimension 0 is not allowed");
                int neg_ones = static_cast<int>(std::count(dims.begin(), dims.end(), -1));
                if (neg_ones > 1) throw nb::value_error("reshape: only one -1 is allowed");
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_reshape(&err, &out, *self, dims.data(), static_cast<int64_t>(dims.size())), err);
                return tensor_wrapper{out};
            },
            "shape"_a,
            "Return a view with the given shape. Use -1 for one inferred dimension."
        )
        .def("transpose",
            [](const tensor_wrapper &self, int64_t dim0 = 0, int64_t dim1 = 1) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                if (dim0 == dim1)
                    throw nb::value_error("transpose: dim0 and dim1 must be different");
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_transpose(&err, &out, *self, dim0, dim1), err);
                return tensor_wrapper{out};
            },
            "dim0"_a = 0, "dim1"_a = 1,
            "Swap two dimensions."
        )
        .def_prop_ro("T", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            mag_error_t err {};
            throw_if_error(mag_T(&err, &out, *self), err);
            return tensor_wrapper{out};
        }, "Transpose of the tensor (dims 0 and 1 swapped).")
        .def("permute",
            [](const tensor_wrapper &self, nb::args dims_args) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                std::vector<int64_t> dims = parse_i64_dims(dims_args, "permute");
                int64_t r = mag_tensor_rank(*self);
                if (static_cast<int64_t>(dims.size()) != r)
                    throw nb::value_error("permute: number of dims must match tensor rank");

                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_permute(&err, &out, *self, dims.data(), static_cast<int64_t>(dims.size())), err);
                return tensor_wrapper{out};
            },
            "dims"_a,
            "Reorder dimensions by the given permutation."
        )
        .def("contiguous",
            [](const tensor_wrapper &self) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_contiguous(&err, &out, *self), err);
                return tensor_wrapper{out};
            },
            "Return a contiguous copy if needed; otherwise self."
        )
        .def("squeeze",
            [](const tensor_wrapper &self, nb::handle dim_h = nb::none()) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                if (dim_h.is_none()) {
                    throw_if_error(mag_squeeze_all(&err, &out, *self), err);
                } else {
                    auto dim = nb::cast<int64_t>(dim_h);
                    throw_if_error(mag_squeeze_dim(&err, &out, *self, dim), err);
                }
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(),
            "Remove size-1 dimensions (all or only dim)."
        )
        .def("unsqueeze",
            [](const tensor_wrapper &self, int64_t dim) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_unsqueeze(&err, &out, *self, dim), err);
                return tensor_wrapper{out};
            },
            "dim"_a,
            "Insert a size-1 dimension at dim."
        )
        .def("flatten",
            [](const tensor_wrapper &self, int64_t start_dim = 0, int64_t end_dim = -1) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_flatten(&err, &out, *self, start_dim, end_dim), err);
                return tensor_wrapper{out};
            },
            "start_dim"_a = 0, "end_dim"_a = -1,
            "Flatten dimensions from start_dim to end_dim (inclusive)."
        )
        .def("unflatten",
            [](const tensor_wrapper &self, int64_t dim, nb::handle sizes_h) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                std::vector<int64_t> sizes = parse_i64_list_handle(sizes_h, "unflatten(sizes)");
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_unflatten(&err, &out, *self, dim, sizes.data(), static_cast<int64_t>(sizes.size())), err);
                return tensor_wrapper{out};
            },
            "dim"_a, "sizes"_a,
            "Expand dim into multiple dimensions with sizes."
        )
        .def("narrow",
            [](const tensor_wrapper &self, int64_t dim, int64_t start, int64_t length) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_narrow(&err, &out, *self, dim, start, length), err);
                return tensor_wrapper{out};
            },
            "dim"_a, "start"_a, "length"_a,
            "View a slice of length along dim starting at start."
        )
        .def("movedim",
            [](const tensor_wrapper &self, int64_t src, int64_t dst) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_movedim(&err, &out, *self, src, dst), err);
                return tensor_wrapper{out};
            },
            "src"_a, "dst"_a,
            "Move dimension src to position dst."
        )
        .def("select",
            [](const tensor_wrapper &self, int64_t dim, int64_t index) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_select(&err, &out, *self, dim, index), err);
                return tensor_wrapper{out};
            },
            "dim"_a, "index"_a,
            "Select a slice at index along dim (reduces rank by 1)."
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
                if (size == 0) return {};
                int64_t n_chunks = (size + split_size - 1)/split_size;
                std::vector<mag_tensor_t*> outs(static_cast<size_t>(n_chunks), nullptr);
                mag_error_t err {};
                throw_if_error(mag_split(&err, outs.data(), n_chunks, *self, split_size, dim), err);
                PyObject *t = PyTuple_New(n_chunks);
                if (!t) throw nb::python_error();
                for (int64_t i=0; i < n_chunks; ++i) {
                    tensor_wrapper tw{outs[static_cast<size_t>(i)]};
                    nb::object obj = nb::cast(tw);
                    PyTuple_SET_ITEM(t, i, obj.release().ptr());
                }
                return nb::steal<nb::tuple>(t);
            },
            "split_size"_a, "dim"_a = 0,
            "Split into chunks of split_size along dim. Returns tuple of tensors."
        )
        .def("mean",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_mean(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false,
            "Mean over dim(s). None = all dims."
        )
        .def("min",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_min(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false,
            "Minimum over dim(s). None = all dims."
        )
        .def("max",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_max(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false,
            "Maximum over dim(s). None = all dims."
        )
        .def("argmin",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_argmin(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false,
            "Indices of minimum over dim(s)."
        )
        .def("argmax",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_argmax(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false,
            "Indices of maximum over dim(s)."
        )
        .def("sum",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_sum(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false,
            "Sum over dim(s). None = all dims."
        )
        .def("prod",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_prod(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false,
            "Product over dim(s). None = all dims."
        )
        .def("all",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_all(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false,
            "Logical AND over dim(s). Boolean tensor."
        )
        .def("any",
            [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto ax = parse_reduction_axes(dim);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_any(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
                return tensor_wrapper{out};
            },
            "dim"_a = nb::none(), "keepdim"_a = false,
            "Logical OR over dim(s). Boolean tensor."
        )
        .def("topk",
            [](const tensor_wrapper &self, int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true) -> nb::tuple {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *values = nullptr;
                mag_tensor_t *indices = nullptr;
                mag_error_t err {};
                throw_if_error(mag_topk(&err, &values, &indices, *self, k, dim, largest, sorted), err);
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
            "k"_a, "dim"_a = -1, "largest"_a = true, "sorted"_a = true,
            "Return (values, indices) of the k largest or smallest elements along dim."
        )
        .def("tril",
            [](const tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_tril(&err, &out, *self, diagonal), err);
                return tensor_wrapper{out};
            },
            "diagonal"_a = 0,
            "Lower triangular part; elements above diagonal set to 0."
        )
        .def("tril_",
            [](tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_tril_(&err, &out, *self, diagonal), err);
                return tensor_wrapper {out};
            },
            "diagonal"_a = 0,
            "In-place lower triangular."
        )
        .def("triu",
            [](const tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_triu(&err, &out, *self, diagonal), err);
                return tensor_wrapper{out};
            },
            "diagonal"_a = 0,
            "Upper triangular part; elements below diagonal set to 0."
        )
        .def("triu_",
            [](tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_triu_(&err, &out, *self, diagonal), err);
                return tensor_wrapper {out};
            },
            "diagonal"_a = 0,
            "In-place upper triangular."
        )
        .def("multinomial",
            [](const tensor_wrapper &self, int64_t num_samples = 1, bool replacement = false) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                if (num_samples <= 0)
                    throw nb::value_error("multinomial: num_samples must be > 0");
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_multinomial(&err, &out, *self, num_samples, replacement), err);
                return tensor_wrapper{out};
            },
            "num_samples"_a = 1, "replacement"_a = false,
            "Sample indices from probabilities (last dim). Returns shape (..., num_samples)."
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
                mag_error_t err {};
                for (size_t i = 0; i < n; ++i) {
                    mag_tensor_t *ci = nullptr;
                    throw_if_error(mag_contiguous(&err, &ci, *tensors[i]), err);
                    contig.emplace_back(ci);
                    ptrs.emplace_back(*contig.back());
                }
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_cat(&err, &out, ptrs.data(), ptrs.size(), dim), err);
                return tensor_wrapper{out};
            },
            "tensors"_a, "dim"_a = 0,
            "Concatenate tensors along the given dimension."
        );


        // Unary operators
        bind_unary_pair(cls, abs, "Element-wise absolute value.");
        bind_unary_pair(cls, sgn, "Element-wise sign (-1, 0, or 1).");
        bind_unary_pair(cls, neg, "Element-wise negation.");
        bind_unary_pair(cls, log, "Natural logarithm.");
        bind_unary_pair(cls, log10, "Base-10 logarithm.");
        bind_unary_pair(cls, log1p, "log(1 + x).");
        bind_unary_pair(cls, log2, "Base-2 logarithm.");
        bind_unary_pair(cls, sqr, "Element-wise square.");
        bind_unary_pair(cls, rcp, "Reciprocal 1/x.");
        bind_unary_pair(cls, sqrt, "Element-wise square root.");
        bind_unary_pair(cls, rsqrt, "Reciprocal square root 1/sqrt(x).");
        bind_unary_pair(cls, sin, "Element-wise sine.");
        bind_unary_pair(cls, cos, "Element-wise cosine.");
        bind_unary_pair(cls, tan, "Element-wise tangent.");
        bind_unary_pair(cls, sinh, "Element-wise hyperbolic sine.");
        bind_unary_pair(cls, cosh, "Element-wise hyperbolic cosine.");
        bind_unary_pair(cls, tanh, "Element-wise hyperbolic tangent.");
        bind_unary_pair(cls, asin, "Element-wise arc sine.");
        bind_unary_pair(cls, acos, "Element-wise arc cosine.");
        bind_unary_pair(cls, atan, "Element-wise arc tangent.");
        bind_unary_pair(cls, asinh, "Element-wise inverse hyperbolic sine.");
        bind_unary_pair(cls, acosh, "Element-wise inverse hyperbolic cosine.");
        bind_unary_pair(cls, atanh, "Element-wise inverse hyperbolic tangent.");
        bind_unary_pair(cls, step, "Heaviside step (0 if x < 0 else 1).");
        bind_unary_pair(cls, erf, "Error function.");
        bind_unary_pair(cls, erfc, "Complementary error function.");
        bind_unary_pair(cls, exp, "Element-wise exp(x).");
        bind_unary_pair(cls, expm1, "exp(x) - 1.");
        bind_unary_pair(cls, floor, "Round down to integer.");
        bind_unary_pair(cls, ceil, "Round up to integer.");
        bind_unary_pair(cls, round, "Round to nearest integer.");
        bind_unary_pair(cls, trunc, "Truncate toward zero.");

        // Softmax has params and required a specialized binding
        cls.def("softmax",
            [](const tensor_wrapper &self, [[maybe_unused]] int64_t dim) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_softmax(&err, &out, *self), err); // TODO: respect dim
                return tensor_wrapper{out};
            },
            "dim"_a,
            "Softmax over dim (normalizes to sum to 1)."
        );
        cls.def("softmax_",
            [](tensor_wrapper &self, [[maybe_unused]] int64_t dim) -> tensor_wrapper& {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_softmax_(&err, &out, *self), err); // TODO: respect dim
                if (self.p) mag_tensor_decref(self.p);
                self.p = out;
                return self;
            },
            "dim"_a, "In-place softmax.", nb::rv_policy::reference
        );

        bind_unary_pair(cls, softmax_dv, "Softmax derivative (for autodiff).");
        bind_unary_pair(cls, sigmoid, "Sigmoid 1/(1+exp(-x)).");
        bind_unary_pair(cls, sigmoid_dv, "Sigmoid derivative.");
        bind_unary_pair(cls, hard_sigmoid, "Hard sigmoid approximation.");
        bind_unary_pair(cls, silu, "SiLU (Swish) x*sigmoid(x).");
        bind_unary_pair(cls, silu_dv, "SiLU derivative.");
        bind_unary_pair(cls, tanh_dv, "Tanh derivative.");
        bind_unary_pair(cls, relu, "ReLU max(0, x).");
        bind_unary_pair(cls, relu_dv, "ReLU derivative.");
        bind_unary_pair(cls, gelu, "GELU activation.");
        bind_unary_pair(cls, gelu_approx, "GELU approximate form.");
        bind_unary_pair(cls, gelu_dv, "GELU derivative.");
        cls
        .def("__neg__", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            mag_error_t err {};
            throw_if_error(mag_neg(&err, &out, *self), err);
            return tensor_wrapper{out};
        }, "Element-wise negation (unary -).")
        .def("__pos__", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            return self;
        }, "Unary + (returns self).")
        .def("__abs__", [](const tensor_wrapper &self) -> tensor_wrapper {
            std::lock_guard lock {get_global_mutex()};
            mag_tensor_t *out = nullptr;
            mag_error_t err {};
            throw_if_error(mag_abs(&err, &out, *self), err);
            return tensor_wrapper{out};
        }, "Element-wise absolute value.");

        // Binary operators
        bind_binary_full_named(cls, add, add, add, "Element-wise addition.");
        bind_binary_full_named(cls, sub, sub, sub, "Element-wise subtraction.");
        bind_binary_full_named(cls, mul, mul, mul, "Element-wise multiplication.");
        bind_binary_full_named(cls, mod, mod, mod, "Element-wise modulo.");
        bind_binary_full_named(cls, pow, pow, pow, "Element-wise exponentiation.");
        bind_binary_full_named(cls, truediv, div, truediv, "Element-wise true division.");
        bind_binary_full_named(cls, floordiv, floordiv, floordiv, "Element-wise floor division.");
        bind_binary_full_named(cls, and, and, logical_and, "Element-wise logical AND.");
        bind_binary_full_named(cls, or,  or,  logical_or, "Element-wise logical OR.");
        bind_binary_full_named(cls, xor, xor, logical_xor, "Element-wise logical XOR.");
        bind_binary_full_named(cls, lshift, shl, lshift, "Element-wise left shift.");
        bind_binary_full_named(cls, rshift, shr, rshift, "Element-wise right shift.");
        bind_compare(cls, lt, lt, lt, "Element-wise less than. Returns boolean tensor.");
        bind_compare(cls, le, le, le, "Element-wise less or equal.");
        bind_compare(cls, gt, gt, gt, "Element-wise greater than.");
        bind_compare(cls, ge, ge, ge, "Element-wise greater or equal.");
        bind_compare(cls, eq, eq, eq, "Element-wise equality.");
        bind_compare(cls, ne, ne, ne, "Element-wise not equal.");

        // Matmul
        cls.def("__matmul__",
            [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                tensor_wrapper b = normalize_rhs_to_tensor(a, rhs);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_matmul(&err, &out, *a, *b), err);
                return tensor_wrapper{out};
            },
            "rhs"_a,
            "Matrix multiplication (or batch matmul). Supports @ operator."
        );
    }
}
