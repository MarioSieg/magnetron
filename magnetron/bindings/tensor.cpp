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

// Helper to convert a C array of int64_t to a Python tuple of ints.
// Nanobind doesn't have a built-in way to do this and would require allocating a list and then conerting it to a tuple, which sucks.
[[nodiscard]] static nb::tuple tuple_from_i64_span(const int64_t *p, Py_ssize_t n) {
    PyObject *t = PyTuple_New(n);
    if (!t) throw nb::python_error {};
    for (Py_ssize_t i=0; i < n; ++i) {
        PyObject *v = PyLong_FromLongLong(static_cast<long long>(p[i]));
        if (!v) {
            Py_DECREF(t);
            throw nb::python_error {};
        }
        PyTuple_SET_ITEM(t, i, v);
    }
    return nb::steal<nb::tuple>(t);
}

static void throw_if_ctx_error(mag_context_t *ctx) {
    if (!mag_ctx_has_error(ctx)) return;
    mag_status_t code = mag_ctx_get_error_code(ctx);
    const char *name = mag_status_get_name(code);
    throw std::runtime_error {name ? name : "Unknown Magnetron Error"};
}

static void throw_if_error(mag_status_t st) {
    if (st == MAG_STATUS_OK) return;
    const char *name = mag_status_get_name(st);
    throw std::runtime_error {name ? name : "Unknown Magnetron Error"};
}

struct tensor_wrapper final {
    mag_tensor_t *p = nullptr;

    constexpr tensor_wrapper() noexcept = default;
    explicit tensor_wrapper(mag_tensor_t *ptr) noexcept : p{ptr} {}
    tensor_wrapper(const tensor_wrapper &other) noexcept : p{other.p} { if (p) mag_tensor_incref(p); }
    constexpr tensor_wrapper(tensor_wrapper &&other) noexcept : p{other.p} { other.p = nullptr; }
    tensor_wrapper &operator=(const tensor_wrapper &other) noexcept {
        if (this != &other) {
            if (other.p) mag_tensor_incref(other.p);
            if (p) mag_tensor_decref(p);
            p = other.p;
        }
        return *this;
    }
    tensor_wrapper &operator=(tensor_wrapper &&other) noexcept {
        if (this != &other) {
            if (p) mag_tensor_decref(p);
            p = other.p;
            other.p = nullptr;
        }
        return *this;
    }
    ~tensor_wrapper() {
        if (p) mag_tensor_decref(p);
    }
    explicit constexpr operator bool() const noexcept { return p != nullptr; }
    constexpr mag_tensor_t *operator * () const noexcept { return p; }

};

// Create a 1-element tensor (a scalar) with dtype = dt and value from a Python scalar.
[[nodiscard]] static tensor_wrapper tensor_from_py_scalar(nb::handle obj, mag_dtype_t dt) {
    mag_context_t *ctx = get_ctx();
    mag_tensor_t *out = nullptr;
    if (nb::isinstance<nb::bool_>(obj)) { // Prefer bool first: in Python, bool is a subclass of int.
        bool b = nb::cast<bool>(obj);
        mag_scalar_t s = mag_scalar_from_u64(!!b);
        throw_if_error(mag_scalar(&out, ctx, dt, s));
        return tensor_wrapper{out};
    }
    if (nb::isinstance<nb::int_>(obj)) { // Python int is arbitrary precision; nb::cast<int64_t> will throw if too big.
        auto v = nb::cast<int64_t>(obj);
        mag_scalar_t s = mag_scalar_from_i64(v);
        throw_if_error(mag_scalar(&out, ctx, dt, s));
        throw_if_ctx_error(ctx);
        return tensor_wrapper{out};
    }
    if (nb::isinstance<nb::float_>(obj)) {
        auto v = nb::cast<double>(obj);
        mag_scalar_t s = mag_scalar_from_f64(v);
        throw_if_error(mag_scalar(&out, ctx, dt, s));
        throw_if_ctx_error(ctx);
        return tensor_wrapper{out};
    }
    throw nb::type_error("rhs must be Tensor, int, float, or bool");
}

[[nodiscard]] static tensor_wrapper normalize_rhs_to_tensor(const tensor_wrapper &lhs, nb::handle rhs) {
    if (nb::isinstance<tensor_wrapper>(rhs)) return nb::cast<tensor_wrapper>(rhs);
    return tensor_from_py_scalar(rhs, mag_tensor_type(*lhs));
}

[[nodiscard]] static std::vector<int64_t> parse_shape_from_args(const nb::args &args) {
    std::vector<int64_t> shape {};
    if (args.size() == 1 && nb::isinstance<nb::sequence>(args[0])) {
        auto seq = nb::cast<nb::sequence>(args[0]);
        shape.reserve(nb::len(seq));
        for (nb::handle h : seq)
            shape.emplace_back(nb::cast<int64_t>(h));
    } else {
        shape.reserve(args.size());
        for (nb::handle h : args)
            shape.emplace_back(nb::cast<int64_t>(h));
    }
    return shape;
}

static void validate_shape(const std::vector<int64_t> &shape) {
    if (shape.size() > MAG_MAX_DIMS) {
        auto msg = "Invalid number of dimensions, must be <= " + std::to_string(MAG_MAX_DIMS);
        throw nb::value_error(msg.c_str());
    }
    for (auto d : shape) {
        if (d <= 0) throw nb::value_error("Invalid dimension size (must be > 0)");
    }
}

static std::vector<int64_t> parse_i64_dims(nb::args args, const char *what) {
    std::vector<int64_t> out;

    if (args.size() == 1 && nb::isinstance<nb::sequence>(args[0])) {
        nb::sequence seq = nb::cast<nb::sequence>(args[0]);
        out.reserve(nb::len(seq));
        for (nb::handle h : seq)
            out.emplace_back(nb::cast<int64_t>(h));
    } else {
        out.reserve(args.size());
        for (nb::handle h : args)
            out.emplace_back(nb::cast<int64_t>(h));
    }

    if (out.empty())
        throw nb::value_error((std::string(what) + ": expected at least one dimension").c_str());

    return out;
}

static std::vector<int64_t> parse_i64_list_handle(nb::handle h, const char *what) {
    nb::sequence seq = nb::cast<nb::sequence>(h);
    std::vector<int64_t> out;
    out.reserve(nb::len(seq));
    for (nb::handle x : seq)
        out.emplace_back(nb::cast<int64_t>(x));
    if (out.empty())
        throw nb::value_error((std::string(what) + ": empty sequence").c_str());
    return out;
}

struct reduction_axes {
    std::vector<int64_t> storage;
    const int64_t *ptr = nullptr;
    int64_t rank = 0;
};
static reduction_axes parse_reduction_axes(nb::handle dim_h) {
    reduction_axes ax{};
    if (dim_h.is_none()) {
        ax.ptr = nullptr;
        ax.rank = 0;
        return ax;
    }
    if (nb::isinstance<nb::int_>(dim_h)) {
        ax.storage = { nb::cast<int64_t>(dim_h) };
        ax.ptr = ax.storage.data();
        ax.rank = 1;
        return ax;
    }
    if (nb::isinstance<nb::sequence>(dim_h)) {
        auto seq = nb::cast<nb::sequence>(dim_h);
        ax.storage.reserve(nb::len(seq));
        for (nb::handle h : seq)
            ax.storage.emplace_back(nb::cast<int64_t>(h));
        ax.ptr = ax.storage.data();
        ax.rank = static_cast<int64_t>(ax.storage.size());
        return ax;
    }
    throw nb::type_error("dim must be None, int, or a sequence of ints");
}

[[nodiscard]] static dtype_wrapper kw_dtype_or(nb::kwargs &kwargs, dtype_wrapper def = dtype_wrapper{MAG_DTYPE_FLOAT32}) {
    if (kwargs.contains("dtype"))
        return nb::cast<dtype_wrapper>(kwargs["dtype"]);
    return def;
}

[[nodiscard]] static bool kw_requires_grad_or(nb::kwargs &kwargs, bool def = false) {
    if (kwargs.contains("requires_grad"))
        return nb::cast<bool>(kwargs["requires_grad"]);
    return def;
}

static void maybe_set_requires_grad(mag_context_t *ctx, mag_tensor_t *t, bool requires_grad) {
    if (!requires_grad) return;
    mag_tensor_set_requires_grad(t, true);
    throw_if_ctx_error(ctx);
}

static mag_scalar_t scalar_from_py(nb::handle h) {
    if (nb::isinstance<nb::bool_>(h)) {
        bool b = nb::cast<bool>(h);
        return mag_scalar_from_u64(b ? 1u : 0u);
    }
    if (nb::isinstance<nb::int_>(h)) {
        auto v = nb::cast<int64_t>(h);
        return mag_scalar_from_i64(v);
    }
    if (nb::isinstance<nb::float_>(h)) {
        auto v = nb::cast<double>(h);
        return mag_scalar_from_f64(v);
    }
    throw nb::type_error("Expected scalar (bool|int|float)");
}

[[nodiscard]] static dtype_wrapper deduce_dtype_from_py_scalar(nb::handle h) {
    if (nb::isinstance<nb::bool_>(h)) return dtype_wrapper{MAG_DTYPE_BOOLEAN};
    if (nb::isinstance<nb::int_>(h)) return dtype_wrapper{MAG_DTYPE_INT64};
    if (nb::isinstance<nb::float_>(h)) return dtype_wrapper{MAG_DTYPE_FLOAT32};
    throw nb::type_error("Cannot deduce dtype from this object");
}

#define bind_unary_pair(cls, name) \
    cls \
        .def(#name, [](const tensor_wrapper &self) -> tensor_wrapper { \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##name(&out, *self)); \
            return tensor_wrapper {out}; \
        }) \
        .def(#name "_", [](tensor_wrapper &self) -> tensor_wrapper& { \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##name##_(&out, *self)); \
            if (self.p) mag_tensor_decref(self.p); \
            self.p = out; \
            return self; \
        }, nb::rv_policy::reference)

#define bind_binary_full_named(cls, dunder_name, c_name, named_name) \
    cls.def("__" #dunder_name "__", \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *a, *b)); \
            return tensor_wrapper{out}; \
        }, "rhs"_a); \
    cls.def("__r" #dunder_name "__", \
        [](const tensor_wrapper &a, nb::handle lhs) -> tensor_wrapper { \
            tensor_wrapper l = normalize_rhs_to_tensor(a, lhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *l, *a)); \
            return tensor_wrapper{out}; \
        }, "lhs"_a); \
    cls.def("__i" #dunder_name "__", \
        [](tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper& { \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name##_(&out, *a, *b)); \
            if (a.p) mag_tensor_decref(a.p); \
            a.p = out; \
            return a; \
        }, "rhs"_a, nb::rv_policy::reference); \
    cls.def(#named_name, \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *a, *b)); \
            return tensor_wrapper{out}; \
        }, "rhs"_a); \
    cls.def(#named_name "_", \
        [](tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper& { \
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
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *a, *b)); \
            return tensor_wrapper{out}; \
        }, "rhs"_a); \
    cls.def(#named_name, \
        [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
            tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
            mag_tensor_t *out = nullptr; \
            throw_if_error(mag_##c_name(&out, *a, *b)); \
            return tensor_wrapper{out}; \
        }, "rhs"_a)


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

static void init_tensor_special_methods(nb::class_<tensor_wrapper> &cls) {
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

static void init_tensor_class_factories(nb::class_<tensor_wrapper> &cls) {
    cls.attr("empty") = nb::cpp_function(
        [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
            dtype_wrapper dt {MAG_DTYPE_FLOAT32};
            bool requires_grad = false;
            if (kwargs.contains("dtype"))
                dt = nb::cast<dtype_wrapper>(kwargs["dtype"]);
            if (kwargs.contains("requires_grad"))
                requires_grad = nb::cast<bool>(kwargs["requires_grad"]);
            std::vector<int64_t> shape {};
            if (args.size() == 1 && nb::isinstance<nb::sequence>(args[0])) {
                nb::sequence seq = nb::cast<nb::sequence>(args[0]);
                shape.reserve(nb::len(seq));
                for (auto &&h : seq)
                    shape.emplace_back(nb::cast<int64_t>(h));
            } else {
                shape.reserve(args.size());
                for (auto &&h : args)
                    shape.emplace_back(nb::cast<int64_t>(h));
            }
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            if (shape.empty()) throw_if_error(mag_empty_scalar(&out, ctx, dt.v));
            else throw_if_error(mag_empty(&out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data()));
            throw_if_ctx_error(ctx);
            if (requires_grad) {
                mag_tensor_set_requires_grad(out, true);
                throw_if_ctx_error(ctx);
            }
            return tensor_wrapper {out};
        }
    );
    cls.attr("empty_like") = nb::cpp_function(
        [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_empty_like(&out, *like));
            mag_context_t *ctx = get_ctx();
            throw_if_ctx_error(ctx);
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("scalar") = nb::cpp_function(
        [](nb::handle value, nb::kwargs kwargs) -> tensor_wrapper {
            dtype_wrapper dt = kw_dtype_or(kwargs, deduce_dtype_from_py_scalar(value));
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            mag_scalar_t s = scalar_from_py(value);
            throw_if_error(mag_scalar(&out, ctx, dt.v, s));
            throw_if_ctx_error(ctx);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("full") = nb::cpp_function(
        [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
            if (!kwargs.contains("fill_value"))
                throw nb::type_error("full(...): missing keyword argument 'fill_value'");
            nb::handle fill_value = kwargs["fill_value"];
            dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            std::vector<int64_t> shape = parse_shape_from_args(args);
            validate_shape(shape);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            mag_scalar_t s = scalar_from_py(fill_value);
            throw_if_error(mag_full(&out, ctx, dt.v, (int64_t) shape.size(), shape.data(), s));
            throw_if_ctx_error(ctx);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("full_like") = nb::cpp_function(
        [](const tensor_wrapper &like, nb::handle fill_value, nb::kwargs kwargs) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            mag_scalar_t s = scalar_from_py(fill_value);
            throw_if_error(mag_full_like(&out, *like, s));
            mag_context_t *ctx = get_ctx();
            throw_if_ctx_error(ctx);
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("zeros") = nb::cpp_function(
        [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
            dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            std::vector<int64_t> shape = parse_shape_from_args(args);
            validate_shape(shape);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_zeros(&out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data()));
            throw_if_ctx_error(ctx);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("zeros_like") = nb::cpp_function(
        [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_zeros_like(&out, *like));
            mag_context_t *ctx = get_ctx();
            throw_if_ctx_error(ctx);
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("ones") = nb::cpp_function(
        [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
            dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            std::vector<int64_t> shape = parse_shape_from_args(args);
            validate_shape(shape);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_ones(&out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data()));
            throw_if_ctx_error(ctx);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("ones_like") = nb::cpp_function(
        [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_ones_like(&out, *like));
            mag_context_t *ctx = get_ctx();
            throw_if_ctx_error(ctx);
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("uniform") = nb::cpp_function(
        [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
            dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            mag_scalar_t low  = kwargs.contains("low")  ? scalar_from_py(kwargs["low"])  : mag_scalar_from_f64(0.0);
            mag_scalar_t high = kwargs.contains("high") ? scalar_from_py(kwargs["high"]) : mag_scalar_from_f64(1.0);
            std::vector<int64_t> shape = parse_shape_from_args(args);
            validate_shape(shape);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_uniform(&out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data(), low, high));
            throw_if_ctx_error(ctx);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("uniform_like") = nb::cpp_function(
        [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
            mag_scalar_t low  = kwargs.contains("low")  ? scalar_from_py(kwargs["low"])  : mag_scalar_from_f64(0.0);
            mag_scalar_t high = kwargs.contains("high") ? scalar_from_py(kwargs["high"]) : mag_scalar_from_f64(1.0);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_uniform_like(&out, *like, low, high));
            mag_context_t *ctx = get_ctx();
            throw_if_ctx_error(ctx);
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("normal") = nb::cpp_function(
        [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
            dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            mag_scalar_t mean = kwargs.contains("mean") ? scalar_from_py(kwargs["mean"]) : mag_scalar_from_f64(0.0);
            mag_scalar_t std  = kwargs.contains("std")  ? scalar_from_py(kwargs["std"])  : mag_scalar_from_f64(1.0);
            std::vector<int64_t> shape = parse_shape_from_args(args);
            validate_shape(shape);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_normal(&out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data(), mean, std));
            throw_if_ctx_error(ctx);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("normal_like") = nb::cpp_function(
        [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
            mag_scalar_t mean = kwargs.contains("mean") ? scalar_from_py(kwargs["mean"]) : mag_scalar_from_f64(0.0);
            mag_scalar_t std  = kwargs.contains("std")  ? scalar_from_py(kwargs["std"])  : mag_scalar_from_f64(1.0);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_normal_like(&out, *like, mean, std));
            mag_context_t *ctx = get_ctx();
            throw_if_ctx_error(ctx);
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("bernoulli") = nb::cpp_function(
        [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
            mag_scalar_t p = kwargs.contains("p") ? scalar_from_py(kwargs["p"]) : mag_scalar_from_f64(0.5);
            std::vector<int64_t> shape = parse_shape_from_args(args);
            validate_shape(shape);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_bernoulli(&out, ctx, static_cast<int64_t>(shape.size()), shape.data(), p));
            throw_if_ctx_error(ctx);
            return tensor_wrapper{out};
        }
    );
    cls.attr("bernoulli_like") = nb::cpp_function(
        [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
            mag_scalar_t p = kwargs.contains("p") ? scalar_from_py(kwargs["p"]) : mag_scalar_from_f64(0.5);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_bernoulli_like(&out, *like, p));
            mag_context_t *ctx = get_ctx();
            throw_if_ctx_error(ctx);
            return tensor_wrapper{out};
        }
    );
    cls.attr("arange") = nb::cpp_function(
        [](nb::kwargs kwargs) -> tensor_wrapper {
            if (!kwargs.contains("stop") && !kwargs.contains("end"))
                throw nb::type_error("arange(...): missing 'stop' (or 'end')");
            nb::handle stop_h = kwargs.contains("stop") ? kwargs["stop"] : kwargs["end"];
            nb::handle start_h = kwargs.contains("start") ? kwargs["start"] : nb::handle{};
            nb::handle step_h = kwargs.contains("step")  ? kwargs["step"] : nb::handle{};
            bool stop_is_float = nb::isinstance<nb::float_>(stop_h);
            nb::object start_obj = kwargs.contains("start") ? nb::borrow<nb::object>(start_h) : stop_is_float ? nb::object{nb::float_{0.0}} : nb::object{nb::int_{1.0}};
            nb::object step_obj  = kwargs.contains("step") ? nb::borrow<nb::object>(step_h) : stop_is_float ? nb::object{nb::float_{1.0}} : nb::object{nb::int_{1.0}};
            dtype_wrapper dt = kwargs.contains("dtype") ? nb::cast<dtype_wrapper>(kwargs["dtype"]) : deduce_dtype_from_py_scalar(start_obj);
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            mag_scalar_t start = scalar_from_py(start_obj);
            mag_scalar_t stop = scalar_from_py(stop_h);
            mag_scalar_t step = scalar_from_py(step_obj);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_arange(&out, ctx, dt.v, start, stop, step));
            throw_if_ctx_error(ctx);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("rand_perm") = nb::cpp_function(
        [](int64_t n, nb::kwargs kwargs) -> tensor_wrapper {
            dtype_wrapper dt = kw_dtype_or(kwargs, dtype_wrapper{MAG_DTYPE_INT64});
            bool requires_grad = kw_requires_grad_or(kwargs, false);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_rand_perm(&out, ctx, dt.v, n));
            throw_if_ctx_error(ctx);
            maybe_set_requires_grad(ctx, out, requires_grad);
            return tensor_wrapper{out};
        }
    );
    cls.attr("load_image") = nb::cpp_function(
        [](const std::string &path, nb::kwargs kwargs) -> tensor_wrapper {
            std::string channels = "RGB";
            uint32_t rw = 0, rh = 0;
            if (kwargs.contains("channels"))
                channels = nb::cast<std::string>(kwargs["channels"]);
            if (kwargs.contains("resize_to")) {
                nb::handle rt = kwargs["resize_to"];
                auto t = nb::cast<nb::tuple>(rt);
                if (t.size() != 2) throw nb::value_error("resize_to must be a tuple of (w,h)");
                rw = static_cast<uint32_t>(nb::cast<int64_t>(t[0]));
                rh = static_cast<uint32_t>(nb::cast<int64_t>(t[1]));
            }
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_load_image(&out, ctx, path.c_str(), channels.c_str(), rw, rh));
            throw_if_ctx_error(ctx);
            return tensor_wrapper{out};
        }
    );
    cls.attr("as_strided") = nb::cpp_function(
        [](const tensor_wrapper &base, nb::handle shape_h, nb::handle strides_h, nb::kwargs kwargs) -> tensor_wrapper {
            auto shape_seq = nb::cast<nb::sequence>(shape_h);
            auto strides_seq = nb::cast<nb::sequence>(strides_h);
            if (nb::len(shape_seq) != nb::len(strides_seq))
                throw nb::value_error("shape and strides must have same length");
            std::vector<int64_t> shape;
            std::vector<int64_t> strides;
            shape.reserve(nb::len(shape_seq));
            strides.reserve(nb::len(strides_seq));
            for (nb::handle h : shape_seq) shape.emplace_back(nb::cast<int64_t>(h));
            for (nb::handle h : strides_seq) strides.emplace_back(nb::cast<int64_t>(h));
            validate_shape(shape);
            int64_t offset = 0;
            if (kwargs.contains("offset"))
                offset = nb::cast<int64_t>(kwargs["offset"]);
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_as_strided(&out, ctx, *base, static_cast<int64_t>(shape.size()), shape.data(), strides.data(), offset));
            throw_if_ctx_error(ctx);
            return tensor_wrapper{out};
        }
    );
}

static void init_tensor_class_operators(nb::class_<tensor_wrapper> &cls) {
    cls
    .def("fill_",
        [](tensor_wrapper &self, nb::handle value) -> tensor_wrapper& {
            throw_if_error(mag_fill_(*self, scalar_from_py(value)));
            return self;
        },
        "value"_a
    )
    .def("masked_fill_",
        [](tensor_wrapper &self, const tensor_wrapper &mask, nb::handle value) -> tensor_wrapper& {
            if (mag_tensor_type(*mask) != MAG_DTYPE_BOOLEAN)
                throw nb::type_error("masked_fill_: mask must have dtype boolean");
            throw_if_error(mag_masked_fill_(*self, *mask, scalar_from_py(value)));
            return self;
        },
        "mask"_a, "value"_a
    )
    .def("uniform_",
        [](tensor_wrapper &self, nb::handle low_h = nb::none(), nb::handle high_h = nb::none()) -> tensor_wrapper& {
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
            throw_if_error(mag_normal_(*self, scalar_from_py(mean_h), scalar_from_py(std_h)));
            return self;
        },
        "mean"_a = 0.0,
        "std"_a  = 1.0
    )
    .def("bernoulli_",
        [](tensor_wrapper &self, nb::handle p_h = nb::float_(0.5)) -> tensor_wrapper& {
            throw_if_error(mag_bernoulli_(*self, scalar_from_py(p_h)));
            return self;
        },
        "p"_a = 0.5
    )
    .def("clone", [](const tensor_wrapper &self) -> tensor_wrapper {
        mag_tensor_t *out = nullptr;
        throw_if_error(mag_clone(&out, *self));
        return tensor_wrapper{out};
    })
    .def("cast", [](const tensor_wrapper &self, dtype_wrapper dt) -> tensor_wrapper {
        mag_tensor_t *out = nullptr;
        throw_if_error(mag_cast(&out, *self, dt.v));
        return tensor_wrapper{out};
    }, "dtype"_a)
    .def("view", [](const tensor_wrapper &self, nb::handle shape_h) -> tensor_wrapper {
        auto shape_seq = nb::cast<nb::sequence>(shape_h);
        std::vector<int64_t> shape {};
        shape.reserve(nb::len(shape_seq));
        for (nb::handle h : shape_seq)
            shape.emplace_back(nb::cast<int64_t>(h));
        validate_shape(shape);
        mag_tensor_t *out = nullptr;
        throw_if_error(mag_view(&out, *self, shape.data(), static_cast<int64_t>(shape.size())));
        return tensor_wrapper{out};
    }, "shape"_a)
    .def("view_slice", [](const tensor_wrapper &self, int64_t dim, int64_t start, int64_t len, int64_t step) -> tensor_wrapper {
        mag_tensor_t *out = nullptr;
        throw_if_error(mag_view_slice(&out, *self, dim, start, len, step));
        return tensor_wrapper{out};
    }, "dim"_a, "start"_a, "len"_a, "step"_a)
    .def("reshape",
        [](const tensor_wrapper &self, nb::args dims_args) -> tensor_wrapper {
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
            if (dim0 == dim1)
                throw nb::value_error("transpose: dim0 and dim1 must be different");
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_transpose(&out, *self, dim0, dim1));
            return tensor_wrapper{out};
        },
        "dim0"_a = 0, "dim1"_a = 1
    )
    .def("permute",
        [](const tensor_wrapper &self, nb::args dims_args) -> tensor_wrapper {
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
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_contiguous(&out, *self));
            return tensor_wrapper{out};
        }
    )
    .def("squeeze",
        [](const tensor_wrapper &self, nb::handle dim_h = nb::none()) -> tensor_wrapper {
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
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_unsqueeze(&out, *self, dim));
            return tensor_wrapper{out};
        },
        "dim"_a
    )
    .def("flatten",
        [](const tensor_wrapper &self, int64_t start_dim = 0, int64_t end_dim = -1) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_flatten(&out, *self, start_dim, end_dim));
            return tensor_wrapper{out};
        },
        "start_dim"_a = 0, "end_dim"_a = -1
    )
    .def("unflatten",
        [](const tensor_wrapper &self, int64_t dim, nb::handle sizes_h) -> tensor_wrapper {
            std::vector<int64_t> sizes = parse_i64_list_handle(sizes_h, "unflatten(sizes)");
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_unflatten(&out, *self, dim, sizes.data(), (int64_t) sizes.size()));
            return tensor_wrapper{out};
        },
        "dim"_a, "sizes"_a
    )
    .def("narrow",
        [](const tensor_wrapper &self, int64_t dim, int64_t start, int64_t length) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_narrow(&out, *self, dim, start, length));
            return tensor_wrapper{out};
        },
        "dim"_a, "start"_a, "length"_a
    )
    .def("movedim",
        [](const tensor_wrapper &self, int64_t src, int64_t dst) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_movedim(&out, *self, src, dst));
            return tensor_wrapper{out};
        },
        "src"_a, "dst"_a
    )
    .def("select",
        [](const tensor_wrapper &self, int64_t dim, int64_t index) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_select(&out, *self, dim, index));
            return tensor_wrapper{out};
        },
        "dim"_a, "index"_a
    )
    .def("split",
        [](const tensor_wrapper &self, int64_t split_size, int64_t dim = 0) -> nb::tuple {
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
                nb::object obj = nb::cast(tensor_wrapper{outs[static_cast<size_t>(i)]});
                PyTuple_SET_ITEM(t, i, obj.release().ptr());
            }
            return nb::steal<nb::tuple>(t);
        },
        "split_size"_a, "dim"_a = 0
    )
    .def("mean",
        [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
            auto ax = parse_reduction_axes(dim);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_mean(&out, *self, ax.ptr, ax.rank, keepdim));
            return tensor_wrapper{out};
        },
        "dim"_a = nb::none(), "keepdim"_a = false
    )
    .def("min",
        [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
            auto ax = parse_reduction_axes(dim);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_min(&out, *self, ax.ptr, ax.rank, keepdim));
            return tensor_wrapper{out};
        },
        "dim"_a = nb::none(), "keepdim"_a = false
    )
    .def("max",
        [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
            auto ax = parse_reduction_axes(dim);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_max(&out, *self, ax.ptr, ax.rank, keepdim));
            return tensor_wrapper{out};
        },
        "dim"_a = nb::none(), "keepdim"_a = false
    )
    .def("argmin",
        [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
            auto ax = parse_reduction_axes(dim);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_argmin(&out, *self, ax.ptr, ax.rank, keepdim));
            return tensor_wrapper{out};
        },
        "dim"_a = nb::none(), "keepdim"_a = false
    )
    .def("argmax",
        [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
            auto ax = parse_reduction_axes(dim);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_argmax(&out, *self, ax.ptr, ax.rank, keepdim));
            return tensor_wrapper{out};
        },
        "dim"_a = nb::none(), "keepdim"_a = false
    )
    .def("sum",
        [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
            auto ax = parse_reduction_axes(dim);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_sum(&out, *self, ax.ptr, ax.rank, keepdim));
            return tensor_wrapper{out};
        },
        "dim"_a = nb::none(), "keepdim"_a = false
    )
    .def("prod",
        [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
            auto ax = parse_reduction_axes(dim);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_prod(&out, *self, ax.ptr, ax.rank, keepdim));
            return tensor_wrapper{out};
        },
        "dim"_a = nb::none(), "keepdim"_a = false
    )
    .def("all",
        [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
            auto ax = parse_reduction_axes(dim);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_all(&out, *self, ax.ptr, ax.rank, keepdim));
            return tensor_wrapper{out};
        },
        "dim"_a = nb::none(), "keepdim"_a = false
    )
    .def("any",
        [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
            auto ax = parse_reduction_axes(dim);
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_any(&out, *self, ax.ptr, ax.rank, keepdim));
            return tensor_wrapper{out};
        },
        "dim"_a = nb::none(), "keepdim"_a = false
    )
    .def("topk",
        [](const tensor_wrapper &self, int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true) -> nb::tuple {
            mag_tensor_t *values = nullptr;
            mag_tensor_t *indices = nullptr;
            throw_if_error(mag_topk(&values, &indices, *self, k, dim, largest, sorted));
            PyObject *t = PyTuple_New(2);
            if (!t) throw nb::python_error();
            nb::object v = nb::cast(tensor_wrapper{values});
            nb::object i = nb::cast(tensor_wrapper{indices});
            PyTuple_SET_ITEM(t, 0, v.release().ptr());
            PyTuple_SET_ITEM(t, 1, i.release().ptr());
            return nb::steal<nb::tuple>(t);
        },
        "k"_a, "dim"_a = -1, "largest"_a = true, "sorted"_a = true
    )
    .def("tril",
        [](const tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_tril(&out, *self, diagonal));
            return tensor_wrapper{out};
        },
        "diagonal"_a = 0
    )
    .def("tril_",
        [](tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_tril_(&out, *self, diagonal));
            return tensor_wrapper {out};
        },
        "diagonal"_a = 0
    )
    .def("triu",
        [](const tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_triu(&out, *self, diagonal));
            return tensor_wrapper{out};
        },
        "diagonal"_a = 0
    )
    .def("triu_",
        [](tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_triu_(&out, *self, diagonal));
            return tensor_wrapper {out};
        },
        "diagonal"_a = 0
    )
    .def("multinomial",
        [](const tensor_wrapper &self, int64_t num_samples = 1, bool replacement = false) -> tensor_wrapper {
            if (num_samples <= 0)
                throw nb::value_error("multinomial: num_samples must be > 0");
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_multinomial(&out, *self, num_samples, replacement));
            return tensor_wrapper{out};
        },
        "num_samples"_a = 1, "replacement"_a = false
    );
    cls.attr("cat") = nb::cpp_function([](nb::handle tensors_h, int64_t dim = 0) -> tensor_wrapper {
            if (!nb::isinstance<nb::sequence>(tensors_h))
                throw nb::type_error("cat: 'tensors' must be a sequence of Tensor");
            auto seq = nb::cast<nb::sequence>(tensors_h);
            size_t n = nb::len(seq);
            if (n == 0)
                throw nb::value_error("cat: at least one tensor is required");
            std::vector<tensor_wrapper> keep {};
            keep.reserve(n);
            std::vector<mag_tensor_t *> ptrs {};
            ptrs.reserve(n);
            for (nb::handle h : seq) {
                auto tw = nb::cast<tensor_wrapper>(h);
                if (!tw) throw nb::value_error("cat: encountered a null Tensor");
                keep.emplace_back(tw);
                ptrs.emplace_back(*keep.back());
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
    bind_unary_pair(cls, softmax);
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
}

void mag_init_bindings_tensor(nb::module_ &m) {
    auto cls = nb::class_<tensor_wrapper>{m, "Tensor"};
    init_tensor_class_base(cls);
    init_tensor_special_methods(cls);
    init_tensor_class_factories(cls);
    init_tensor_class_operators(cls);
}
