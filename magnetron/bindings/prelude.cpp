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
    // Helper to convert a C array of int64_t to a Python tuple of ints.
    // Nanobind doesn't have a built-in way to do this and would require allocating a list and then conerting it to a tuple, which sucks.
    nb::tuple tuple_from_i64_span(const int64_t *p, Py_ssize_t n) {
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

    void throw_if_ctx_error(mag_context_t *ctx) {
        if (!mag_ctx_has_error(ctx)) return;
        mag_status_t code = mag_ctx_get_error_code(ctx);
        const char *name = mag_status_get_name(code);
        throw std::runtime_error {name ? name : "Unknown Magnetron Error"};
    }

    void throw_if_error(mag_status_t st) {
        if (st == MAG_STATUS_OK) return;
        const char *name = mag_status_get_name(st);
        throw std::runtime_error {name ? name : "Unknown Magnetron Error"};
    }

    tensor_wrapper tensor_from_py_scalar(nb::handle obj, mag_dtype_t dt) {
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

    tensor_wrapper normalize_rhs_to_tensor(const tensor_wrapper &lhs, nb::handle rhs) {
        if (nb::isinstance<tensor_wrapper>(rhs)) return nb::cast<tensor_wrapper>(rhs);
        return tensor_from_py_scalar(rhs, mag_tensor_type(*lhs));
    }

    std::vector<int64_t> parse_shape_from_args(const nb::args &args) {
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

    void validate_shape(const std::vector<int64_t> &shape) {
        if (shape.size() > MAG_MAX_DIMS) {
            auto msg = "Invalid number of dimensions, must be <= " + std::to_string(MAG_MAX_DIMS);
            throw nb::value_error(msg.c_str());
        }
        for (auto d : shape) {
            if (d <= 0) throw nb::value_error("Invalid dimension size (must be > 0)");
        }
    }

    std::vector<int64_t> parse_i64_dims(nb::args args, const char *what) {
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

    std::vector<int64_t> parse_i64_list_handle(nb::handle h, const char *what) {
        nb::sequence seq = nb::cast<nb::sequence>(h);
        std::vector<int64_t> out;
        out.reserve(nb::len(seq));
        for (nb::handle x : seq)
            out.emplace_back(nb::cast<int64_t>(x));
        if (out.empty())
            throw nb::value_error((std::string(what) + ": empty sequence").c_str());
        return out;
    }

    reduction_axes parse_reduction_axes(nb::handle dim_h) {
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

    mag_scalar_t scalar_from_py(nb::handle h) {
        if (nb::isinstance<nb::bool_>(h)) return mag_scalar_from_u64(nb::cast<bool>(h));
        if (nb::isinstance<nb::int_>(h)) return mag_scalar_from_i64(nb::cast<int64_t>(h));
        if (nb::isinstance<nb::float_>(h)) return mag_scalar_from_f64(nb::cast<double>(h));
        throw nb::type_error("Expected scalar (bool|int|float)");
    }

    dtype_wrapper deduce_dtype_from_py_scalar(nb::handle h) {
        if (nb::isinstance<nb::bool_>(h)) return dtype_wrapper{MAG_DTYPE_BOOLEAN};
        if (nb::isinstance<nb::int_>(h)) return dtype_wrapper{MAG_DTYPE_INT64};
        if (nb::isinstance<nb::float_>(h)) return dtype_wrapper{MAG_DTYPE_FLOAT32};
        throw nb::type_error("Cannot deduce dtype from this object");
    }
}
