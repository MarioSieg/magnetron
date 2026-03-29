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

#include <string_view>
#include <sstream>
#include <charconv>
#include <algorithm>

namespace mag::bindings {
    // Helper to convert a C array of int64_t to a Python tuple of ints.
    // Nanobind doesn't have a built-in way to do this and would require allocating a list and then conerting it to a tuple, which sucks.
    nb::tuple tuple_from_i64_span(const int64_t *p, Py_ssize_t n) {
        PyObject *t = PyTuple_New(n);
        if (!t) throw nb::python_error {};
        for (Py_ssize_t i=0; i < n; ++i) {
            PyObject *v = PyLong_FromLongLong(static_cast<long long int>(p[i]));
            if (!v) {
                Py_DECREF(t);
                throw nb::python_error {};
            }
            PyTuple_SET_ITEM(t, i, v);
        }
        return nb::steal<nb::tuple>(t);
    }

    tensor_wrapper tensor_from_py_scalar(nb::handle obj, mag_dtype_t dt, mag_device_id_t device) {
        mag_context_t *ctx = get_ctx();
        mag_tensor_t *out = nullptr;
        if (nb::isinstance<nb::bool_>(obj)) { // Prefer bool first: in Python, bool is a subclass of int.
            bool b = nb::cast<bool>(obj);
            mag_scalar_t s = mag_scalar_from_u64(!!b);
            mag_error_t err {};
            throw_if_error(mag_scalar(&err, &out, ctx, dt, s, device), err);
            return tensor_wrapper{out};
        }
        if (nb::isinstance<nb::int_>(obj)) { // Python int is arbitrary precision; nb::cast<int64_t> will throw if too big.
            auto v = nb::cast<int64_t>(obj);
            mag_scalar_t s = mag_scalar_from_i64(v);
            mag_error_t err {};
            throw_if_error(mag_scalar(&err, &out, ctx, dt, s, device), err);
            return tensor_wrapper{out};
        }
        if (nb::isinstance<nb::float_>(obj)) {
            auto v = nb::cast<double>(obj);
            mag_scalar_t s = mag_scalar_from_f64(v);
            mag_error_t err {};
            throw_if_error(mag_scalar(&err, &out, ctx, dt, s, device), err);
            return tensor_wrapper{out};
        }
        throw nb::type_error("rhs must be Tensor, int, float, or bool");
    }

    tensor_wrapper normalize_rhs_to_tensor(const tensor_wrapper &lhs, nb::handle rhs) {
        if (nb::isinstance<tensor_wrapper>(rhs)) return nb::cast<tensor_wrapper>(rhs);
        return tensor_from_py_scalar(rhs, mag_tensor_type(*lhs), mag_tensor_device_id(*lhs));
    }

    std::vector<int64_t> parse_shape_from_args(const nb::args &args) {
        std::vector<int64_t> shape {};
        if (args.size() == 1 && nb::isinstance<nb::sequence>(args[0])) {
            auto seq = nb::cast<nb::sequence>(args[0]);
            shape.reserve(nb::len(seq));
            for (auto &&h : seq)
                shape.emplace_back(nb::cast<int64_t>(h));
        } else {
            shape.reserve(args.size());
            for (auto &&h : args)
                shape.emplace_back(nb::cast<int64_t>(h));
        }
        return shape;
    }

    void validate_shape(const std::vector<int64_t> &shape) {
        if (shape.size() > MAG_MAX_DIMS) {
            auto msg = "Invalid number of dimensions, must be <= " + std::to_string(MAG_MAX_DIMS);
            throw nb::value_error(msg.c_str());
        }
        if (std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d <= 0; }))
            throw nb::value_error("Invalid dimension size (must be > 0)");
    }

    static void flatten_i64_handle(nb::handle h, std::vector<int64_t> &out) {
        if (nb::isinstance<nb::sequence>(h) && !nb::isinstance<nb::str>(h) && !nb::isinstance<tensor_wrapper>(h)) {
            auto seq = nb::cast<nb::sequence>(h);
            for (auto &&child : seq)
                flatten_i64_handle(child, out);
        } else {
            out.emplace_back(nb::cast<int64_t>(h));
        }
    }

    std::vector<int64_t> parse_i64_dims(const nb::args &args, const char *what) {
        std::vector<int64_t> out;
        if (args.size() == 1 && nb::isinstance<nb::sequence>(args[0])) {
            flatten_i64_handle(args[0], out);
        } else {
            for (auto &&h : args)
                flatten_i64_handle(h, out);
        }
        if (out.empty())
            throw nb::value_error((std::string(what) + ": expected at least one dimension").c_str());
        return out;
    }

    std::vector<int64_t> parse_i64_list_handle(nb::handle h, const char *what) {
        auto seq = nb::cast<nb::sequence>(h);
        std::vector<int64_t> out {};
        out.reserve(nb::len(seq));
        for (auto &&x : seq)
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
            for (auto &&h : seq)
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

    nb::object py_scalar_from_mag_scalar(const mag_scalar_t &scalar) {
        switch (scalar.type) {
            case MAG_SCALAR_TYPE_I64: return nb::cast<int64_t>(mag_scalar_as_i64(scalar));
            case MAG_SCALAR_TYPE_U64: return nb::cast<uint64_t>(mag_scalar_as_u64(scalar));
            case MAG_SCALAR_TYPE_F64: return nb::cast<double>(mag_scalar_as_f64(scalar));
            default: throw nb::type_error("Unsupported scalar type");
        }
    }

    dtype_wrapper deduce_dtype_from_py_scalar(nb::handle h) {
        if (nb::isinstance<nb::bool_>(h)) return dtype_wrapper{MAG_DTYPE_BOOLEAN};
        if (nb::isinstance<nb::int_>(h)) return dtype_wrapper{MAG_DTYPE_INT64};
        if (nb::isinstance<nb::float_>(h)) return dtype_wrapper{MAG_DTYPE_FLOAT32};
        throw nb::type_error("Cannot deduce dtype from this object");
    }

    std::string format_error_msg(const mag_error_t &err) {
        std::stringstream ss {};
        const char *name = mag_status_get_name(err.code);
        ss << "MagnetronError[" << (name ? name : "UNKNOWN") << "]: ";
        ss << (err.message[0] ? err.message : "unknown error");
        if (err.file) {
            ss << " (" << err.file << ":" << err.line;
            if (err.func)
                ss << " in " << err.func;
            ss << ")";
        }
        return ss.str();
    }

    std::optional<mag_device_id_t> parse_device_id_str(std::string &&str) {
        std::string_view input{str};
        if (input.empty()) return std::nullopt;
        auto sep = input.find(':');
        std::string_view name = input.substr(0, sep);
        if (name.empty() || name.size() >= 32) return std::nullopt;
        const auto iequals = [](std::string_view a, const char* b) -> bool {
            size_t i=0;
            for (; i < a.size() && b[i] != '\0'; ++i) {
                const auto ac = static_cast<unsigned char>(a[i]);
                const auto bc = static_cast<unsigned char>(b[i]);
                if (std::tolower(ac) != std::tolower(bc)) return false;
            }
            return i == a.size() && b[i] == '\0';
        };
        mag_backend_type_t type = MAG_BACKEND_TYPE__COUNT;
        static_assert(static_cast<std::underlying_type_t<mag_backend_type_t>>(MAG_BACKEND_TYPE_CPU) == 0);
        for (std::underlying_type_t<mag_backend_type_t> type_idx = MAG_BACKEND_TYPE_CPU; type_idx < MAG_BACKEND_TYPE__COUNT; ++type_idx) {
            if (iequals(name, mag_backend_type_to_str(static_cast<mag_backend_type_t>(type_idx)))) {
                type = static_cast<mag_backend_type_t>(type_idx);
                break;
            }
        }
        if (type == MAG_BACKEND_TYPE__COUNT) return std::nullopt;
        uint32_t ordinal = 0;
        if (sep != std::string_view::npos) {
            std::string_view ord_str = input.substr(sep + 1);
            if (ord_str.empty()) return std::nullopt;
            const char *first = ord_str.data();
            const char *last  = ord_str.data() + ord_str.size();
            auto [ptr, ec] = std::from_chars(first, last, ordinal);
            if (ec != std::errc{} || ptr != last) return std::nullopt;
        }
        return mag_device_id_t{
            .type = type,
            .device_ordinal = ordinal,
        };
    }
}
