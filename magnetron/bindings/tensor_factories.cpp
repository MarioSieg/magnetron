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

#include <core/mag_operator.h>

namespace mag::bindings {
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
        mag_error_t err {};
        throw_if_error(mag_tensor_set_requires_grad(&err, t, true), err);
    }

    void init_tensor_class_factories(nb::class_<tensor_wrapper> &cls) {
        cls.attr("empty") = nb::cpp_function(
            [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
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
                mag_error_t err {};
                if (shape.empty()) throw_if_error(mag_empty_scalar(&err, &out, ctx, dt.v), err);
                else throw_if_error(mag_empty(&err, &out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data()), err);
                if (requires_grad) {
                    throw_if_error(mag_tensor_set_requires_grad(&err, out, true), err);
                }
                return tensor_wrapper {out};
            }
        );
        cls.attr("empty_like") = nb::cpp_function(
            [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_empty_like(&err, &out, *like), err);
                mag_context_t *ctx = get_ctx();
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("scalar") = nb::cpp_function(
            [](nb::handle value, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                dtype_wrapper dt = kw_dtype_or(kwargs, deduce_dtype_from_py_scalar(value));
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_scalar_t s = scalar_from_py(value);
                mag_error_t err {};
                throw_if_error(mag_scalar(&err, &out, ctx, dt.v, s), err);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("full") = nb::cpp_function(
            [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                if (!kwargs.contains("fill_value"))
                    throw nb::type_error("full() missing keyword argument 'fill_value'");
                nb::handle fill_value = kwargs["fill_value"];
                dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                std::vector<int64_t> shape = parse_shape_from_args(args);
                validate_shape(shape);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_scalar_t s = scalar_from_py(fill_value);
                mag_error_t err {};
                throw_if_error(mag_full(&err, &out, ctx, dt.v, (int64_t) shape.size(), shape.data(), s), err);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("full_like") = nb::cpp_function(
            [](const tensor_wrapper &like, nb::handle fill_value, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_scalar_t s = scalar_from_py(fill_value);
                mag_error_t err {};
                throw_if_error(mag_full_like(&err, &out, *like, s), err);
                mag_context_t *ctx = get_ctx();
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("zeros") = nb::cpp_function(
            [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                std::vector<int64_t> shape = parse_shape_from_args(args);
                validate_shape(shape);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_zeros(&err, &out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data()), err);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("zeros_like") = nb::cpp_function(
            [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_zeros_like(&err, &out, *like), err);
                mag_context_t *ctx = get_ctx();
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("ones") = nb::cpp_function(
            [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                std::vector<int64_t> shape = parse_shape_from_args(args);
                validate_shape(shape);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_ones(&err, &out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data()), err);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("ones_like") = nb::cpp_function(
            [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_ones_like(&err, &out, *like), err);
                mag_context_t *ctx = get_ctx();
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("uniform") = nb::cpp_function(
            [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                mag_scalar_t low  = kwargs.contains("low")  ? scalar_from_py(kwargs["low"])  : mag_scalar_from_f64(0.0);
                mag_scalar_t high = kwargs.contains("high") ? scalar_from_py(kwargs["high"]) : mag_scalar_from_f64(1.0);
                std::vector<int64_t> shape = parse_shape_from_args(args);
                validate_shape(shape);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_uniform(&err, &out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data(), low, high), err);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("uniform_like") = nb::cpp_function(
            [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_scalar_t low  = kwargs.contains("low")  ? scalar_from_py(kwargs["low"])  : mag_scalar_from_f64(0.0);
                mag_scalar_t high = kwargs.contains("high") ? scalar_from_py(kwargs["high"]) : mag_scalar_from_f64(1.0);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_uniform_like(&err, &out, *like, low, high), err);
                mag_context_t *ctx = get_ctx();
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("normal") = nb::cpp_function(
            [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                dtype_wrapper dt = kw_dtype_or(kwargs, {MAG_DTYPE_FLOAT32});
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                mag_scalar_t mean = kwargs.contains("mean") ? scalar_from_py(kwargs["mean"]) : mag_scalar_from_f64(0.0);
                mag_scalar_t std  = kwargs.contains("std")  ? scalar_from_py(kwargs["std"])  : mag_scalar_from_f64(1.0);
                std::vector<int64_t> shape = parse_shape_from_args(args);
                validate_shape(shape);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_normal(&err, &out, ctx, dt.v, static_cast<int64_t>(shape.size()), shape.data(), mean, std), err);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("normal_like") = nb::cpp_function(
            [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_scalar_t mean = kwargs.contains("mean") ? scalar_from_py(kwargs["mean"]) : mag_scalar_from_f64(0.0);
                mag_scalar_t std  = kwargs.contains("std")  ? scalar_from_py(kwargs["std"])  : mag_scalar_from_f64(1.0);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_normal_like(&err, &out, *like, mean, std), err);
                mag_context_t *ctx = get_ctx();
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("bernoulli") = nb::cpp_function(
            [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_scalar_t p = kwargs.contains("p") ? scalar_from_py(kwargs["p"]) : mag_scalar_from_f64(0.5);
                std::vector<int64_t> shape = parse_shape_from_args(args);
                validate_shape(shape);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_bernoulli(&err, &out, ctx, static_cast<int64_t>(shape.size()), shape.data(), p), err);
                return tensor_wrapper{out};
            }
        );
        cls.attr("bernoulli_like") = nb::cpp_function(
            [](const tensor_wrapper &like, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                mag_scalar_t p = kwargs.contains("p") ? scalar_from_py(kwargs["p"]) : mag_scalar_from_f64(0.5);
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_bernoulli_like(&err, &out, *like, p), err);
                mag_context_t *ctx = get_ctx();
                return tensor_wrapper{out};
            }
        );
        cls.attr("arange") = nb::cpp_function(
            [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                nb::handle start_h{};
                nb::handle stop_h{};
                nb::handle step_h{};
                if (args.empty()) {
                    if (!kwargs.contains("stop") && !kwargs.contains("end"))
                        throw nb::type_error("arange() missing 'stop' or 'end'");
                    stop_h  = kwargs.contains("stop") ? kwargs["stop"] : kwargs["end"];
                    start_h = kwargs.contains("start") ? kwargs["start"] : nb::handle{};
                    step_h  = kwargs.contains("step")  ? kwargs["step"]  : nb::handle{};
                } else {
                    if (args.size() > 3) {
                        std::ostringstream oss;
                        oss << "arange() takes 1 to 3 positional args, got " << args.size();
                        throw nb::type_error(oss.str().c_str());
                    }
                    if (args.size() == 1) {
                        stop_h = args[0];
                    } else if (args.size() == 2) {
                        start_h = args[0];
                        stop_h  = args[1];
                    } else {
                        start_h = args[0];
                        stop_h  = args[1];
                        step_h  = args[2];
                    }
                }
                bool any_float = nb::isinstance<nb::float_>(stop_h) || (start_h.is_valid() && nb::isinstance<nb::float_>(start_h)) || (step_h.is_valid()  && nb::isinstance<nb::float_>(step_h));
                auto start_obj = start_h.is_valid() ? nb::borrow<nb::object>(start_h) : (any_float ? nb::object{nb::float_{0.0}} : nb::object{nb::int_{0}});
                auto step_obj = step_h.is_valid() ? nb::borrow<nb::object>(step_h) : (any_float ? nb::object{nb::float_{1.0}} : nb::object{nb::int_{1}});
                auto stop_obj = nb::borrow<nb::object>(stop_h);
                dtype_wrapper dt = kwargs.contains("dtype") ? nb::cast<dtype_wrapper>(kwargs["dtype"]) : deduce_dtype_from_py_scalar(any_float ? nb::object{nb::float_{0.0}} : nb::object{nb::int_{0}});
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                mag_scalar_t start = scalar_from_py(start_obj);
                mag_scalar_t stop = scalar_from_py(stop_obj);
                mag_scalar_t step = scalar_from_py(step_obj);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_arange(&err, &out, ctx, dt.v, start, stop, step), err);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("rand_perm") = nb::cpp_function(
            [](int64_t n, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                dtype_wrapper dt = kw_dtype_or(kwargs, dtype_wrapper{MAG_DTYPE_INT64});
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_rand_perm(&err, &out, ctx, dt.v, n), err);
                maybe_set_requires_grad(ctx, out, requires_grad);
                return tensor_wrapper{out};
            }
        );
        cls.attr("load_image") = nb::cpp_function(
            [](const std::string &path, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                std::string channels = "RGB";
                uint32_t rw = 0, rh = 0;
                if (kwargs.contains("channels"))
                    channels = nb::cast<std::string>(kwargs["channels"]);
                if (kwargs.contains("resize_to")) {
                    nb::handle rt = kwargs["resize_to"];
                    auto t = nb::cast<nb::tuple>(rt);
                    if (t.size() != 2) {
                        std::ostringstream oss;
                        oss << "resize_to must be (w, h), got tuple of size " << t.size();
                        throw nb::value_error(oss.str().c_str());
                    }
                    rw = static_cast<uint32_t>(nb::cast<int64_t>(t[0]));
                    rh = static_cast<uint32_t>(nb::cast<int64_t>(t[1]));
                }
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_load_image(&err, &out, ctx, path.c_str(), channels.c_str(), rw, rh), err);
                return tensor_wrapper{out};
            }
        );
        cls.attr("as_strided") = nb::cpp_function(
            [](const tensor_wrapper &base, nb::handle shape_h, nb::handle strides_h, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                auto shape_seq = nb::cast<nb::sequence>(shape_h);
                auto strides_seq = nb::cast<nb::sequence>(strides_h);
                if (nb::len(shape_seq) != nb::len(strides_seq)) {
                    std::ostringstream oss;
                    oss << "shape (len " << nb::len(shape_seq) << ") and strides (len " << nb::len(strides_seq) << ") length mismatch";
                    throw nb::value_error(oss.str().c_str());
                }
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
                mag_error_t err {};
                throw_if_error(mag_as_strided(&err, &out, ctx, *base, static_cast<int64_t>(shape.size()), shape.data(), strides.data(), offset), err);
                return tensor_wrapper{out};
            }
        );
        cls.attr("of") = nb::cpp_function(
            [](nb::handle data_h, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                dtype_wrapper dt = kwargs.contains("dtype") ? nb::cast<dtype_wrapper>(kwargs["dtype"]) : dtype_wrapper{MAG_DTYPE__NUM};
                bool requires_grad = kw_requires_grad_or(kwargs, false);
                if (nb::isinstance<nb::int_>(data_h) || nb::isinstance<nb::float_>(data_h) || nb::isinstance<nb::bool_>(data_h)) {
                    if (dt.v == MAG_DTYPE__NUM)
                        dt = deduce_dtype_from_py_scalar(data_h);
                    mag_context_t *ctx = get_ctx();
                    mag_tensor_t *out = nullptr;
                    mag_scalar_t s = scalar_from_py(data_h);
                    mag_error_t err {};
                    throw_if_error(mag_scalar(&err, &out, ctx, dt.v, s), err);
                    maybe_set_requires_grad(ctx, out, requires_grad);
                    return tensor_wrapper{out};
                }
                if (!nb::isinstance<nb::sequence>(data_h))
                    throw nb::type_error("Tensor.of() requires scalar or nested sequence");
                std::vector<int64_t> shape {};
                std::vector<nb::handle> stack {};
                std::vector<int64_t> idx_stack {};
                std::vector<nb::handle> flat_h {};
                nb::handle cur = data_h;
                {
                    nb::handle tmp = cur;
                    while (nb::isinstance<nb::sequence>(tmp) && !nb::isinstance<nb::str>(tmp) && !nb::isinstance<tensor_wrapper>(tmp)) {
                        auto s = nb::cast<nb::sequence>(tmp);
                        size_t n = nb::len(s);
                        if (n == 0)
                            throw nb::value_error("Tensor.of() does not support empty lists; use Tensor.empty(shape, ...)");
                        shape.emplace_back(static_cast<int64_t>(n));
                        tmp = s[0];
                    }
                    stack.emplace_back(cur);
                    idx_stack.emplace_back(0);
                    while (!stack.empty()) {
                        nb::handle top = stack.back();
                        auto &i = idx_stack.back();
                        if (nb::isinstance<nb::sequence>(top) && !nb::isinstance<nb::str>(top) && !nb::isinstance<tensor_wrapper>(top)) {
                            auto s = nb::cast<nb::sequence>(top);
                            auto n = static_cast<int64_t>(nb::len(s));
                            int64_t depth = static_cast<int64_t>(stack.size()) - 1;
                            if (depth < static_cast<int64_t>(shape.size()) && n != shape[static_cast<size_t>(depth)])
                                throw nb::value_error("Tensor.of(): ragged nested sequence");
                            if (i >= n) {
                                stack.pop_back();
                                idx_stack.pop_back();
                                if (!idx_stack.empty())
                                    idx_stack.back() += 1;
                                continue;
                            }
                            nb::handle child = s[i];
                            stack.emplace_back(child);
                            idx_stack.emplace_back(0);
                            continue;
                        }
                        flat_h.emplace_back(top);
                        stack.pop_back();
                        idx_stack.pop_back();
                        if (!idx_stack.empty())
                            idx_stack.back() += 1;
                    }
                }
                if (flat_h.empty())
                    throw nb::value_error("Tensor.of(): empty data");
                if (dt.v == MAG_DTYPE__NUM)
                    dt = deduce_dtype_from_py_scalar(flat_h[0]);
                enum class Kind { Float, SInt, UInt, Bool };
                Kind kind {};
                mag_dtype_t wide = MAG_DTYPE_FLOAT32;
                mag_dtype_mask_t mask = mag_dtype_bit(dt.v);
                if (mask & MAG_DTYPE_MASK_FP) {
                    kind = Kind::Float;
                    wide = MAG_DTYPE_FLOAT32;
                } else if (mask & MAG_DTYPE_MASK_SINT) {
                    kind = Kind::SInt;
                    wide = MAG_DTYPE_INT64;
                } else if (mask & MAG_DTYPE_MASK_UINT) {
                    kind = Kind::UInt;
                    wide = MAG_DTYPE_UINT64;
                } else if (dt.v == MAG_DTYPE_BOOLEAN) {
                    kind = Kind::Bool;
                    wide = MAG_DTYPE_UINT8;
                } else {
                    throw nb::type_error("Tensor.of(): unsupported dtype");
                }
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *raw = nullptr;
                mag_error_t err {};
                if (shape.empty()) throw_if_error(mag_empty_scalar(&err, &raw, ctx, wide), err);
                else throw_if_error(mag_empty(&err, &raw, ctx, wide, static_cast<int64_t>(shape.size()), shape.data()), err);
                maybe_set_requires_grad(ctx, raw, requires_grad);
                on_scope_exit defer_raw([raw] { mag_tensor_decref(raw); });
                size_t n = flat_h.size();
                if (kind == Kind::Float) {
                    std::vector<float> buf {};
                    buf.reserve(n);
                    for (auto &&h : flat_h) buf.emplace_back(static_cast<float>(nb::cast<double>(h)));
                    throw_if_error(mag_copy_raw_(&err, raw, buf.data(), buf.size() * sizeof(float)), err);
                } else if (kind == Kind::SInt) {
                    std::vector<int64_t> buf {};
                    buf.reserve(n);
                    for (auto &&h : flat_h) buf.emplace_back(nb::cast<int64_t>(h));
                    throw_if_error(mag_copy_raw_(&err, raw, buf.data(), buf.size() * sizeof(int64_t)), err);
                } else if (kind == Kind::UInt) {
                    std::vector<uint64_t> buf {};
                    buf.reserve(n);
                    for (auto &&h : flat_h) buf.emplace_back(nb::cast<uint64_t>(h));
                    throw_if_error(mag_copy_raw_(&err, raw, buf.data(), buf.size() * sizeof(uint64_t)), err);
                } else {
                    std::vector<uint8_t> buf {};
                    buf.reserve(n);
                    for (auto &&h : flat_h) buf.emplace_back(static_cast<uint8_t>(nb::cast<bool>(h) ? 1 : 0));
                    throw_if_error(mag_copy_raw_(&err, raw, buf.data(), buf.size() * sizeof(uint8_t)), err);
                }
                if (wide == dt.v) {
                    mag_tensor_incref(raw);
                    return tensor_wrapper{raw};
                }
                mag_tensor_t *out = nullptr;
                throw_if_error(mag_cast(&err, &out, raw, dt.v), err);
                return tensor_wrapper{out};
            }
        );
    }
}
