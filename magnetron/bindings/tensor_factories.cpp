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

    void init_tensor_class_factories(nb::class_<tensor_wrapper> &cls) {
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
            [](nb::args args, nb::kwargs kwargs) -> tensor_wrapper {
                nb::handle start_h = nb::handle{};
                nb::handle stop_h  = nb::handle{};
                nb::handle step_h  = nb::handle{};

                // kwargs path: arange(stop=..., start=..., step=...)
                if (args.size() == 0) {
                    if (!kwargs.contains("stop") && !kwargs.contains("end"))
                        throw nb::type_error("arange(...): missing 'stop' (or 'end')");
                    stop_h  = kwargs.contains("stop") ? kwargs["stop"] : kwargs["end"];
                    start_h = kwargs.contains("start") ? kwargs["start"] : nb::handle{};
                    step_h  = kwargs.contains("step")  ? kwargs["step"]  : nb::handle{};
                } else {
                    // positional path: arange(stop) / arange(start, stop) / arange(start, stop, step)
                    if (args.size() > 3)
                        throw nb::type_error("arange(start?, stop, step?): expected 1..3 positional args");

                    if (args.size() == 1) {
                        stop_h = args[0];
                    } else if (args.size() == 2) {
                        start_h = args[0];
                        stop_h  = args[1];
                    } else { // 3
                        start_h = args[0];
                        stop_h  = args[1];
                        step_h  = args[2];
                    }
                }

                bool any_float =
                    nb::isinstance<nb::float_>(stop_h) ||
                    (start_h.is_valid() && nb::isinstance<nb::float_>(start_h)) ||
                    (step_h.is_valid()  && nb::isinstance<nb::float_>(step_h));

                nb::object start_obj = start_h.is_valid()
                    ? nb::borrow<nb::object>(start_h)
                    : (any_float ? nb::object{nb::float_{0.0}} : nb::object{nb::int_{0}});

                nb::object step_obj = step_h.is_valid()
                    ? nb::borrow<nb::object>(step_h)
                    : (any_float ? nb::object{nb::float_{1.0}} : nb::object{nb::int_{1}});

                nb::object stop_obj = nb::borrow<nb::object>(stop_h);

                dtype_wrapper dt = kwargs.contains("dtype")
                    ? nb::cast<dtype_wrapper>(kwargs["dtype"])
                    : deduce_dtype_from_py_scalar(any_float ? nb::object{nb::float_{0.0}} : nb::object{nb::int_{0}});

                bool requires_grad = kw_requires_grad_or(kwargs, false);

                mag_scalar_t start = scalar_from_py(start_obj);
                mag_scalar_t stop  = scalar_from_py(stop_obj);
                mag_scalar_t step  = scalar_from_py(step_obj);

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
}
