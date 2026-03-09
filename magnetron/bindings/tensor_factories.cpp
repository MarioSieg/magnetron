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

#include <nanobind/ndarray.h>

#include <core/mag_operator.h>

namespace mag::bindings {

    /** Map nanobind ndarray dtype to mag_dtype_t. Returns MAG_DTYPE__NUM if unsupported. */
    template <typename... Args>
    [[nodiscard]] static mag_dtype_t ndarray_dtype_to_mag_dtype(const nb::ndarray<Args...> &arr) {
        if (arr.dtype() == nb::dtype<float>()) return MAG_DTYPE_FLOAT32;
        /* MAG_DTYPE_FLOAT64 not yet in magnetron; float64 arrays rejected below */
        if (arr.dtype() == nb::dtype<bool>()) return MAG_DTYPE_BOOLEAN;
        if (arr.dtype() == nb::dtype<uint8_t>()) return MAG_DTYPE_UINT8;
        if (arr.dtype() == nb::dtype<int8_t>()) return MAG_DTYPE_INT8;
        if (arr.dtype() == nb::dtype<uint16_t>()) return MAG_DTYPE_UINT16;
        if (arr.dtype() == nb::dtype<int16_t>()) return MAG_DTYPE_INT16;
        if (arr.dtype() == nb::dtype<uint32_t>()) return MAG_DTYPE_UINT32;
        if (arr.dtype() == nb::dtype<int32_t>()) return MAG_DTYPE_INT32;
        if (arr.dtype() == nb::dtype<uint64_t>()) return MAG_DTYPE_UINT64;
        if (arr.dtype() == nb::dtype<int64_t>()) return MAG_DTYPE_INT64;
        return MAG_DTYPE__NUM;
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
        mag_error_t err {};
        throw_if_error(mag_tensor_set_requires_grad(&err, t, true), err);
    }

    // Create a tensor from a Python scalar, list or Numpy/Pytorch CPU tensor.
    [[nodiscard]] static tensor_wrapper tensor_from_data(nb::handle data_h, nb::kwargs &kwargs) {
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
        /* Accept NumPy, PyTorch, etc. when CPU and C-contiguous (nanobind may copy if not). */
        try {
            auto arr = nb::cast<nb::ndarray<nb::c_contig, nb::device::cpu>>(data_h);
            mag_dtype_t elem_dtype = ndarray_dtype_to_mag_dtype(arr);
            if (elem_dtype == MAG_DTYPE__NUM)
                throw nb::type_error("Tensor() from array: unsupported dtype (use float32, bool, or int/uint 8/16/32/64)");
            mag_dtype_t target_dtype = (dt.v != MAG_DTYPE__NUM ? dt.v : elem_dtype);
            std::vector<int64_t> shape {};
            shape.reserve(arr.ndim());
            for (size_t i=0; i < arr.ndim(); ++i)
                shape.emplace_back(static_cast<int64_t>(arr.shape(i)));
            mag_context_t *ctx = get_ctx();
            mag_tensor_t *raw = nullptr;
            mag_error_t err {};
            if (shape.empty()) throw_if_error(mag_empty_scalar(&err, &raw, ctx, elem_dtype), err);
            else throw_if_error(mag_empty(&err, &raw, ctx, elem_dtype, static_cast<int64_t>(shape.size()), shape.data()), err);
            maybe_set_requires_grad(ctx, raw, requires_grad);
            on_scope_exit defer_raw([raw] { mag_tensor_decref(raw); });
            size_t numel = 1;
            for (int64_t s : shape)
                numel *= static_cast<size_t>(s);
            size_t item_size = mag_type_trait(elem_dtype)->size;
            throw_if_error(mag_copy_raw_(&err, raw, arr.data(), numel * item_size), err);
            if (target_dtype == elem_dtype) {
                mag_tensor_incref(raw);
                return tensor_wrapper{raw};
            }
            mag_tensor_t *out = nullptr;
            throw_if_error(mag_cast(&err, &out, raw, target_dtype), err);
            return tensor_wrapper{out};
        } catch (const nb::cast_error &) {
            /* Not array-like - fall through to sequence path */
        }
        if (!nb::isinstance<nb::sequence>(data_h))
            throw nb::type_error("Tensor() requires scalar, array (NumPy/torch CPU contiguous), or nested sequence");
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
                    throw nb::value_error("Tensor() does not support empty lists; use Tensor.empty(shape, ...)");
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
                        throw nb::value_error("Tensor(): ragged nested sequence");
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
            throw nb::value_error("Tensor(): empty data");
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
            throw nb::type_error("Tensor(): unsupported dtype");
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

    void init_tensor_class_factories(nb::class_<tensor_wrapper> &cls) {
        cls.def("__init__",
            [](tensor_wrapper *self, const tensor_wrapper &other) {
                std::lock_guard lock {get_global_mutex()};
                new (self) tensor_wrapper(other);
                mag_error_t err {};
                throw_if_error(mag_tensor_set_requires_grad(&err, **self, true), err);
            },
            "other"_a,
            "Wrap an existing tensor (e.g. for Parameter). Sets requires_grad=True."
        );
        cls.def("__init__",
            [](tensor_wrapper *self, nb::handle data_h, nb::kwargs kwargs) {
                std::lock_guard lock {get_global_mutex()};
                new (self) tensor_wrapper {tensor_from_data(data_h, kwargs)};
            },
            "data"_a, "kwargs"_a,
            "Create tensor from scalar, array (NumPy/PyTorch), or nested list. Kwargs: dtype, requires_grad."
        );
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
                    auto seq = nb::cast<nb::sequence>(args[0]);
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
            },
            "Create an uninitialized tensor. Args: shape (ints or sequence). Kwargs: dtype, requires_grad."
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
            },
            "Uninitialized tensor with same shape and dtype as like."
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
            },
            "Create a 0-dim tensor from a Python scalar. Kwargs: dtype, requires_grad."
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
            },
            "Tensor of shape filled with fill_value. Kwargs: dtype, fill_value, requires_grad."
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
            },
            "Tensor like input filled with fill_value. Kwargs: requires_grad."
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
            },
            "Tensor of shape filled with zeros. Kwargs: dtype, requires_grad."
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
            },
            "Zeros with same shape and dtype as like. Kwargs: requires_grad."
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
            },
            "Tensor of shape filled with ones. Kwargs: dtype, requires_grad."
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
            },
            "Ones with same shape and dtype as like. Kwargs: requires_grad."
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
            },
            "Tensor of shape with values from uniform(low, high). Kwargs: dtype, low, high, requires_grad."
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
            },
            "Like input, filled with uniform(low, high). Kwargs: low, high, requires_grad."
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
            },
            "Tensor of shape with values from normal(mean, std). Kwargs: dtype, mean, std, requires_grad."
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
            },
            "Like input, filled with normal(mean, std). Kwargs: mean, std, requires_grad."
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
            },
            "Binary tensor of shape from Bernoulli(p). Kwargs: p."
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
            },
            "Binary tensor like input from Bernoulli(p). Kwargs: p."
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
            },
            "1D tensor of values [start, stop) with step. Use stop only, or start/stop, or start/stop/step. Kwargs: dtype, requires_grad."
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
            },
            "Random permutation of [0, n). Kwargs: dtype, requires_grad."
        );
        cls.attr("load_image") = nb::cpp_function(
            [](const std::string &path, nb::kwargs kwargs) -> tensor_wrapper {
                std::lock_guard lock {get_global_mutex()};
                std::string channels = "RGB";
                uint32_t rw = 0, rh = 0;
                if (kwargs.contains("channels"))
                    channels = nb::cast<std::string>(kwargs["channels"]);
                for (auto &c : channels) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                if (channels != "GRAY" && channels != "GRAY_ALPHA" && channels != "RGB" && channels != "RGBA") {
                    std::ostringstream oss;
                    oss << "Invalid channels: " << channels << ". Must be one of GRAY, GRAY_ALPHA, RGB, RGBA.";
                    throw nb::value_error(oss.str().c_str());
                }
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
            },
            "Load image from path. Kwargs: channels (e.g. 'RGB'), resize_to (w, h)."
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
                std::vector<int64_t> shape {};
                std::vector<int64_t> strides {};
                shape.reserve(nb::len(shape_seq));
                strides.reserve(nb::len(strides_seq));
                for (auto &&h : shape_seq) shape.emplace_back(nb::cast<int64_t>(h));
                for (auto &&h : strides_seq) strides.emplace_back(nb::cast<int64_t>(h));
                validate_shape(shape);
                int64_t offset = 0;
                if (kwargs.contains("offset"))
                    offset = nb::cast<int64_t>(kwargs["offset"]);
                mag_context_t *ctx = get_ctx();
                mag_tensor_t *out = nullptr;
                mag_error_t err {};
                throw_if_error(mag_as_strided(&err, &out, ctx, *base, static_cast<int64_t>(shape.size()), shape.data(), strides.data(), offset), err);
                return tensor_wrapper{out};
            },
            "View of base with given shape and strides. Kwargs: offset."
        );
    }
}
