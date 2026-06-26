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
#include "op_recorder.hpp"

#include <core/mag_operator.h>

#include <algorithm>

#define bind_unary_pair(cls, name, opcode, doc) \
  cls \
    .def(#name, [](const tensor_wrapper &self) -> tensor_wrapper { \
      std::lock_guard lock {get_global_mutex()}; \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      if constexpr (enable_op_recorder) { \
        op_recorder::singleton().profile(opcode, [&] { \
          throw_if_error(mag_##name(&err, &out, *self), err); \
        }, {*self}); \
      } else { \
        throw_if_error(mag_##name(&err, &out, *self), err); \
      } \
      return tensor_wrapper {out}; \
    }, doc) \
    .def(#name "_", [](tensor_wrapper &self) -> tensor_wrapper& { \
      std::lock_guard lock {get_global_mutex()}; \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      if constexpr (enable_op_recorder) { \
        op_recorder::singleton().profile(opcode, [&] { \
          throw_if_error(mag_##name##_(&err, &out, *self), err); \
        }, {*self}); \
      } else { \
        throw_if_error(mag_##name##_(&err, &out, *self), err); \
      } \
      if (self) mag_tensor_decref(*self); \
      *self = out; \
      return self; \
    }, "In-place version.", nb::rv_policy::reference)

#define bind_binary_full_named(cls, dunder_name, c_name, named_name, opcode, doc) \
  cls.def("__" #dunder_name "__", \
    [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
      std::lock_guard lock {get_global_mutex()}; \
      tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      if constexpr (enable_op_recorder) { \
        op_recorder::singleton().profile(opcode, [&] { \
          throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
        }, {*a, *b}); \
      } else { \
        throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
      } \
      return tensor_wrapper{out}; \
    }, "rhs"_a, doc); \
  cls.def("__r" #dunder_name "__", \
    [](const tensor_wrapper &a, nb::handle lhs) -> tensor_wrapper { \
      std::lock_guard lock {get_global_mutex()}; \
      tensor_wrapper l = normalize_rhs_to_tensor(a, lhs); \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      if constexpr (enable_op_recorder) { \
        op_recorder::singleton().profile(opcode, [&] { \
          throw_if_error(mag_##c_name(&err, &out, *l, *a), err); \
        }, {*l, *a}); \
      } else { \
        throw_if_error(mag_##c_name(&err, &out, *l, *a), err); \
      } \
      return tensor_wrapper{out}; \
    }, "lhs"_a, "Right-hand side of " #named_name " (reflected)."); \
  cls.def("__i" #dunder_name "__", \
    [](tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper& { \
      std::lock_guard lock {get_global_mutex()}; \
      tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      if constexpr (enable_op_recorder) { \
        op_recorder::singleton().profile(opcode, [&] { \
          throw_if_error(mag_##c_name##_(&err, &out, *a, *b), err); \
        }, {*a, *b}); \
      } else { \
        throw_if_error(mag_##c_name##_(&err, &out, *a, *b), err); \
      } \
      if (a) mag_tensor_decref(*a); \
      *a = out; \
      return a; \
    }, "rhs"_a, "In-place " #named_name ".", nb::rv_policy::reference); \
  cls.def(#named_name, \
    [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
      std::lock_guard lock {get_global_mutex()}; \
      tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      if constexpr (enable_op_recorder) { \
        op_recorder::singleton().profile(opcode, [&] { \
          throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
        }, {*a, *b}); \
      } else { \
        throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
      } \
      return tensor_wrapper{out}; \
    }, "rhs"_a, doc); \
  cls.def(#named_name "_", \
    [](tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper& { \
      std::lock_guard lock {get_global_mutex()}; \
      tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      throw_if_error(mag_##c_name##_(&err, &out, *a, *b), err); \
      if (a) mag_tensor_decref(*a); \
      *a = out; \
      return a; \
    }, "rhs"_a, "In-place version.", nb::rv_policy::reference)

#define bind_compare(cls, dunder_name, c_name, named_name, opcode, doc) \
  cls.def("__" #dunder_name "__", \
    [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
      std::lock_guard lock {get_global_mutex()}; \
      tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      if constexpr (enable_op_recorder) { \
        op_recorder::singleton().profile(opcode, [&] { \
          throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
        }, {*a, *b}); \
      } else { \
        throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
      } \
      return tensor_wrapper{out}; \
    }, "rhs"_a, doc); \
  cls.def(#named_name, \
    [](const tensor_wrapper &a, nb::handle rhs) -> tensor_wrapper { \
      std::lock_guard lock {get_global_mutex()}; \
      tensor_wrapper b = normalize_rhs_to_tensor(a, rhs); \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      if constexpr (enable_op_recorder) { \
        op_recorder::singleton().profile(opcode, [&] { \
          throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
        }, {*a, *b}); \
      } else { \
        throw_if_error(mag_##c_name(&err, &out, *a, *b), err); \
      } \
      return tensor_wrapper{out}; \
    }, "rhs"_a, doc)

#define bind_stack_alias(py_name, c_fn, doc) \
  cls.attr(py_name) = nb::cpp_function( \
    [](nb::handle tensors_h) -> tensor_wrapper { \
      std::lock_guard lock {get_global_mutex()}; \
      auto tensors = parse_tensor_sequence(tensors_h, py_name); \
      auto ptrs = tensor_ptrs(tensors); \
      mag_tensor_t *out = nullptr; \
      mag_error_t err {}; \
      throw_if_error(c_fn(&err, &out, ptrs.data(), ptrs.size()), err); \
      return tensor_wrapper{out}; \
    }, \
    "tensors"_a, \
    doc \
  )

#undef MAG_BIND_STACK_ALIAS

namespace mag::bindings {
  [[nodiscard]] static std::pair<tensor_wrapper, tensor_wrapper> normalize_where_operands(const tensor_wrapper &cond, nb::handle xh, nb::handle yh) {
    if (mag_tensor_type(*cond) != MAG_DTYPE_BOOLEAN)
      throw nb::type_error("where: condition must have dtype boolean");
    bool x_is_tensor = nb::isinstance<tensor_wrapper>(xh);
    bool y_is_tensor = nb::isinstance<tensor_wrapper>(yh);
    tensor_wrapper x;
    tensor_wrapper y;
    if (x_is_tensor && y_is_tensor) {
      x = nb::cast<tensor_wrapper>(xh);
      y = nb::cast<tensor_wrapper>(yh);
    } else if (x_is_tensor) {
      x = nb::cast<tensor_wrapper>(xh);
      y = normalize_rhs_to_tensor(x, yh);
    } else if (y_is_tensor) {
      y = nb::cast<tensor_wrapper>(yh);
      x = normalize_rhs_to_tensor(y, xh);
    } else {
      dtype_wrapper dx = deduce_dtype_from_py_scalar(xh);
      dtype_wrapper dy = deduce_dtype_from_py_scalar(yh);
      mag_dtype_t promoted {};
      if (!mag_promote_type(&promoted, dx.v, dy.v))
        throw nb::type_error("where: could not promote scalar dtypes for x and y");
      x = tensor_from_py_scalar(xh, promoted, mag_tensor_device_id(*cond));
      y = tensor_from_py_scalar(yh, promoted, mag_tensor_device_id(*cond));
    }
    if (!x || !y) throw nb::value_error("where: x and y must not be null");
    return {x, y};
  }

  [[nodiscard]] static std::vector<tensor_wrapper> parse_tensor_sequence(nb::handle tensors_h, const char *op) {
    if (nb::isinstance<tensor_wrapper>(tensors_h))
      throw nb::type_error((std::string{op} + ": expected sequence of Tensor, got single Tensor").c_str());
    if (!nb::isinstance<nb::sequence>(tensors_h))
      throw nb::type_error((std::string{op} + ": 'tensors' must be a sequence of Tensor").c_str());
    auto seq = nb::cast<nb::sequence>(tensors_h);
    size_t n = nb::len(seq);
    if (n == 0)
      throw nb::value_error((std::string{op} + ": at least one tensor is required").c_str());
    std::vector<tensor_wrapper> tensors {};
    tensors.reserve(n);
    for (auto &&handle : seq) {
      auto wrapper = nb::cast<tensor_wrapper>(handle);
      if (!wrapper)
        throw nb::value_error((std::string{op} + ": encountered a null Tensor").c_str());
      tensors.emplace_back(wrapper);
    }
    return tensors;
  }

  [[nodiscard]] static std::vector<mag_tensor_t *> tensor_ptrs(std::vector<tensor_wrapper> &tensors) {
    std::vector<mag_tensor_t *> ptrs {};
    ptrs.reserve(tensors.size());
    for (auto &t : tensors)
      ptrs.emplace_back(*t);
    return ptrs;
  }

  void init_tensor_class_operators(nb::class_<tensor_wrapper> &cls) {
    cls
    .def("fill_",
      [](tensor_wrapper &self, nb::handle value) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_error_t err {};
        mag_scalar_t s = scalar_from_py_number(value);
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_FILL, [&] {
            throw_if_error(mag_fill_(&err, *self, s), err);
          }, {*self});
        } else {
          throw_if_error(mag_fill_(&err, *self, s), err);
        }
        return self;
      },
      "value"_a,
      "Fill the tensor with a scalar value."
    )
    .def("zeros_",
      [](tensor_wrapper &self) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_error_t err {};
        throw_if_error(mag_zeros_(&err, *self), err);
        return self;
      },
      "Fill the tensor with 0."
    )
    .def("ones_",
      [](tensor_wrapper &self) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_error_t err {};
        throw_if_error(mag_ones_(&err, *self), err);
        return self;
      },
      "Fill the tensor with 1."
    )
    .def("masked_fill",
      [](tensor_wrapper &self, const tensor_wrapper &mask, nb::handle value) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        if (mag_tensor_type(*mask) != MAG_DTYPE_BOOLEAN)
          throw nb::type_error("masked_fill_: mask must have dtype boolean");
        mag_error_t err {};
        mag_tensor_t *result = nullptr;
        mag_scalar_t s = scalar_from_py_number(value);
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_MASKED_FILL, [&] {
            throw_if_error(mag_masked_fill(&err, &result, *self, *mask, s), err);
          }, {*self, *mask});
        } else {
          throw_if_error(mag_masked_fill(&err, &result, *self, *mask, s), err);
        }
        return tensor_wrapper{result};
      },
      "mask"_a, "value"_a,
      "Fill elements where mask is True with value - out of place version of masked_fill_."
    )
    .def("masked_fill_",
      [](tensor_wrapper &self, const tensor_wrapper &mask, nb::handle value) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        if (mag_tensor_type(*mask) != MAG_DTYPE_BOOLEAN)
          throw nb::type_error("masked_fill_: mask must have dtype boolean");
        mag_error_t err {};
        mag_scalar_t s = scalar_from_py_number(value);
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_MASKED_FILL, [&] {
            throw_if_error(mag_masked_fill_(&err, *self, *mask, s), err);
          }, {*self, *mask});
        } else {
          throw_if_error(mag_masked_fill_(&err, *self, *mask, s), err);
        }
        return self;
      },
      "mask"_a, "value"_a,
      "Fill elements where mask is True with value."
    )
    .def("uniform_",
      [](tensor_wrapper &self, nb::handle low_h = nb::none(), nb::handle high_h = nb::none()) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_scalar_t low = low_h.is_none() ? mag_scalar_from_f64(0.0) : scalar_from_py_number(low_h);
        mag_scalar_t high = high_h.is_none() ? mag_scalar_from_f64(1.0) : scalar_from_py_number(high_h);
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_RAND_UNIFORM, [&] {
            throw_if_error(mag_uniform_(&err, *self, low, high), err);
          }, {*self});
        } else {
          throw_if_error(mag_uniform_(&err, *self, low, high), err);
        }
        return self;
      },
      "low"_a = nb::none(),
      "high"_a = nb::none(),
      "Fill with samples from uniform(low, high). Default [0, 1)."
    )
    .def("normal_",
      [](tensor_wrapper &self, nb::handle mean = nb::float_(0.0), nb::handle stdd = nb::float_(1.0)) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_scalar_t m = scalar_from_py_number(mean);
        mag_scalar_t s = scalar_from_py_number(stdd);
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_RAND_NORMAL, [&] {
            throw_if_error(mag_normal_(&err, *self, m, s), err);
          }, {*self});
        } else {
          throw_if_error(mag_normal_(&err, *self, m, s), err);
        }
        return self;
      },
      "mean"_a = 0.0,
      "std"_a = 1.0,
      "Fill with samples from normal(mean, std)."
    )
    .def("bernoulli_",
      [](tensor_wrapper &self, nb::handle p = nb::float_(0.5)) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_scalar_t pv = scalar_from_py_number(p);
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_RAND_BERNOULLI, [&] {
            throw_if_error(mag_bernoulli_(&err, *self, pv), err);
          }, {*self});
        } else {
          throw_if_error(mag_bernoulli_(&err, *self, pv), err);
        }
        return self;
      },
      "p"_a = 0.5,
      "Fill with 0/1 from Bernoulli(p)."
    )
    .def("clone", [](const tensor_wrapper &self) -> tensor_wrapper {
      std::lock_guard lock {get_global_mutex()};
      mag_tensor_t *out = nullptr;
      mag_error_t err {};
      if constexpr (enable_op_recorder) {
        op_recorder::singleton().profile(MAG_OP_CLONE, [&] {
          throw_if_error(mag_clone(&err, &out, *self), err);
        }, {*self});
      } else {
        throw_if_error(mag_clone(&err, &out, *self), err);
      }
      return tensor_wrapper{out};
    }, "Return a copy with the same data and dtype.")
    .def("copy_", [](tensor_wrapper &self, const tensor_wrapper &src) -> tensor_wrapper& {
      std::lock_guard lock {get_global_mutex()};
      mag_error_t err {};
      throw_if_error(mag_copy_(&err, *self, *src), err);
      return self;
    }, "src"_a, "Copy data from src into this tensor in-place.")
    .def("cast", [](const tensor_wrapper &self, dtype_wrapper dt) -> tensor_wrapper {
      std::lock_guard lock {get_global_mutex()};
      mag_tensor_t *out = nullptr;
      mag_error_t err {};
      if constexpr (enable_op_recorder) {
        op_recorder::singleton().profile(MAG_OP_CAST, [&] {
          throw_if_error(mag_cast(&err, &out, *self, dt.v), err);
        }, {*self});
      } else {
        throw_if_error(mag_cast(&err, &out, *self, dt.v), err);
      }
      return tensor_wrapper{out};
    }, "dtype"_a, "Return a copy with the given dtype.")
    .def("transfer", [](const tensor_wrapper &self, const std::string &device_str) -> tensor_wrapper {
      std::lock_guard lock {get_global_mutex()};
      std::optional<mag_device_id_t> device_id = parse_device_id_str(std::string{device_str});
      if (!device_id) throw std::runtime_error {"Invalid device id"};
      mag_tensor_t *out = nullptr;
      mag_error_t err {};
      throw_if_error(mag_transfer(&err, &out, *self, *device_id), err);
      return tensor_wrapper{out};
    }, "device"_a, "Return a tensor on the given device (e.g. 'cpu', 'cuda:0'). Same device returns self (shared).")
    .def("view", [](const tensor_wrapper &self, nb::args args) -> tensor_wrapper {
      std::lock_guard lock {get_global_mutex()};
      std::vector<int64_t> shape = parse_i64_dims(args, "view");
      validate_shape_infer_one(shape, "view");
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
    .def("broadcast_to", [](const tensor_wrapper &self, nb::args shape_args) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        std::vector<int64_t> shape = parse_i64_dims(shape_args, "broadcast_to");
        if (shape.empty())
          throw nb::value_error("broadcast_to: shape must not be empty");
        int64_t self_rank = mag_tensor_rank(*self);
        if (static_cast<int64_t>(shape.size()) < self_rank)
          throw nb::value_error("broadcast_to: target shape must have rank >= tensor rank");
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        throw_if_error(
          mag_broadcast_to(
            &err,
            &out,
            *self,
            static_cast<int64_t>(shape.size()),
            shape.data()
          ),
          err
        );
        return tensor_wrapper{out};
      },
      "shape"_a,
      "Broadcast tensor to a target shape without copying when possible."
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
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_MEAN, [&] {
            throw_if_error(mag_mean(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
          }, {*self});
        } else {
          throw_if_error(mag_mean(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
        }
        return tensor_wrapper{out};
      },
      "dim"_a = nb::none(), "keepdim"_a = false,
      "Mean over dim(s). None = all dims."
    )
    .def("max",
      [](const tensor_wrapper &self, nb::handle arg = nb::none(), bool keepdim = false) -> tensor_wrapper {
        std::lock_guard lock{get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err{};
        if (!arg.is_none() && nb::isinstance<tensor_wrapper>(arg)) {
          auto rhs = nb::cast<tensor_wrapper>(arg);
          if constexpr (enable_op_recorder) {
            op_recorder::singleton().profile(MAG_OP_MAX, [&] {
              throw_if_error(mag_max(&err, &out, *self, *rhs), err);
            }, {*self, *rhs});
          } else {
            throw_if_error(mag_max(&err, &out, *self, *rhs), err);
          }
          return tensor_wrapper{out};
        }
        auto ax = parse_reduction_axes(arg);
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_MAXIMA, [&] {
            throw_if_error(mag_maxima(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
          }, {*self});
        } else {
          throw_if_error(mag_maxima(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
        }
        return tensor_wrapper{out};
      },
      "dim_or_other"_a = nb::none(),
      "keepdim"_a = false,
      "Elementwise maximum with another tensor, or reduction maximum over dim(s)."
    )
    .def("min",
      [](const tensor_wrapper &self, nb::handle arg = nb::none(), bool keepdim = false) -> tensor_wrapper {
        std::lock_guard lock{get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err{};
        if (!arg.is_none() && nb::isinstance<tensor_wrapper>(arg)) {
          auto rhs = nb::cast<tensor_wrapper>(arg);
          if constexpr (enable_op_recorder) {
            op_recorder::singleton().profile(MAG_OP_MIN, [&] {
              throw_if_error(mag_min(&err, &out, *self, *rhs), err);
            }, {*self, *rhs});
          } else {
            throw_if_error(mag_min(&err, &out, *self, *rhs), err);
          }
          return tensor_wrapper{out};
        }
        auto ax = parse_reduction_axes(arg);
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_MINIMA, [&] {
            throw_if_error(mag_minima(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
          }, {*self});
        } else {
          throw_if_error(mag_minima(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
        }
        return tensor_wrapper{out};
      },
      "dim_or_other"_a = nb::none(),
      "keepdim"_a = false,
      "Elementwise minimum with another tensor, or reduction minimum over dim(s)."
    )
    .def("argmin",
      [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        auto ax = parse_reduction_axes(dim);
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_ARGMIN, [&] {
            throw_if_error(mag_argmin(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
          }, {*self});
        } else {
          throw_if_error(mag_argmin(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
        }
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
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_ARGMAX, [&] {
            throw_if_error(mag_argmax(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
          }, {*self});
        } else {
          throw_if_error(mag_argmax(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
        }
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
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_SUM, [&] {
            throw_if_error(mag_sum(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
          }, {*self});
        } else {
          throw_if_error(mag_sum(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
        }
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
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_PROD, [&] {
            throw_if_error(mag_prod(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
          }, {*self});
        } else {
          throw_if_error(mag_prod(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
        }
        return tensor_wrapper{out};
      },
      "dim"_a = nb::none(), "keepdim"_a = false,
      "Product over dim(s). None = all dims."
    )
    .def("cusum",
      [](const tensor_wrapper &self, int64_t dim) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_CUSUM, [&] {
            throw_if_error(mag_cusum(&err, &out, *self, dim), err);
          }, {*self});
        } else {
          throw_if_error(mag_cusum(&err, &out, *self, dim), err);
        }
        return tensor_wrapper{out};
      },
      "dim"_a,
      "Cumulative sum along dim."
    )
    .def("cuprod",
      [](const tensor_wrapper &self, int64_t dim) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_CUPROD, [&] {
            throw_if_error(mag_cuprod(&err, &out, *self, dim), err);
          }, {*self});
        } else {
          throw_if_error(mag_cuprod(&err, &out, *self, dim), err);
        }
        return tensor_wrapper{out};
      },
      "dim"_a,
      "Cumulative product along dim."
    )
    .def("cumax",
      [](const tensor_wrapper &self, int64_t dim) -> nb::tuple {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *values = nullptr;
        mag_tensor_t *indices = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_CUMAX, [&] {
            throw_if_error(mag_cumax(&err, &values, &indices, *self, dim), err);
          }, {*self});
        } else {
          throw_if_error(mag_cumax(&err, &values, &indices, *self, dim), err);
        }
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
      "dim"_a,
      "Cumulative maximum along dim. Returns (values, indices)."
    )
    .def("cumin",
      [](const tensor_wrapper &self, int64_t dim) -> nb::tuple {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *values = nullptr;
        mag_tensor_t *indices = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_CUMIN, [&] {
            throw_if_error(mag_cumin(&err, &values, &indices, *self, dim), err);
          }, {*self});
        } else {
          throw_if_error(mag_cumin(&err, &values, &indices, *self, dim), err);
        }
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
      "dim"_a,
      "Cumulative minimum along dim. Returns (values, indices)."
    )
    .def("all",
      [](const tensor_wrapper &self, nb::handle dim = nb::none(), bool keepdim = false) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        auto ax = parse_reduction_axes(dim);
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_ALL, [&] {
            throw_if_error(mag_all(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
          }, {*self});
        } else {
          throw_if_error(mag_all(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
        }
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
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_ANY, [&] {
            throw_if_error(mag_any(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
          }, {*self});
        } else {
          throw_if_error(mag_any(&err, &out, *self, ax.ptr, ax.rank, keepdim), err);
        }
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
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_TOPK, [&] {
            throw_if_error(mag_topk(&err, &values, &indices, *self, k, dim, largest, sorted), err);
          }, {*self});
        } else {
          throw_if_error(mag_topk(&err, &values, &indices, *self, k, dim, largest, sorted), err);
        }
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
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_TRIL, [&] {
            throw_if_error(mag_tril(&err, &out, *self, diagonal), err);
          }, {*self});
        } else {
          throw_if_error(mag_tril(&err, &out, *self, diagonal), err);
        }
        return tensor_wrapper{out};
      },
      "diagonal"_a = 0,
      "Lower triangular part; elements above diagonal set to 0."
    )
    .def("triu",
      [](const tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_TRIU, [&] {
            throw_if_error(mag_triu(&err, &out, *self, diagonal), err);
          }, {*self});
        } else {
          throw_if_error(mag_triu(&err, &out, *self, diagonal), err);
        }
        return tensor_wrapper{out};
      },
      "diagonal"_a = 0,
      "Upper triangular part; elements below diagonal set to 0."
    )
    .def("tril_",
      [](tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_TRIL, [&] {
            throw_if_error(mag_tril_(&err, &out, *self, diagonal), err);
          }, {*self});
        } else {
          throw_if_error(mag_tril_(&err, &out, *self, diagonal), err);
        }
        if (self) mag_tensor_decref(*self);
        *self = out;
        return self;
      },
      "diagonal"_a = 0,
      "In-place lower triangular.",
      nb::rv_policy::reference
    )
    .def("triu_",
      [](tensor_wrapper &self, int32_t diagonal = 0) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_TRIU, [&] {
            throw_if_error(mag_triu_(&err, &out, *self, diagonal), err);
          }, {*self});
        } else {
          throw_if_error(mag_triu_(&err, &out, *self, diagonal), err);
        }
        if (self) mag_tensor_decref(*self);
        *self = out;
        return self;
      },
      "diagonal"_a = 0,
      "In-place upper triangular.",
      nb::rv_policy::reference
    )
    .def("multinomial",
      [](const tensor_wrapper &self, int64_t num_samples = 1, bool replacement = false) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        if (num_samples <= 0)
          throw nb::value_error("multinomial: num_samples must be > 0");
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_MULTINOMIAL, [&] {
            throw_if_error(mag_multinomial(&err, &out, *self, num_samples, replacement), err);
          }, {*self});
        } else {
          throw_if_error(mag_multinomial(&err, &out, *self, num_samples, replacement), err);
        }
        return tensor_wrapper{out};
      },
      "num_samples"_a = 1, "replacement"_a = false,
      "Sample indices from probabilities (last dim). Returns shape (..., num_samples)."
    )
    .def("one_hot",
      [](const tensor_wrapper &self, int64_t num_classes) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_ONE_HOT, [&] {
            throw_if_error(mag_one_hot(&err, &out, *self, num_classes), err);
          }, {*self});
        } else {
          throw_if_error(mag_one_hot(&err, &out, *self, num_classes), err);
        }
        return tensor_wrapper{out};
      },
      "num_classes"_a = -1,
      "Return one-hot encoding of int64 class indices. If num_classes is -1, infer it from max(input)+1."
    )
    .def("gather",
      [](const tensor_wrapper &self, int64_t dim, const tensor_wrapper &index) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_GATHER, [&] {
            throw_if_error(mag_gather(&err, &out, *self, dim, *index), err);
          }, {*self, *index});
        } else {
          throw_if_error(mag_gather(&err, &out, *self, dim, *index), err);
        }
        return tensor_wrapper{out};
      },
      "dim"_a = 0,
      "index"_a,
      "Gather values along dim using index. index must have the same rank as self (torch.gather semantics)."
    )
    .def("embedding",
      [](const tensor_wrapper &self, const tensor_wrapper &indices) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_EMBEDDING, [&] {
            throw_if_error(mag_embedding(&err, &out, *self, *indices), err);
          }, {*self, *indices});
        } else {
          throw_if_error(mag_embedding(&err, &out, *self, *indices), err);
        }
        return tensor_wrapper{out};
      },
      "indices"_a,
      "Embedding lookup: self is the weight matrix [vocab_size, ...], indices is an int64 tensor of any shape. Returns indices.shape + self.shape[1:]."
    )
    .def("index_add_",
      [](tensor_wrapper &self, int64_t dim, const tensor_wrapper &index, const tensor_wrapper &source, double alpha = 1.0) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_INDEX_ADD, [&] {
            throw_if_error(mag_index_add_(&err, *self, dim, *index, *source, alpha), err);
          }, {*self, *index, *source});
        } else {
          throw_if_error(mag_index_add_(&err, *self, dim, *index, *source, alpha), err);
        }
        return self;
      },
      "dim"_a,
      "index"_a,
      "source"_a,
      "alpha"_a = 1.0,
      "Accumulate source into self along dim at the given indices."
    )
    .def("clamp",
      [](const tensor_wrapper &self, nb::handle min_h, nb::handle max_h) -> tensor_wrapper {
        std::lock_guard lock{get_global_mutex()};
        tensor_wrapper mn = normalize_rhs_to_tensor(self, min_h);
        tensor_wrapper mx = normalize_rhs_to_tensor(self, max_h);
        mag_tensor_t *out = nullptr;
        mag_error_t err{};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_CLAMP, [&] {
            throw_if_error(mag_clamp(&err, &out, *self, *mn, *mx), err);
          }, {*self, *mn, *mx});
        } else {
          throw_if_error(mag_clamp(&err, &out, *self, *mn, *mx), err);
        }
        return tensor_wrapper{out};
      },
      "min"_a,
      "max"_a,
      "Clamp tensor values elementwise into the interval [min, max]."
    )
    .def("expand",
      [](const tensor_wrapper &self, nb::args dims_args) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        std::vector<int64_t> dims = parse_i64_dims(dims_args, "expand");
        if (dims.empty())
          throw nb::value_error("expand: shape must not be empty");
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        throw_if_error(mag_expand(&err, &out, *self, static_cast<int64_t>(dims.size()), dims.data()), err);
        return tensor_wrapper{out};
      },
      "shape"_a,
      "Return a view of this tensor expanded to the given shape."
    )
    .def("pad",
      [](const tensor_wrapper &self, nb::handle pad_h, const std::string &mode = "constant", nb::handle value = nb::float_{0.0}) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        std::vector<int64_t> pad = parse_i64_list_handle(pad_h, "pad");
        mag_scalar_t sv = scalar_from_py_number(value);
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_PAD, [&] {
            throw_if_error(mag_pad(&err, &out, *self, pad.data(), static_cast<int64_t>(pad.size()), mode.c_str(), sv), err);
          }, {*self});
        } else {
          throw_if_error(mag_pad(&err, &out, *self, pad.data(), static_cast<int64_t>(pad.size()), mode.c_str(), sv), err);
        }
        return tensor_wrapper{out};
      },
      "pad"_a,
      "mode"_a = "constant",
      "value"_a = 0.0,
      "Pad tensor with the given padding."
    )
    .def("repeat",
      [](const tensor_wrapper &self, nb::args repeats_args) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        std::vector<int64_t> repeats = parse_i64_dims(repeats_args, "repeat");
        if (repeats.empty())
          throw nb::value_error("repeat: expected at least one repeat count");
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_REPEAT, [&] {
            throw_if_error(mag_repeat(&err, &out, *self, repeats.data(), static_cast<int64_t>(repeats.size())), err);
          }, {*self});
        } else {
          throw_if_error(mag_repeat(&err, &out, *self, repeats.data(), static_cast<int64_t>(repeats.size())), err);
        }
        return tensor_wrapper{out};
      },
      "*repeats"_a,
      "Repeat this tensor along each dimension."
    )
    .def("repeat_interleave",
      [](const tensor_wrapper &self, nb::handle repeats_h, nb::object dim_o = nb::none()) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        bool flatten = dim_o.is_none();
        int64_t dim = 0;
        if (!flatten)
          dim = nb::cast<int64_t>(dim_o);
        std::vector<int64_t> counts {};
        if (nb::isinstance<tensor_wrapper>(repeats_h)) {
          auto repeats_tensor = nb::cast<tensor_wrapper>(repeats_h);
          mag_tensor_t *tensor = *repeats_tensor;
          if (!tensor || tensor->dtype != MAG_DTYPE_INT64 || tensor->coords.rank != 1)
            throw nb::type_error("repeat_interleave: tensor repeats must be 1-D int64");
          mag_tensor_t *contig = nullptr, *host = nullptr;
          mag_error_t err {};
          throw_if_error(mag_contiguous(&err, &contig, tensor), err);
          throw_if_error(mag_transfer(&err, &host, contig, mag_device(CPU, 0)), err);
          const auto *data_ptr = reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(host));
          counts.assign(data_ptr, data_ptr + host->numel);
          mag_tensor_decref(host);
          mag_tensor_decref(contig);
        } else if (nb::isinstance<nb::int_>(repeats_h) || PyLong_Check(repeats_h.ptr())) {
          counts = { nb::cast<int64_t>(repeats_h) };
        } else {
          counts = parse_i64_list_handle(repeats_h, "repeat_interleave");
        }
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_REPEAT_INTERLEAVE, [&] {
            throw_if_error(mag_repeat_interleave(&err, &out, *self, flatten, dim, counts.data(), static_cast<int64_t>(counts.size())), err);
          }, {*self});
        } else {
          throw_if_error(mag_repeat_interleave(&err, &out, *self, flatten, dim, counts.data(), static_cast<int64_t>(counts.size())), err);
        }
        return tensor_wrapper{out};
      },
      "repeats"_a,
      "dim"_a = nb::none(),
      "Repeat elements of this tensor interleaved along a dimension."
    );

    cls.attr("cat") = nb::cpp_function(
     [](nb::handle tensors_h, int64_t dim = 0) -> tensor_wrapper {
       std::lock_guard lock {get_global_mutex()};
       auto tensors = parse_tensor_sequence(tensors_h, "cat");
       auto ptrs = tensor_ptrs(tensors);
       mag_tensor_t *out = nullptr;
       mag_error_t err {};
       if constexpr (enable_op_recorder) {
         op_recorder::singleton().profile(MAG_OP_CAT, [&] {
           throw_if_error(mag_cat(&err, &out, ptrs.data(), ptrs.size(), dim), err);
         }, ptrs);
       } else {
         throw_if_error(mag_cat(&err, &out, ptrs.data(), ptrs.size(), dim), err);
       }
       return tensor_wrapper{out};
     },
     "tensors"_a,
     "dim"_a = 0,
     "Concatenate tensors along the given dimension."
    );

    cls.attr("stack") = nb::cpp_function(
      [](nb::handle tensors_h, int64_t dim = 0) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        auto tensors = parse_tensor_sequence(tensors_h, "stack");
        auto ptrs = tensor_ptrs(tensors);
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        throw_if_error(mag_stack(&err, &out, ptrs.data(), ptrs.size(), dim), err);
        return tensor_wrapper{out};
      },
      "tensors"_a,
      "dim"_a = 0,
      "Stack tensors along a new dimension."
    );

    bind_stack_alias("hstack", mag_hstack, "Stack tensors horizontally.");
    bind_stack_alias("vstack", mag_vstack, "Stack tensors vertically.");
    bind_stack_alias("dstack", mag_dstack, "Stack tensors depthwise.");

    cls.attr("where") = nb::cpp_function([](const tensor_wrapper &cond, nb::handle xh, nb::handle yh) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        auto [x, y] = normalize_where_operands(cond, xh, yh);
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_WHERE, [&] {
            throw_if_error(mag_where(&err, &out, *cond, *x, *y), err);
          }, {*cond, *x, *y});
        } else {
          throw_if_error(mag_where(&err, &out, *cond, *x, *y), err);
        }
        return tensor_wrapper{out};
      },
      "condition"_a, "x"_a, "y"_a,
      "Return elements from x where condition is True, otherwise from y."
    );

    cls.def_static("einsum", [](const std::string &equation, nb::args operands) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        size_t n = operands.size();
        if (n == 0)
          throw nb::value_error("einsum: at least one tensor is required");
        std::vector<tensor_wrapper> tensors {};
        tensors.reserve(n);
        std::vector<mag_tensor_t *> ptrs {};
        ptrs.reserve(n);
        for (auto &&handle : operands) {
          auto wrapper = nb::cast<tensor_wrapper>(handle);
          if (!wrapper)
            throw nb::value_error("einsum: encountered a null Tensor");
          ptrs.emplace_back(*tensors.emplace_back(wrapper));
        }
        mag_error_t err {};
        mag_tensor_t *out = nullptr;
        throw_if_error(mag_einsum(&err, &out, equation.c_str(), ptrs.data(), ptrs.size()), err);
        return tensor_wrapper{out};
      },
      "equation"_a,
      "*operands"_a,
      "Einstein summation over the given operands."
    );

    // Unary operators
    bind_unary_pair(cls, abs, MAG_OP_ABS, "Element-wise absolute value.");
    bind_unary_pair(cls, sgn, MAG_OP_SGN, "Element-wise sign (-1, 0, or 1).");
    bind_unary_pair(cls, neg, MAG_OP_NEG, "Element-wise negation.");
    bind_unary_pair(cls, log, MAG_OP_LOG, "Natural logarithm.");
    bind_unary_pair(cls, log10, MAG_OP_LOG10, "Base-10 logarithm.");
    bind_unary_pair(cls, log1p, MAG_OP_LOG1P, "log(1 + x).");
    bind_unary_pair(cls, log2, MAG_OP_LOG2, "Base-2 logarithm.");
    bind_unary_pair(cls, sqr, MAG_OP_SQR, "Element-wise square.");
    bind_unary_pair(cls, rcp, MAG_OP_RCP, "Reciprocal 1/x.");
    bind_unary_pair(cls, sqrt, MAG_OP_SQRT, "Element-wise square root.");
    bind_unary_pair(cls, rsqrt, MAG_OP_RSQRT, "Reciprocal square root 1/sqrt(x).");
    bind_unary_pair(cls, sin, MAG_OP_SIN, "Element-wise sine.");
    bind_unary_pair(cls, cos, MAG_OP_COS, "Element-wise cosine.");
    bind_unary_pair(cls, tan, MAG_OP_TAN, "Element-wise tangent.");
    bind_unary_pair(cls, sinh, MAG_OP_SINH, "Element-wise hyperbolic sine.");
    bind_unary_pair(cls, cosh, MAG_OP_COSH, "Element-wise hyperbolic cosine.");
    bind_unary_pair(cls, tanh, MAG_OP_TANH, "Element-wise hyperbolic tangent.");
    bind_unary_pair(cls, asin, MAG_OP_ASIN, "Element-wise arc sine.");
    bind_unary_pair(cls, acos, MAG_OP_ACOS, "Element-wise arc cosine.");
    bind_unary_pair(cls, atan, MAG_OP_ATAN, "Element-wise arc tangent.");
    bind_unary_pair(cls, asinh, MAG_OP_ASINH, "Element-wise inverse hyperbolic sine.");
    bind_unary_pair(cls, acosh, MAG_OP_ACOSH, "Element-wise inverse hyperbolic cosine.");
    bind_unary_pair(cls, atanh, MAG_OP_ATANH, "Element-wise inverse hyperbolic tangent.");
    bind_unary_pair(cls, step, MAG_OP_STEP, "Heaviside step (0 if x < 0 else 1).");
    bind_unary_pair(cls, erf, MAG_OP_ERF, "Error function.");
    bind_unary_pair(cls, erfc, MAG_OP_ERFC, "Complementary error function.");
    bind_unary_pair(cls, exp, MAG_OP_EXP, "Element-wise exp(x).");
    bind_unary_pair(cls, exp2, MAG_OP_EXP2, "Element-wise base-2 exponential.");
    bind_unary_pair(cls, expm1, MAG_OP_EXPM1, "exp(x) - 1.");
    bind_unary_pair(cls, floor, MAG_OP_FLOOR, "Round down to integer.");
    bind_unary_pair(cls, ceil, MAG_OP_CEIL, "Round up to integer.");
    bind_unary_pair(cls, round, MAG_OP_ROUND, "Round to nearest integer.");
    bind_unary_pair(cls, trunc, MAG_OP_TRUNC, "Truncate toward zero.");

    // Softmax has params and required a specialized binding
    cls.def("softmax",
      [](const tensor_wrapper &self, [[maybe_unused]] int64_t dim) -> tensor_wrapper {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_SOFTMAX, [&] {
            throw_if_error(mag_softmax(&err, &out, *self), err); // TODO: respect dim
          }, {*self});
        } else {
          throw_if_error(mag_softmax(&err, &out, *self), err); // TODO: respect dim
        }
        return tensor_wrapper{out};
      },
      "dim"_a = -1,
      "Softmax over dim (normalizes to sum to 1)."
    );
    cls.def("softmax_",
      [](tensor_wrapper &self, [[maybe_unused]] int64_t dim) -> tensor_wrapper& {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *out = nullptr;
        mag_error_t err {};
        if constexpr (enable_op_recorder) {
          op_recorder::singleton().profile(MAG_OP_SOFTMAX, [&] {
            throw_if_error(mag_softmax_(&err, &out, *self), err); // TODO: respect dim
          }, {*self});
        } else {
          throw_if_error(mag_softmax_(&err, &out, *self), err); // TODO: respect dim
        }
        if (self) mag_tensor_decref(*self);
        *self = out;
        return self;
      },
      "dim"_a = -1,
      "In-place softmax.",
      nb::rv_policy::reference
    );

    bind_unary_pair(cls, softmax_dv, MAG_OP_SOFTMAX_DV, "Softmax derivative (for autodiff).");
    bind_unary_pair(cls, sigmoid, MAG_OP_SIGMOID, "Sigmoid 1/(1+exp(-x)).");
    bind_unary_pair(cls, sigmoid_dv, MAG_OP_SIGMOID_DV, "Sigmoid derivative.");
    bind_unary_pair(cls, hard_sigmoid, MAG_OP_HARD_SIGMOID, "Hard sigmoid approximation.");
    bind_unary_pair(cls, silu, MAG_OP_SILU, "SiLU (Swish) x*sigmoid(x).");
    bind_unary_pair(cls, silu_dv, MAG_OP_SILU_DV, "SiLU derivative.");
    bind_unary_pair(cls, tanh_dv, MAG_OP_TANH_DV, "Tanh derivative.");
    bind_unary_pair(cls, relu, MAG_OP_RELU, "ReLU max(0, x).");
    bind_unary_pair(cls, relu_dv, MAG_OP_RELU_DV, "ReLU derivative.");
    bind_unary_pair(cls, gelu, MAG_OP_GELU, "GELU activation.");
    bind_unary_pair(cls, gelu_approx, MAG_OP_GELU_APPROX, "GELU approximate form.");
    bind_unary_pair(cls, gelu_dv, MAG_OP_GELU_DV, "GELU derivative.");
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
    bind_binary_full_named(cls, add, add, add, MAG_OP_ADD, "Element-wise addition.");
    bind_binary_full_named(cls, sub, sub, sub, MAG_OP_SUB, "Element-wise subtraction.");
    bind_binary_full_named(cls, mul, mul, mul, MAG_OP_MUL, "Element-wise multiplication.");
    bind_binary_full_named(cls, mod, mod, mod, MAG_OP_MOD, "Element-wise modulo.");
    bind_binary_full_named(cls, pow, pow, pow, MAG_OP_POW, "Element-wise exponentiation.");
    bind_binary_full_named(cls, truediv, div, truediv, MAG_OP_DIV, "Element-wise true division.");
    bind_binary_full_named(cls, floordiv, floordiv, floordiv, MAG_OP_FLOORDIV, "Element-wise floor division.");
    bind_binary_full_named(cls, and, and, logical_and, MAG_OP_AND, "Element-wise logical AND.");
    bind_binary_full_named(cls, or, or, logical_or, MAG_OP_OR, "Element-wise logical OR.");
    bind_binary_full_named(cls, xor, xor, logical_xor, MAG_OP_XOR, "Element-wise logical XOR.");
    bind_binary_full_named(cls, lshift, shl, lshift, MAG_OP_SHL, "Element-wise left shift.");
    bind_binary_full_named(cls, rshift, shr, rshift, MAG_OP_SHR, "Element-wise right shift.");
    bind_compare(cls, lt, lt, lt, MAG_OP_LT, "Element-wise less than. Returns boolean tensor.");
    bind_compare(cls, le, le, le, MAG_OP_LE, "Element-wise less or equal.");
    bind_compare(cls, gt, gt, gt, MAG_OP_GT, "Element-wise greater than.");
    bind_compare(cls, ge, ge, ge, MAG_OP_GE, "Element-wise greater or equal.");
    bind_compare(cls, eq, eq, eq, MAG_OP_EQ, "Element-wise equality.");
    bind_compare(cls, ne, ne, ne, MAG_OP_NE, "Element-wise not equal.");

    auto matmul_impl = [](const tensor_wrapper &self, nb::handle rhs) -> tensor_wrapper {
      std::lock_guard lock{get_global_mutex()};
      tensor_wrapper b = normalize_rhs_to_tensor(self, rhs);
      mag_tensor_t *out = nullptr;
      mag_error_t err{};
      if constexpr (enable_op_recorder) {
        op_recorder::singleton().profile(MAG_OP_MATMUL, [&] () -> void {
          throw_if_error(mag_matmul(&err, &out, *self, *b), err);
        }, {*self, *b});
      } else {
        throw_if_error(mag_matmul(&err, &out, *self, *b), err);
      }
      return tensor_wrapper{out};
    };

    cls.def("__matmul__", matmul_impl,
      "rhs"_a,
      "Matrix multiplication. Supports @ operator."
    );
    cls.def("matmul", matmul_impl,
      "rhs"_a,
      "Matrix multiplication."
    );
  }
}
