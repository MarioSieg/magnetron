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
#include "core/mag_tensor.h"

namespace mag::bindings {
  template <typename T>
  [[nodiscard]] static nb::object ndarray_to_obj(nb::ndarray<nb::numpy, T> &&arr) { return nb::cast(std::move(arr)); }

  [[nodiscard]] static nb::object ndarray_f32(void *base, int64_t rank, const std::vector<size_t> &shape, const std::vector<int64_t> &strides, const nb::object &owner) {
    if (!rank) return ndarray_to_obj(nb::ndarray<nb::numpy, float>(base, {}, owner, {}, nb::dtype<float>()));
    return ndarray_to_obj(nb::ndarray<nb::numpy, float>(base, static_cast<size_t>(rank), shape.data(), owner, strides.data(), nb::dtype<float>()));
  }

  [[nodiscard]] static nb::object ndarray_f16(void *base, int64_t rank, const std::vector<size_t> &shape, const std::vector<int64_t> &strides, const nb::object &owner) {
    if (!rank) return ndarray_to_obj(nb::ndarray<nb::numpy, mag_float16_t>(base, {}, owner, {}, nb::dtype<mag_float16_t>()));
    return ndarray_to_obj(nb::ndarray<nb::numpy, mag_float16_t>(base, static_cast<size_t>(rank), shape.data(), owner, strides.data(), nb::dtype<mag_float16_t>()));
  }

  [[nodiscard]] static nb::object ndarray_bf16(void *base, int64_t rank, const std::vector<size_t> &shape, const std::vector<int64_t> &strides, const nb::object &owner) {
    if (!rank) return ndarray_to_obj(nb::ndarray<nb::numpy, mag_bfloat16_t>(base, {}, owner, {}, nb::dtype<mag_bfloat16_t>()));
    return ndarray_to_obj(nb::ndarray<nb::numpy, mag_bfloat16_t>(base, static_cast<size_t>(rank), shape.data(), owner, strides.data(), nb::dtype<mag_bfloat16_t>()));
  }

  template <typename T>
  [[nodiscard]] static nb::object ndarray_int(void *base, int64_t rank, const std::vector<size_t> &shape, const std::vector<int64_t> &strides, const nb::object &owner) {
    if (!rank) return ndarray_to_obj(nb::ndarray<nb::numpy, T>(static_cast<T *>(base), {}, owner, {}, nb::dtype<T>()));
    return ndarray_to_obj(nb::ndarray<nb::numpy, T>(static_cast<T *>(base), static_cast<size_t>(rank), shape.data(), owner, strides.data(), nb::dtype<T>()));
  }

  [[nodiscard]] static nb::object ndarray_bool(void *base, int64_t rank, const std::vector<size_t> &shape, const std::vector<int64_t> &strides, const nb::object &owner) {
    if (!rank) return ndarray_to_obj(nb::ndarray<nb::numpy, uint8_t>(base, {}, owner, {}, nb::dtype<bool>()));
    return ndarray_to_obj(nb::ndarray<nb::numpy, uint8_t>(base, static_cast<size_t>(rank), shape.data(), owner, strides.data(), nb::dtype<bool>()));
  }

  [[nodiscard]] static nb::object tensor_from_numpy(const tensor_wrapper &self) {
    std::lock_guard lock {get_global_mutex()};
    mag_error_t err {};
    mag_device_id_t cpu = mag_device(CPU, 0);
    mag_tensor_t *tensor = *self;
    tensor_wrapper host {};
    if (!mag_device_id_eq(mag_tensor_device_id(tensor), cpu)) {
      mag_tensor_t *host_ptr = nullptr;
      throw_if_error(mag_transfer(&err, &host_ptr, tensor, cpu), err);
      host = tensor_wrapper{host_ptr};
      tensor = host_ptr;
    }
    int64_t rank = mag_tensor_rank(tensor);
    const int64_t *p_shape = mag_tensor_shape_ptr(tensor);
    const int64_t *p_strides = mag_tensor_strides_ptr(tensor);
    void *base = reinterpret_cast<void *>(mag_tensor_data_ptr(tensor));
    std::vector<size_t> shape(static_cast<size_t>(std::max<int64_t>(0, rank)));
    std::vector<int64_t> strides(static_cast<size_t>(std::max<int64_t>(0, rank)));
    for (size_t i=0; i < static_cast<size_t>(rank); ++i) {
      shape[i] = static_cast<size_t>(p_shape[i]);
      strides[i] = p_strides[i];
    }
    mag_dtype_t dtype = mag_tensor_type(tensor);
    nb::object owner = *host ? nb::cast(host) : nb::cast(self);
    switch (dtype) {
      /* MAG_DTYPE_FLOAT64 not yet in magnetron; float64 arrays rejected below */
      case MAG_DTYPE_FLOAT32: return ndarray_f32(base, rank, shape, strides, owner);
      case MAG_DTYPE_FLOAT16: return ndarray_f16(base, rank, shape, strides, owner);
      case MAG_DTYPE_BFLOAT16: return ndarray_bf16(base, rank, shape, strides, owner);
      case MAG_DTYPE_BOOLEAN: return ndarray_bool(base, rank, shape, strides, owner);
      case MAG_DTYPE_UINT8: return ndarray_int<uint8_t>(base, rank, shape, strides, owner);
      case MAG_DTYPE_INT8: return ndarray_int<int8_t>(base, rank, shape, strides, owner);
      case MAG_DTYPE_UINT16: return ndarray_int<uint16_t>(base, rank, shape, strides, owner);
      case MAG_DTYPE_INT16: return ndarray_int<int16_t>(base, rank, shape, strides, owner);
      case MAG_DTYPE_UINT32: return ndarray_int<uint32_t>(base, rank, shape, strides, owner);
      case MAG_DTYPE_INT32: return ndarray_int<int32_t>(base, rank, shape, strides, owner);
      case MAG_DTYPE_UINT64: return ndarray_int<uint64_t>(base, rank, shape, strides, owner);
      case MAG_DTYPE_INT64: return ndarray_int<int64_t>(base, rank, shape, strides, owner);
      default: throw nb::type_error("Tensor.numpy(): unsupported dtype");
    }
  }

  template <typename T, typename PT>
  [[nodiscard]] static nb::object build_list_recursive(
    const T *data,
    const int64_t *shape,
    const int64_t *strides,
    int64_t rank,
    int64_t offset,
    int64_t dim
  ) {
    if (dim == rank) return PT(data[offset]);
    int64_t size = shape[dim];
    PyObject *raw = PyList_New(size); // We use the raw Python API to preallocate the list's capacity
    if (!raw) throw std::runtime_error {"Failed to allocate list for tolist()"};
    for (int64_t i=0; i < size; ++i) {
      nb::object item = build_list_recursive<T, PT>(
        data,
        shape,
        strides,
        rank,
        offset + i*strides[dim],
        dim + 1
      );
      PyList_SET_ITEM(raw, i, item.release().ptr());
    }
    return nb::steal<nb::object>(raw);
  }

  static void init_tensor_class_base(nb::class_<tensor_wrapper> &cls) {
    cls
    .def_prop_ro("rank", [](const tensor_wrapper &self) -> int64_t {
      std::lock_guard lock {get_global_mutex()};
      return mag_tensor_rank(*self);
    }, "Number of dimensions.")
    .def_prop_ro("numel", [](const tensor_wrapper &self) -> int64_t {
      std::lock_guard lock {get_global_mutex()};
       return mag_tensor_numel(*self);
    }, "Total number of elements.")
    .def_prop_ro("numbytes", [](const tensor_wrapper &self) -> size_t {
      std::lock_guard lock {get_global_mutex()};
       return mag_tensor_numbytes(*self);
    }, "Size in bytes of the tensor data.")
    .def_prop_ro("dtype", [](const tensor_wrapper &self) -> dtype_wrapper {
      std::lock_guard lock {get_global_mutex()};
      return dtype_wrapper{ mag_tensor_type(*self) };
    }, "Data type of the tensor (e.g. float32, int64).")
    .def_prop_ro("is_transposed", [](const tensor_wrapper &self) -> bool {
      std::lock_guard lock {get_global_mutex()};
       return mag_tensor_is_transposed(*self);
    }, "True if this tensor is a transpose of another.")
    .def_prop_ro("is_permuted", [](const tensor_wrapper &self) -> bool {
      std::lock_guard lock {get_global_mutex()};
       return mag_tensor_is_permuted(*self);
    }, "True if dimensions have been permuted.")
    .def_prop_ro("is_view", [](const tensor_wrapper &self) -> bool {
      std::lock_guard lock {get_global_mutex()};
       return mag_tensor_is_view(*self);
    }, "True if this tensor shares storage with another.")
    .def_prop_ro("is_contiguous", [](const tensor_wrapper &self) -> bool {
      std::lock_guard lock {get_global_mutex()};
       return mag_tensor_is_contiguous(*self);
    }, "True if elements are stored in contiguous memory order.")
    .def_prop_ro("shape",
    [](const tensor_wrapper &self) -> nb::tuple {
      std::lock_guard lock {get_global_mutex()};
      return tuple_from_i64_span(mag_tensor_shape_ptr(*self), mag_tensor_rank(*self));
    }, "Tuple of dimension sizes.")
    .def_prop_ro("strides",
    [](const tensor_wrapper &self) -> nb::tuple {
      std::lock_guard lock {get_global_mutex()};
      return tuple_from_i64_span(mag_tensor_strides_ptr(*self), mag_tensor_rank(*self));
    }, "Tuple of strides (in elements) per dimension.")
    .def_prop_ro("device", [](const tensor_wrapper &self) -> std::string {
      std::lock_guard lock {get_global_mutex()};
      auto id = mag_tensor_device_id(*self);
      char fmt[32] = {0};
      mag_device_id_to_str(id, &fmt);
      return fmt;
    }, "Device where the tensor is stored (e.g. 'cpu', 'cuda:0').")
    .def_prop_ro("data_ptr", [](const tensor_wrapper &self) -> uintptr_t {
      std::lock_guard lock {get_global_mutex()};
      return mag_tensor_data_ptr(*self);
    }, "Raw pointer to the first element (read-only).")
    .def_prop_ro("data_ptr_mut", [](const tensor_wrapper &self) -> uintptr_t {
      std::lock_guard lock {get_global_mutex()};
      return mag_tensor_data_ptr_mut(*self);
    }, "Raw pointer to the first element (mutable).")
    .def_prop_ro("data_storage_ptr", [](const tensor_wrapper &self) -> uintptr_t {
      std::lock_guard lock {get_global_mutex()};
       return mag_tensor_data_storage_ptr(*self);
    }, "Pointer to the underlying storage block.")
    .def_prop_ro("data_storage_ptr_mut", [](const tensor_wrapper &self) -> uintptr_t {
      std::lock_guard lock {get_global_mutex()};
       return mag_tensor_data_storage_ptr_mut(*self);
    }, "Mutable pointer to the underlying storage block.")
    .def("can_broadcast", [](const tensor_wrapper &self, const tensor_wrapper &rhs) -> bool {
      std::lock_guard lock {get_global_mutex()};
       return mag_tensor_can_broadcast(*self, *rhs);
    }, "rhs"_a, "Return True if this tensor can broadcast with rhs.")
    .def_prop_rw("requires_grad", [](const tensor_wrapper &self) -> bool {
      std::lock_guard lock {get_global_mutex()};
      return mag_tensor_requires_grad(*self);
    }, [](const tensor_wrapper &self, bool req) {
      std::lock_guard lock {get_global_mutex()};
      mag_error_t err {};
      throw_if_error(mag_tensor_set_requires_grad(&err, *self, req), err);
    }, "If True, gradients are recorded for autodiff.")
   .def_prop_rw("grad", [](const tensor_wrapper &self) -> nb::object {
        std::lock_guard lock {get_global_mutex()};
        mag_tensor_t *grad = mag_tensor_grad(*self);
        if (!grad) return nb::none();
        return nb::cast(tensor_wrapper {grad});
      }, [](tensor_wrapper &self, nb::handle value) -> void {
        std::lock_guard lock {get_global_mutex()};
        mag_error_t err {};
        if (value.is_none()) {
          throw_if_error(mag_tensor_set_grad(&err, *self, nullptr), err);
          return;
        }
        auto grad = nb::cast<tensor_wrapper>(value);
        throw_if_error(mag_tensor_set_grad(&err, *self, *grad), err);
      },
      "Gradient accumulated for this tensor."
    )
    .def("backward", [](const tensor_wrapper &self) -> void {
      std::lock_guard lock {get_global_mutex()};
      mag_error_t err {};
      throw_if_error(mag_tensor_backward(&err, *self), err);
    }, "Compute gradients for all tensors that contributed to this one.")
    .def("zero_grad", [](const tensor_wrapper &self) -> void {
      std::lock_guard lock {get_global_mutex()};
      mag_error_t err {};
      throw_if_error(mag_tensor_zero_grad(&err, *self), err);
    }, "Set stored gradient to zero.")
    .def("_replace", [](tensor_wrapper *self, const tensor_wrapper &other) -> void {
      std::lock_guard lock {get_global_mutex()};
      *self = other;
    }, "other"_a, "Replace this tensor's storage with another.")
    .def("item", [](const tensor_wrapper &self) -> nb::object {
      std::lock_guard lock {get_global_mutex()};
      if (mag_tensor_numel(*self) != 1)
        throw nb::value_error("Tensor must have exactly one element to retrieve an item");
      mag_scalar_t s {};
      mag_error_t err {};
      throw_if_error(mag_tensor_item(&err, *self, &s), err);
      if (mag_scalar_is_f64(s)) return nb::float_(mag_scalar_as_f64(s));
      if (mag_scalar_is_i64(s)) return nb::int_(mag_scalar_as_i64(s));
      if (mag_scalar_is_u64(s)) {
        uint64_t v = mag_scalar_as_u64(s);
        if (mag_tensor_type(*self) == MAG_DTYPE_BOOLEAN)
          return nb::bool_{v != 0};
        return nb::int_{v};
      }
      throw nb::type_error("Unsupported scalar type for item()");
    }, "Return the value of a single-element tensor as a Python scalar.")
    .def("detach", [](const tensor_wrapper &self) -> tensor_wrapper {
      std::lock_guard lock {get_global_mutex()};
      return tensor_wrapper{mag_tensor_detach(*self)};
    }, "Return a new tensor detached from the autodiff graph.")
    .def("tolist", [](const tensor_wrapper &self) -> nb::object {
      std::lock_guard lock {get_global_mutex()};
      if (!mag_tensor_numel(*self)) return nb::list();
      mag_error_t err {};
      mag_tensor_t *host = nullptr;
      throw_if_error(mag_transfer(&err, &host, *self, mag_device(CPU, 0)), err);
      on_scope_exit defer_host {[host] { mag_tensor_decref(host); }};
      mag_tensor_t *tensor = host;
      enum class CastedTypeFamily { F64, I64, U64, Bool };
      CastedTypeFamily casted_type {};
      mag_tensor_t *contig = nullptr;
      {
        mag_tensor_t *casted = nullptr;
        if (mag_tensor_is_floating_point_typed(tensor)) {
          throw_if_error(mag_cast(&err, &casted, tensor, MAG_DTYPE_FLOAT32), err);
          casted_type = CastedTypeFamily::F64;
        } else if (mag_tensor_is_signed_integer_typed(tensor)) {
          throw_if_error(mag_cast(&err, &casted, tensor, MAG_DTYPE_INT64), err);
          casted_type = CastedTypeFamily::I64;
        } else if (mag_tensor_is_unsigned_integer_typed(tensor)) {
          throw_if_error(mag_cast(&err, &casted, tensor, MAG_DTYPE_UINT64), err);
          casted_type = CastedTypeFamily::U64;
        } else if (mag_tensor_type(tensor) == MAG_DTYPE_BOOLEAN) {
          throw_if_error(mag_cast(&err, &casted, tensor, MAG_DTYPE_UINT8), err);
          casted_type = CastedTypeFamily::Bool;
        } else {
          throw nb::type_error("Unsupported dtype for tolist()");
        }
        on_scope_exit defer_decref {[casted] { mag_tensor_decref(casted); }};
        throw_if_error(mag_contiguous(&err, &contig, casted), err);
      }
      on_scope_exit defer_decref2 {[contig] { mag_tensor_decref(contig); }};
      const auto *ptr = reinterpret_cast<const void *>(mag_tensor_data_ptr(contig));
      int64_t rank = mag_tensor_rank(contig);
      if (rank == 0) {
        switch (casted_type) {
          case CastedTypeFamily::F64: return nb::float_{*static_cast<const float *>(ptr)};
          case CastedTypeFamily::I64: return nb::int_{*static_cast<const int64_t *>(ptr)};
          case CastedTypeFamily::U64: return nb::int_{*static_cast<const uint64_t *>(ptr)};
          case CastedTypeFamily::Bool: return nb::bool_{static_cast<bool>(*static_cast<const uint8_t *>(ptr))};
          default: throw nb::value_error("Unsupported dtype for tolist()");
        }
      }
      const int64_t *shape = mag_tensor_shape_ptr(contig);
      std::vector<int64_t> strides(rank);
      strides[rank-1] = 1;
      for (int64_t i=rank-2; i >= 0; --i) // RowMaj strides
        strides[i] = strides[i+1]*shape[i+1];
      nb::object result {};
      switch (casted_type) {
        case CastedTypeFamily::F64: result = build_list_recursive<float, nb::float_>(static_cast<const float *>(ptr), shape, strides.data(), rank, 0, 0); break;
        case CastedTypeFamily::I64: result = build_list_recursive<int64_t, nb::int_>(static_cast<const int64_t *>(ptr), shape, strides.data(), rank, 0, 0); break;
        case CastedTypeFamily::U64: result = build_list_recursive<uint64_t, nb::int_>(static_cast<const uint64_t *>(ptr), shape, strides.data(), rank, 0, 0); break;
        case CastedTypeFamily::Bool: result = build_list_recursive<uint8_t, nb::bool_>(static_cast<const uint8_t *>(ptr), shape, strides.data(), rank, 0, 0); break;
        default: throw nb::value_error("Unsupported dtype for tolist()");
      }
      return result;
    }, "Convert tensor to a nested Python list (copies through host if needed).")
    .def("save_image", [](const tensor_wrapper &self, const std::string &file_name) -> void {
      std::lock_guard lock {get_global_mutex()};
      mag_error_t err {};
      throw_if_error(mag_save_image(&err, *self, file_name.c_str()), err);
    })
    .def("save_audio", [](const tensor_wrapper &self, const std::string &path, uint32_t sample_rate) {
      std::lock_guard lock {get_global_mutex()};
      mag_error_t err {};
      throw_if_error(mag_save_audio(&err, *self, path.c_str(), sample_rate), err);
    })
    .def("numpy", [](const tensor_wrapper &self) {
      return tensor_from_numpy(self);
    }, "Return a NumPy ndarray sharing CPU storage when possible. Non-CPU tensors are copied to the host first.");
  }

  extern void init_tensor_special_methods(nb::class_<tensor_wrapper> &cls);
  extern void init_tensor_class_factories(nb::class_<tensor_wrapper> &cls);
  extern void init_tensor_class_operators(nb::class_<tensor_wrapper> &cls);

  void init_bindings_tensor(nb::module_ &m) {
    auto cls = nb::class_<tensor_wrapper>{m, "Tensor",
      "A multi-dimensional array with for automatic differentiation. "
      "Supports CPU, CUDA and custom backends, dtypes, and in-place or out-of-place ops."
    };
    init_tensor_class_base(cls);
    init_tensor_special_methods(cls);
    init_tensor_class_factories(cls);
    init_tensor_class_operators(cls);
  }
}
