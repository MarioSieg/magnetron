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

namespace mag::bindings {
  [[nodiscard]] static nb::list expand_ellipsis(nb::tuple idxs, int64_t rank) {
    int64_t consuming = 0;
    int64_t ellipsis_occ = 0;
    int64_t ellipsis_pos = -1;
    for (size_t i=0; i < idxs.size(); ++i) {
      const auto &x = idxs[i];
      if (x.ptr() == Py_Ellipsis) {
        ++ellipsis_occ;
        if (ellipsis_pos < 0) ellipsis_pos = static_cast<int64_t>(i);
      } else if (x.is_none()) {
      } else if (nb::isinstance<tensor_wrapper>(x) && mag_tensor_type(*nb::cast<tensor_wrapper>(x)) == MAG_DTYPE_BOOLEAN) {
        consuming += mag_tensor_rank(*nb::cast<tensor_wrapper>(x));
      } else { ++consuming; }
    }
    if (ellipsis_occ > 1)
      throw nb::index_error("Only one Ellipsis (...) is allowed in the index tuple");
    nb::list out {};
    if (ellipsis_occ == 1) {
      int64_t to_insert = rank - consuming;
      if (to_insert < 0) {
        std::ostringstream oss;
        oss << "Too many indices for tensor of rank " << rank;
        throw nb::index_error(oss.str().c_str());
      }
      for (int64_t i=0; i < ellipsis_pos; ++i)
        out.append(idxs[static_cast<size_t>(i)]);
      for (int64_t k=0; k < to_insert; ++k)
        out.append(nb::slice {nb::none(), nb::none(), nb::none()});
      for (int64_t i=ellipsis_pos+1; i < static_cast<int64_t>(idxs.size()); ++i)
        out.append(idxs[static_cast<size_t>(i)]);
    } else {
      if (consuming > rank) {
        std::ostringstream oss;
        oss << "Too many indices for tensor of rank " << rank;
        throw nb::index_error(oss.str().c_str());
      }
      if (consuming < rank) {
        for (auto &&idx : idxs)
          out.append(idx);
        for (int64_t k=0; k < rank - consuming; ++k)
          out.append(nb::slice(nb::none(), nb::none(), nb::none()));
      } else {
        for (auto &&idx : idxs)
          out.append(idx);
      }
    }
    return out;
  }

  [[nodiscard]] static bool try_index_all_ints(const tensor_wrapper &self, PyObject *index, tensor_wrapper &out) {
    int64_t rank = mag_tensor_rank(*self);
    if (rank <= 0) return false;
    PyObject **items = &index;
    int64_t count = 1;
    if (PyTuple_CheckExact(index)) {
      count = static_cast<int64_t>(PyTuple_GET_SIZE(index));
      if (count == 0 || count > rank) return false;
      items = &PyTuple_GET_ITEM(index, 0);
    }
    for (Py_ssize_t i=0; i < count; ++i)
      if (!PyLong_CheckExact(items[i])) return false;
    const auto *shape = mag_tensor_shape_ptr(*self);
    const auto *strides = mag_tensor_strides_ptr(*self);
    auto offset = static_cast<int64_t>(mag_tensor_data_offset(*self)/mag_type_trait(mag_tensor_type(*self))->size);
    for (Py_ssize_t ax=0; ax < count; ++ax) {
      int64_t i = PyLong_AsLongLong(items[ax]);
      if (i == -1 && PyErr_Occurred()) throw nb::python_error();
      int64_t dim_size = shape[ax];
      if (i < 0) i += dim_size;
      if (i < 0 || i >= dim_size) {
        std::ostringstream oss;
        oss << "Index " << i << " out of bounds for axis " << ax << " (size " << dim_size << ")";
        throw nb::index_error(oss.str().c_str());
      }
      offset += i*strides[ax];
    }
    int64_t nrank = rank-count;
    int64_t nshape[MAG_MAX_DIMS];
    int64_t nstrides[MAG_MAX_DIMS];
    for (int64_t dim=0; dim < nrank; ++dim) {
      nshape[dim] = shape[count+dim];
      nstrides[dim] = strides[count+dim];
    }
    mag_tensor_t *res = nullptr;
    mag_error_t err {};
    throw_if_error(mag_strided_view(&err, &res, mag_tensor_context(*self), *self, nrank, nrank ? nshape : nullptr, nrank ? nstrides : nullptr, offset), err);
    out = tensor_wrapper{res};
    return true;
  }

  [[nodiscard]] static tensor_wrapper i64_scalar(int64_t v, mag_device_id_t device) {
    return call_op(mag_scalar, get_ctx(), MAG_DTYPE_INT64, mag_scalar_from_int64(v), device);
  }

  [[nodiscard]] static tensor_wrapper slice_axis(const tensor_wrapper &x, int64_t ax, const nb::slice &s) {
    auto [start, stop, step, length] = s.compute(mag_tensor_shape_ptr(*x)[ax]);
    if (step <= 0) {
      std::ostringstream oss;
      oss << "Slice step must be positive, got " << step;
      throw nb::value_error(oss.str().c_str());
    }
    return call_op(mag_view_slice, *x, ax, start, static_cast<int64_t>(length), step);
  }

  [[nodiscard]] static tensor_wrapper int_index_from_sequence(nb::handle h, mag_device_id_t device) {
    std::vector<int64_t> data {};
    for (auto &&v : nb::cast<nb::sequence>(h))
      data.emplace_back(nb::cast<int64_t>(v));
    auto n = static_cast<int64_t>(data.size());
    tensor_wrapper t = call_op(mag_empty, get_ctx(), MAG_DTYPE_INT64, 1, &n, device);
    if (n) {
      mag_error_t err {};
      throw_if_error(mag_copy_raw_(&err, *t, data.data(), data.size()*sizeof(int64_t)), err);
    }
    return t;
  }

  [[nodiscard]] static bool is_advanced_component(nb::handle idx) {
    if (nb::isinstance<tensor_wrapper>(idx)) return true;
    return !idx.is_none() && idx.ptr() != Py_Ellipsis && !nb::isinstance<nb::int_>(idx) && !nb::isinstance<nb::slice>(idx) && nb::isinstance<nb::sequence>(idx);
  }

  struct advanced_index_plan final {
    tensor_wrapper base {};
    std::vector<tensor_wrapper> idx {};
    fixed_dim_vec axes {};
    fixed_dim_vec broadcast_shape {};
    bool adjacent = true;
  };

  [[nodiscard]] static advanced_index_plan plan_advanced_index(const tensor_wrapper &self, const nb::list &indices) {
    advanced_index_plan plan {};
    tensor_wrapper curr = self;
    mag_device_id_t device = mag_tensor_device_id(*self);
    std::vector<int64_t> comp_pos {};
    int64_t ax = 0;
    int64_t pos = 0;
    auto push = [&](int64_t axis, tensor_wrapper t) {
      if (comp_pos.empty() || comp_pos.back() != pos) comp_pos.emplace_back(pos);
      plan.axes.emplace_back(axis);
      plan.idx.emplace_back(std::move(t));
    };
    for (auto &&idx : indices) {
      if (idx.is_none()) {
        curr = call_op(mag_unsqueeze, *curr, ax);
        ++ax;
      } else if (nb::isinstance<nb::slice>(idx)) {
        curr = slice_axis(curr, ax, nb::cast<nb::slice>(idx));
        ++ax;
      } else if (nb::isinstance<nb::int_>(idx)) {
        auto i = nb::cast<int64_t>(idx);
        int64_t dim_size = mag_tensor_shape_ptr(*curr)[ax];
        if (i < 0) i += dim_size;
        if (i < 0 || i >= dim_size) {
          std::ostringstream oss;
          oss << "Index " << i << " out of bounds for axis " << ax << " (size " << dim_size << ")";
          throw nb::index_error(oss.str().c_str());
        }
        curr = call_op(mag_select, *curr, ax, i);
        continue;
      } else if (nb::isinstance<tensor_wrapper>(idx) && mag_tensor_type(*nb::cast<tensor_wrapper>(idx)) == MAG_DTYPE_BOOLEAN) {
        auto mask = nb::cast<tensor_wrapper>(idx);
        if (mag_tensor_rank(*mask) == 0) {
          curr = call_op(mag_unsqueeze, *curr, ax);
          int64_t one = 1;
          mask = call_op(mag_reshape, *mask, &one, 1);
        }
        int64_t k = mag_tensor_rank(*mask);
        int64_t rank = mag_tensor_rank(*curr);
        if (ax+k > rank) {
          std::ostringstream oss;
          oss << "Boolean index of rank " << k << " at axis " << ax << " exceeds tensor rank " << rank;
          throw nb::index_error(oss.str().c_str());
        }
        for (int64_t j=0; j < k; ++j) {
          int64_t ms = mag_tensor_shape_ptr(*mask)[j];
          int64_t ts = mag_tensor_shape_ptr(*curr)[ax+j];
          if (ms != ts) {
            std::ostringstream oss;
            oss << "Boolean index shape mismatch at axis " << ax+j << ": mask has " << ms << ", tensor has " << ts;
            throw nb::index_error(oss.str().c_str());
          }
        }
        tensor_wrapper nz = call_op(mag_nonzero, *mask);
        for (int64_t j=0; j < k; ++j)
          push(ax+j, call_op(mag_select, *nz, 1, j));
        ax += k;
      } else if (nb::isinstance<tensor_wrapper>(idx)) {
        auto t = nb::cast<tensor_wrapper>(idx);
        if (!mag_tensor_is_integer_typed(*t))
          throw nb::index_error("tensors used as indices must be integer or boolean");
        if (mag_tensor_type(*t) != MAG_DTYPE_INT64)
          t = call_op(mag_cast, *t, MAG_DTYPE_INT64);
        push(ax, std::move(t));
        ++ax;
      } else if (nb::isinstance<nb::sequence>(idx)) {
        push(ax, int_index_from_sequence(idx, device));
        ++ax;
      } else throw nb::type_error("Invalid index component type");
      ++pos;
    }
    for (size_t i=1; i < comp_pos.size(); ++i)
      if (comp_pos[i] != comp_pos[i-1]+1) plan.adjacent = false;
    for (auto &&t : plan.idx) {
      int64_t tr = mag_tensor_rank(*t);
      const int64_t *ts = mag_tensor_shape_ptr(*t);
      fixed_dim_vec out(std::max<size_t>(plan.broadcast_shape.size(), static_cast<size_t>(tr)), 1);
      for (size_t i=0; i < out.size(); ++i) {
        int64_t a = i < plan.broadcast_shape.size() ? plan.broadcast_shape[plan.broadcast_shape.size()-1-i] : 1;
        int64_t b = i < static_cast<size_t>(tr) ? ts[tr-1-static_cast<int64_t>(i)] : 1;
        if (a != b && a != 1 && b != 1)
          throw nb::index_error("shape mismatch: indexing tensors could not be broadcast together");
        out[out.size()-1-i] = a == 1 ? b : a;
      }
      plan.broadcast_shape = std::move(out);
    }
    const int64_t *shape = mag_tensor_shape_ptr(*curr);
    auto brank = static_cast<int64_t>(plan.broadcast_shape.size());
    for (size_t j=0; j < plan.idx.size(); ++j) {
      tensor_wrapper &t = plan.idx[j];
      int64_t size = shape[plan.axes[j]];
      if (mag_tensor_numel(*t) > 0) {
        tensor_wrapper lo = i64_scalar(-size, device);
        tensor_wrapper hi = i64_scalar(size, device);
        tensor_wrapper under = call_op(mag_lt, *t, *lo);
        tensor_wrapper over = call_op(mag_ge, *t, *hi);
        tensor_wrapper bad = call_op(mag_or, *under, *over);
        tensor_wrapper any = call_op(mag_any, *bad, nullptr, 0, false);
        mag_scalar_t flag {};
        mag_error_t err {};
        throw_if_error(mag_tensor_item(&err, *any, &flag), err);
        if (mag_scalar_as_int64(flag) || (mag_scalar_is_uint64(flag) && mag_scalar_as_uint64(flag))) {
          std::ostringstream oss;
          oss << "Index out of bounds for axis " << plan.axes[j] << " (size " << size << ")";
          throw nb::index_error(oss.str().c_str());
        }
        tensor_wrapper zero = i64_scalar(0, device);
        tensor_wrapper neg = call_op(mag_lt, *t, *zero);
        tensor_wrapper wrapped = call_op(mag_add, *t, *hi);
        t = call_op(mag_where, *neg, *wrapped, *t);
      }
      t = call_op(mag_expand, *t, brank, plan.broadcast_shape.data());
    }
    plan.base = std::move(curr);
    return plan;
  }

  [[nodiscard]] static fixed_dim_vec advanced_result_perm(const advanced_index_plan &plan, int64_t out_rank) {
    auto nb = static_cast<int64_t>(plan.broadcast_shape.size());
    int64_t first = plan.axes.front();
    fixed_dim_vec perm {};
    if (!plan.adjacent || first == 0) return perm;
    for (int64_t i=0; i < first; ++i) perm.emplace_back(nb+i);
    for (int64_t i=0; i < nb; ++i) perm.emplace_back(i);
    for (int64_t i=nb+first; i < out_rank; ++i) perm.emplace_back(i);
    return perm;
  }

  [[nodiscard]] static tensor_wrapper apply_perm(const tensor_wrapper &x, const fixed_dim_vec &perm) {
    if (perm.empty()) return x;
    return call_op(mag_permute, *x, perm.data(), static_cast<int64_t>(perm.size()));
  }

  [[nodiscard]] static tensor_wrapper advanced_index_get(const advanced_index_plan &plan) {
    const tensor_wrapper &x = plan.base;
    mag_device_id_t device = mag_tensor_device_id(*x);
    int64_t rank = mag_tensor_rank(*x);
    const int64_t *shape = mag_tensor_shape_ptr(*x);
    uint64_t is_adv=0;
    static_assert(MAG_MAX_DIMS <= (sizeof(is_adv)<<3));
    for (int64_t ax : plan.axes) is_adv|=(1ull<<ax);
   fixed_dim_vec perm {};
   fixed_dim_vec rest {};
    for (int64_t ax : plan.axes) perm.emplace_back(ax);
    for (int64_t dim=0; dim < rank; ++dim) {
      if (!(is_adv&(1ull<<dim))) {
        perm.emplace_back(dim);
        rest.emplace_back(shape[dim]);
      }
    }
    tensor_wrapper xp = apply_perm(x, perm);
    tensor_wrapper lin {};
    int64_t merged = 1;
    for (size_t j=plan.idx.size(); j --> 0;) {
      tensor_wrapper term = plan.idx[j];
      if (merged != 1) {
        tensor_wrapper m = i64_scalar(merged, device);
        term = call_op(mag_mul, *term, *m);
      }
      lin = lin ? call_op(mag_add, *lin, *term) : term;
      merged *= shape[plan.axes[j]];
    }
    fixed_dim_vec flat_shape {merged};
    flat_shape.insert(flat_shape.end(), rest.begin(), rest.end());
    tensor_wrapper xc = call_op(mag_contiguous, *xp);
    tensor_wrapper xf = call_op(mag_reshape, *xc, flat_shape.data(), static_cast<int64_t>(flat_shape.size()));
    auto nb = static_cast<int64_t>(plan.broadcast_shape.size());
    int64_t n = 1;
    for (int64_t v : plan.broadcast_shape) n *= v;
    fixed_dim_vec out_shape = plan.broadcast_shape;
    out_shape.insert(out_shape.end(), rest.begin(), rest.end());
    tensor_wrapper lin_flat = call_op(mag_reshape, *lin, &n, 1);
    tensor_wrapper picked = call_op(mag_embedding, *xf, *lin_flat);
    tensor_wrapper res = call_op(mag_reshape, *picked, out_shape.empty() ? nullptr : out_shape.data(), static_cast<int64_t>(out_shape.size()));
    return apply_perm(res, advanced_result_perm(plan, nb + static_cast<int64_t>(rest.size())));
  }

  static void advanced_index_set(const tensor_wrapper &self, const advanced_index_plan &plan, nb::handle value) {
    const tensor_wrapper &x = plan.base;
    mag_device_id_t device = mag_tensor_device_id(*x);
    mag_dtype_t dt = mag_tensor_type(*self);
    int64_t rank = mag_tensor_rank(*x);
    const int64_t *shape = mag_tensor_shape_ptr(*x);
    const int64_t *strides = mag_tensor_strides_ptr(*x);
    std::array<bool, MAG_MAX_DIMS> is_adv {};
    for (int64_t a : plan.axes) is_adv[static_cast<size_t>(a)] = true;
   fixed_dim_vec rest_axes {};
    for (int64_t d=0; d < rank; ++d) if (!is_adv[static_cast<size_t>(d)]) rest_axes.emplace_back(d);
    auto nb = static_cast<int64_t>(plan.broadcast_shape.size());
    auto nr = static_cast<int64_t>(rest_axes.size());
    int64_t out_rank = nb + nr;
    fixed_dim_vec out_shape = plan.broadcast_shape;
    for (int64_t dim : rest_axes) out_shape.emplace_back(shape[dim]);
    auto reshape_to = [&](const tensor_wrapper &t, const fixed_dim_vec &dims) {
      return call_op(mag_reshape, *t, dims.empty() ? nullptr : dims.data(), static_cast<int64_t>(dims.size()));
    };
    auto add = [&](const tensor_wrapper &a, const tensor_wrapper &b) { return call_op(mag_add, *a, *b); };
    auto mul = [&](const tensor_wrapper &a, int64_t v) {
      tensor_wrapper s = i64_scalar(v, device);
      return call_op(mag_mul, *a, *s);
    };
    auto offset = static_cast<int64_t>(mag_tensor_data_offset(*x)/mag_type_trait(dt)->size);
    tensor_wrapper off = i64_scalar(offset, device);
    fixed_dim_vec broadcast_dims = plan.broadcast_shape;
    for (int64_t i=0; i < nr; ++i) broadcast_dims.emplace_back(1);
    for (size_t j=0; j < plan.idx.size(); ++j)
      off = add(off, reshape_to(mul(plan.idx[j], strides[plan.axes[j]]), broadcast_dims));
    for (int64_t i=0; i < nr; ++i) {
      int64_t d = rest_axes[static_cast<size_t>(i)];
      tensor_wrapper r = call_op(mag_arange, get_ctx(), MAG_DTYPE_INT64, mag_scalar_from_int64(0), mag_scalar_from_int64(shape[d]), mag_scalar_from_int64(1), device);
      fixed_dim_vec ret_dims(static_cast<size_t>(out_rank), 1);
      ret_dims[static_cast<size_t>(nb+i)] = shape[d];
      off = add(off, reshape_to(mul(r, strides[d]), ret_dims));
    }
    off = call_op(mag_expand, *off, out_rank, out_shape.data());
    off = apply_perm(off, advanced_result_perm(plan, out_rank));
    int64_t res_rank = mag_tensor_rank(*off);
    fixed_dim_vec res_shape(mag_tensor_shape_ptr(*off), mag_tensor_shape_ptr(*off)+res_rank);
    int64_t n = 1;
    for (int64_t v : res_shape) n *= v;
    if (!n) return;
    tensor_wrapper src {};
    if (nb::isinstance<tensor_wrapper>(value)) {
      src = nb::cast<tensor_wrapper>(value);
      if (mag_tensor_type(*src) != dt)
        src = call_op(mag_cast, *src, dt);
    } else if (nb::isinstance<nb::bool_>(value) || nb::isinstance<nb::int_>(value) || nb::isinstance<nb::float_>(value)) {
      mag_scalar_t sv = scalar_from_py_number(value);
      src = call_op(mag_scalar, get_ctx(), dt, sv, device);
    } else throw nb::type_error("__setitem__ value must be Tensor, int, float, or bool");
    src = call_op(mag_expand, *src, res_rank, res_shape.data());
    fixed_dim_vec flat {n};
    tensor_wrapper off_flat = reshape_to(call_op(mag_contiguous, *off), flat);
    tensor_wrapper src_flat = reshape_to(call_op(mag_contiguous, *src), flat);
    auto storage_numel = static_cast<int64_t>(mag_tensor_storage_numbytes(*self)/mag_type_trait(dt)->size);
    int64_t one = 1;
    tensor_wrapper storage = call_op(mag_strided_view, mag_tensor_context(*self), *self, 1, &storage_numel, &one, 0);
    mag_error_t err {};
    throw_if_error(mag_scatter_(&err, *storage, 0, *off_flat, *src_flat), err);
  }

  [[nodiscard]] static tensor_wrapper tensor_index_impl(const tensor_wrapper &self, const nb::object &index) {
      if (tensor_wrapper fast; try_index_all_ints(self, index.ptr(), fast)) return fast;
      nb::tuple idxs_in = nb::isinstance<nb::tuple>(index) ? nb::cast<nb::tuple>(index) : nb::make_tuple(index);
      tensor_wrapper curr = self;
      int64_t rank0 = mag_tensor_rank(*curr);
      nb::list indices = expand_ellipsis(idxs_in, rank0);
      bool advanced = false;
      for (auto &&idx : indices) advanced |= is_advanced_component(idx);
      if (advanced) return advanced_index_get(plan_advanced_index(self, indices));
      int64_t ax = 0;
      for (auto &&idx : indices) {
          if (idx.is_none()) {
              curr = call_op(mag_unsqueeze, *curr, ax);
              ++ax;
          } else if (nb::isinstance<nb::int_>(idx)) {
              auto i = nb::cast<int64_t>(idx);
              int64_t dim_size = mag_tensor_shape_ptr(*curr)[ax];
              if (i < 0) i += dim_size;
              if (i < 0 || i >= dim_size) {
                  std::ostringstream oss;
                  oss << "Index " << i << " out of bounds for axis " << ax << " (size " << dim_size << ")";
                  throw nb::index_error(oss.str().c_str());
              }
              curr = call_op(mag_select, *curr, ax, i);
          } else if (nb::isinstance<nb::slice>(idx)) {
              curr = slice_axis(curr, ax, nb::cast<nb::slice>(idx));
              ++ax;
          } else throw nb::type_error("Invalid index component type");
      }
      return curr;
  }

  [[nodiscard]] static bool index_is_advanced(const tensor_wrapper &self, const nb::object &index) {
    if (nb::isinstance<nb::tuple>(index)) {
      for (auto &&idx : nb::cast<nb::tuple>(index))
        if (is_advanced_component(idx)) return true;
      return false;
    }
    (void)self;
    return is_advanced_component(index);
  }

  void init_tensor_special_methods(nb::class_<tensor_wrapper> &cls) {
    cls
     .def("__iter__", [](const tensor_wrapper &self) -> nb::object {
       if (mag_tensor_rank(*self) == 0)
         throw nb::type_error("iteration over a 0-d tensor");
       int64_t dim0 = *mag_tensor_shape_ptr(*self);
       nb::list items {};
       for (int64_t i=0; i < dim0; ++i)
         items.append(call_op(mag_select, *self, 0, i));
       return nb::steal(PyObject_GetIter(items.ptr()));
     }, "Iterate over the first dimension, yielding views.")
     .def("__len__", [](const tensor_wrapper &self) -> int64_t {
       if (mag_tensor_rank(*self) == 0)
         throw nb::type_error("len() of a 0-d tensor");
       return *mag_tensor_shape_ptr(*self);
     }, "Length of the first dimension.")
     .def("__str__", [](const tensor_wrapper &self) -> nb::str {
       const char *cstr = mag_tensor_to_string(*self, 3, 3, 1000);
       if (!cstr) throw std::runtime_error("Failed to convert tensor to string");
       on_scope_exit defer_free {[cstr] { mag_tensor_to_string_free_data(cstr); }};
       auto str = nb::str{cstr};
       return str;
     }, "Short string representation.")
     .def("__repr__", [](const tensor_wrapper &self) -> nb::str {
       const char *cstr = mag_tensor_to_string(*self, 3, 3, 1000);
       if (!cstr) throw std::runtime_error("Failed to convert tensor to string");
       on_scope_exit defer_free {[cstr] { mag_tensor_to_string_free_data(cstr); }};
       auto str = nb::str {cstr};
       return str;
     }, "Full repr (shape, dtype, values).")
    .def("__int__", [](const tensor_wrapper &self) -> int64_t {
      if (mag_tensor_numel(*self) != 1)
        throw nb::value_error("Only one-element tensors can be converted to Python scalars");
      mag_scalar_t val {};
      mag_error_t err {};
      throw_if_error(mag_tensor_item(&err, *self, &val), err);
      if (mag_scalar_is_float64(val)) return static_cast<int64_t>(mag_scalar_as_float64(val));
      if (mag_scalar_is_int64(val)) return mag_scalar_as_int64(val);
      if (mag_scalar_is_uint64(val)) return static_cast<int64_t>(mag_scalar_as_uint64(val));
      throw nb::type_error("Unsupported scalar type for __int__()");
    })
    .def("__float__", [](const tensor_wrapper &self) -> double {
      if (mag_tensor_numel(*self) != 1)
        throw nb::value_error("Only one-element tensors can be converted to Python scalars");
      mag_scalar_t val {};
      mag_error_t err {};
      throw_if_error(mag_tensor_item(&err, *self, &val), err);
      if (mag_scalar_is_float64(val)) return mag_scalar_as_float64(val);
      if (mag_scalar_is_int64(val)) return static_cast<double>(mag_scalar_as_int64(val));
      if (mag_scalar_is_uint64(val)) return static_cast<double>(mag_scalar_as_uint64(val));
      throw nb::type_error("Unsupported scalar type for __float__()");
    })
    .def("__index__", [](const tensor_wrapper &self) -> int64_t {
      if (mag_tensor_numel(*self) != 1)
        throw nb::value_error("Only one-element tensors can be converted to an index");
      mag_scalar_t val {};
      mag_error_t err {};
      throw_if_error(mag_tensor_item(&err, *self, &val), err);
      if (mag_scalar_is_int64(val)) return mag_scalar_as_int64(val);
      if (mag_scalar_is_uint64(val)) return static_cast<int64_t>(mag_scalar_as_uint64(val));
      throw nb::type_error("Only integer tensors can be converted to an index");
    })
    .def("__bool__", [](const tensor_wrapper &self) -> bool {
      if (mag_tensor_numel(*self) != 1)
        throw nb::value_error("Tensor with >1 element has ambiguous truth value; use .any() or .all()");
      mag_scalar_t s {};
      mag_error_t err {};
      throw_if_error(mag_tensor_item(&err, *self, &s), err);
      if (mag_scalar_is_float64(s)) return mag_scalar_as_float64(s) != 0.0;
      if (mag_scalar_is_int64(s)) return mag_scalar_as_int64(s) != 0;
      if (mag_scalar_is_uint64(s)) return mag_scalar_as_uint64(s) != 0;
      throw nb::type_error("Unsupported scalar type for __bool__()");
    }, "True if single element is non-zero (only for 0-dim or 1-element tensors).")
    .def("__getitem__", [](const tensor_wrapper &self, nb::object index) -> tensor_wrapper {
       return tensor_index_impl(self, index);
    }, "Index with int, slice, ellipsis, or boolean/int index tensor. Supports NumPy-style indexing.")
    .def("__setitem__", [](tensor_wrapper &self, nb::object index, nb::object value) {
      if (index_is_advanced(self, index)) {
        nb::tuple idxs_in = nb::isinstance<nb::tuple>(index) ? nb::cast<nb::tuple>(index) : nb::make_tuple(index);
        advanced_index_set(self, plan_advanced_index(self, expand_ellipsis(idxs_in, mag_tensor_rank(*self))), value);
        return;
      }
      tensor_wrapper dst = tensor_index_impl(self, index);
      mag_error_t err {};
      if (nb::isinstance<tensor_wrapper>(value)) {
          auto src = nb::cast<tensor_wrapper>(value);
          throw_if_error(mag_copy_(&err, *dst, *src), err);
          return;
      }
      if (nb::isinstance<nb::bool_>(value) || nb::isinstance<nb::int_>(value) || nb::isinstance<nb::float_>(value)) {
          mag_scalar_t scalar = scalar_from_py_number(value);
          throw_if_error(mag_fill_(&err, *dst, scalar), err);
          return;
      }
      throw nb::type_error("__setitem__ value must be Tensor, int, float, or bool");
    }, "Set item by index. Value can be a Tensor (copied) or a scalar (broadcast).");
  }
}
