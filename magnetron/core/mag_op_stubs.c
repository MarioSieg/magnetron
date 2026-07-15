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

#include "mag_autodiff.h"
#include "mag_context.h"
#include "mag_reduce_plan.h"
#include "mag_einsum.h"
#include "mag_u128.h"
#include "mag_alloc.h"
#include "mag_op_dispatch.h"
#include "mag_op_helpers.h"

/* Create a new tensor. The must be created on the same thread as the context. */
mag_status_t mag_empty(mag_error_t *err, mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device) {
  return mag_tensor_init(err, out, ctx, NULL, type, rank, shape, device);
}

extern mag_tensor_t *mag_tensor_init_header(
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t rank,
  int64_t numel,
  mag_device_t *device,
  mag_storage_buffer_t *storage
);

mag_status_t mag_strided_view(mag_error_t *err, mag_tensor_t **out, mag_context_t *ctx, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset) {
  *out = NULL;
  if (mag_unlikely(mag_thread_id() != ctx->tr_id))
    return mag_set_error(err, MAG_ERR_THREAD, "strided_view: tensor must be created on the thread that owns the context (expected thread 0x%" PRIx64 ", got 0x%" PRIx64 ").", (uint64_t)ctx->tr_id, (uint64_t)mag_thread_id());
  if (mag_unlikely(!(rank >= 0 && rank <= MAG_MAX_DIMS)))
    return mag_set_error(err, MAG_ERR_RANK, "strided_view: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  if (mag_unlikely(offset < 0))
    return mag_set_error(err, MAG_ERR_INDEX, "strided_view: storage offset must be non-negative, but got %" PRIi64 ".", offset);
  if (mag_unlikely(rank > 0 && !(shape && strides)))
    return mag_set_error(err, MAG_ERR_PARAM, "strided_view: shape and strides must not be NULL when rank > 0.");
  int64_t lo=offset;
  int64_t hi=offset;
  int64_t numel=1;
  for (int64_t i=0; i < rank; ++i) {
    if (mag_unlikely(shape[i] < 0))
      return mag_set_error(err, MAG_ERR_DIM, "strided_view: invalid shape at dim %" PRIi64 " (shape=%" PRIi64 "); dimensions must be >= 0.", i, shape[i]);
    int64_t span;
    if (mag_unlikely(mag_mulov64(shape[i]-1, strides[i], &span)))
      return mag_set_error(err, MAG_ERR_DIM, "strided_view: stride span overflowed at dim %" PRIi64 ".", i);
    if (mag_unlikely(mag_mulov64(shape[i], numel, &numel)))
      return mag_set_error(err, MAG_ERR_DIM, "strided_view: element count overflowed at dim %" PRIi64 " (size %" PRIi64 ").", i, shape[i]);
    if (span >= 0) hi += span;
    else lo += span;
  }
  if (numel > 0) {
    int64_t numel_end = (int64_t)(base->storage->size/mag_type_trait(base->meta.dtype)->size);
    if (mag_unlikely(lo < 0))
      return mag_set_error(err, MAG_ERR_BOUNDS, "strided_view: view underflows base tensor storage (start index %" PRIi64 " < 0).", lo);
    if (mag_unlikely(hi >= numel_end))
      return mag_set_error(err, MAG_ERR_BOUNDS, "strided_view: view exceeds base tensor storage (end index %" PRIi64 " >= storage capacity %" PRIi64 ").", hi, numel_end);
  }
  mag_tensor_t *tensor = mag_tensor_init_header(ctx, base->meta.dtype, rank, numel, base->meta.device, NULL); /* Alloc tensor header. */
  if (mag_unlikely(!tensor))
    return mag_set_error(err, MAG_ERR_OOM, "strided_view: failed to allocate tensor header.");
  for (int i=0; i < MAG_MAX_DIMS; ++i) {
    tensor->meta.coords.shape[i] = i < rank && shape ? shape[i] : 1;
    tensor->meta.coords.strides[i] = i < rank && strides ? strides[i] : 1;
  }
  tensor->storage = base->storage;
  mag_rc_incref(base->storage);
  tensor->meta.storage_offset = offset;
  tensor->version = base->version;
  if (!(base->meta.flags & MAG_TFLAG_IS_VIEW)) {
    tensor->view_meta = mag_view_meta_alloc(base);
    if (mag_unlikely(!tensor->view_meta)) {
      mag_tensor_decref(tensor);
      return mag_set_error(err, MAG_ERR_OOM, "strided_view: failed to allocate view metadata.");
    }
  } else {
    tensor->view_meta = base->view_meta;
    mag_rc_incref(tensor->view_meta);
  }
  tensor->meta.flags = base->meta.flags|MAG_TFLAG_IS_VIEW;
  if (mag_ctx_grad_recorder_is_running(ctx) && (base->meta.flags & MAG_TFLAG_REQUIRES_GRAD)) {
    mag_op_params_t params = {0};
    params.strided.rank = rank;
    params.strided.offset = offset;
    memcpy(params.strided.shape, shape, rank*sizeof(*shape));
    memcpy(params.strided.strides, strides, rank*sizeof(*strides));
    mag_status_t status = mag_dispatch(err, MAG_OP_STRIDED_VIEW, false, &base, 1, &tensor, 1, &params);
    if (mag_unlikely(mag_iserr(status))) { mag_tensor_decref(tensor); return status; }
  }
  *out = tensor;
  return MAG_OK;
}

mag_status_t mag_empty_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like) {
  return mag_empty(err, out_result, like->ctx, like->meta.dtype, like->meta.coords.rank, like->meta.coords.shape, mag_tensor_device_id(like));
}

mag_status_t mag_empty_scalar(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_device_id_t device) {
  return mag_empty(err, out_result, ctx, type, 0, NULL, device);
}

mag_status_t mag_scalar(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t value, mag_device_id_t device) {
  mag_status_t status = mag_empty_scalar(err, out_result, ctx, type, device);
  if (mag_iserr(status)) return status;
  return mag_fill_(err, *out_result, value);
}

mag_status_t mag_full(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t value, mag_device_id_t device) {
  mag_status_t status = mag_empty(err, out_result, ctx, type, rank, shape, device);
  if (mag_iserr(status)) return status;
  return mag_fill_(err, *out_result, value);
}

mag_status_t mag_full_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t value) {
  mag_status_t status = mag_empty_like(err, out_result, like);
  if (mag_iserr(status)) return status;
  return mag_fill_(err, *out_result, value);
}

mag_status_t mag_zeros(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device) {
  return mag_full(err, out_result, ctx, type, rank, shape, mag_scalar_from_uint64(0), device);
}

mag_status_t mag_zeros_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like) {
  return mag_full_like(err, out_result, like, mag_scalar_from_uint64(0));
}

mag_status_t mag_ones(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device) {
  return mag_full(err, out_result, ctx, type, rank, shape, mag_scalar_from_uint64(1), device);
}

mag_status_t mag_ones_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like) {
  return mag_full_like(err, out_result, like, mag_scalar_from_uint64(1));
}

mag_status_t mag_uniform(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t min, mag_scalar_t max, mag_device_id_t device) {
  mag_status_t status = mag_empty(err, out_result, ctx, type, rank, shape, device);
  if (mag_iserr(status)) return status;
  return mag_uniform_(err, *out_result, min, max);
}

mag_status_t mag_uniform_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t min, mag_scalar_t max) {
  mag_status_t status = mag_empty_like(err, out_result, like);
  if (mag_iserr(status)) return status;
  return mag_uniform_(err, *out_result, min, max);
}

mag_status_t mag_normal(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t mean, mag_scalar_t stddev, mag_device_id_t device) {
  mag_status_t status = mag_empty(err, out_result, ctx, type, rank, shape, device);
  if (mag_iserr(status)) return status;
  return mag_normal_(err, *out_result, mean, stddev);
}

mag_status_t mag_normal_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t mean, mag_scalar_t stddev) {
  mag_status_t status = mag_empty_like(err, out_result, like);
  if (mag_iserr(status)) return status;
  return mag_normal_(err, *out_result, mean, stddev);
}

mag_status_t mag_bernoulli(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, int64_t rank, const int64_t *shape, double p, mag_device_id_t device) {
  mag_status_t status = mag_empty(err, out_result, ctx, MAG_DTYPE_BOOLEAN, rank, shape, device);
  if (mag_iserr(status)) return status;
  return mag_bernoulli_(err, *out_result, p);
}

mag_status_t mag_bernoulli_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, double p) {
  mag_status_t status = mag_empty(err, out_result, like->ctx, MAG_DTYPE_BOOLEAN, like->meta.coords.rank, like->meta.coords.shape, mag_tensor_device_id(like));
  if (mag_iserr(status)) return status;
  return mag_bernoulli_(err, *out_result, p);
}

mag_status_t mag_broadcast(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x, int64_t rank, const int64_t *shape) {
  int64_t old_rank = x->meta.coords.rank;
  const int64_t *old_shape = x->meta.coords.shape;
  const int64_t *old_strides = x->meta.coords.strides;
  int64_t new_strides[MAG_MAX_DIMS];
  if (mag_unlikely(rank < old_rank)) {
    return mag_set_error(err, MAG_ERR_RANK, "broadcast: target rank %" PRIi64 " must be >= source rank %" PRIi64 ".", rank, old_rank);
  }
  for (int64_t i=0; i < rank; ++i) {
    int64_t new_ax = rank-1-i;
    int64_t old_ax = old_rank-1-i;
    int64_t new_dim = shape[new_ax];
    if (old_ax < 0) {
      new_strides[new_ax] = 0;
      continue;
    }
    int64_t old_dim = old_shape[old_ax];
    int64_t old_stride = old_strides[old_ax];
    if (mag_unlikely(!(old_dim == new_dim || old_dim == 1))) {
      return mag_set_error(err, MAG_ERR_RANK, "broadcast: cannot broadcast dim of size %" PRIi64 " to %" PRIi64 "; only size-1 dims are broadcastable.", old_dim, new_dim);
    }
    new_strides[new_ax] = old_dim == new_dim ? old_stride : 0;
  }
  return mag_strided_view(
    err,
    out,
    mag_tensor_context(x),
    x,
    rank,
    shape,
    new_strides,
    (int64_t)mag_tensor_data_offset(x)
  );
}

mag_status_t mag_expand(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x, int64_t rank, const int64_t *shape) {
  int64_t old_rank = x->meta.coords.rank;
  const int64_t *old_shape = x->meta.coords.shape;
  if (mag_unlikely(rank < old_rank)) {
    return mag_set_error(
      err, MAG_ERR_RANK,
      "expand: target rank %" PRIi64 " must be >= source rank %" PRIi64 ".",
      rank, old_rank
    );
  }
  int64_t resolved[MAG_MAX_DIMS];
  for (int64_t i=0; i < rank; ++i) {
    int64_t new_ax = rank-1-i;
    int64_t old_ax = old_rank-1-i;
    int64_t dim = shape[new_ax];
    if (dim == -1) {
      if (mag_unlikely(old_ax < 0)) {
        return mag_set_error(
          err, MAG_ERR_PARAM,
          "expand: -1 is not allowed for a newly prepended dimension."
        );
      }
      resolved[new_ax] = old_shape[old_ax];
    } else {
      if (mag_unlikely(dim < 0)) {
        return mag_set_error(
          err, MAG_ERR_PARAM,
          "expand: invalid dimension size %" PRIi64 ".",
          dim
        );
      }
      resolved[new_ax] = dim;
    }
  }
  return mag_broadcast(err, out, x, rank, resolved);
}

mag_status_t mag_arange(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t start, mag_scalar_t end, mag_scalar_t step, mag_device_id_t device) {
  *out_result = NULL;
  if (mag_unlikely(!(mag_scalar_same_type(start, end) && mag_scalar_same_type(start, step))))
      return mag_set_error(err, MAG_ERR_PARAM, "arange: start, end and step must have the same scalar type.");
  if (mag_unlikely(!(mag_dtype_bit(type) & MAG_DTYPE_MASK_NUMERIC)))
      return mag_set_error(err, MAG_ERR_PARAM, "arange: requires a numeric dtype.");
  mag_tensor_t *result;
  int64_t numel = 0;
  bool ok = false;
  bool is_signed = mag_type_category_is_signed_integer(type);
  bool is_unsigned = mag_type_category_is_unsigned_integer(type);
  if (is_signed) ok = mag_arange_numel_i64(mag_scalar_as_int64(start), mag_scalar_as_int64(end), mag_scalar_as_int64(step), &numel);
  else if (is_unsigned) ok = mag_arange_numel_u64(mag_scalar_as_uint64(start), mag_scalar_as_uint64(end), mag_scalar_as_uint64(step), &numel);
  else ok = mag_arange_numel_float(mag_scalar_as_float64(start), mag_scalar_as_float64(end), mag_scalar_as_float64(step), &numel);
  if (mag_unlikely(!ok) || numel <= 0)
      return mag_set_error(err, MAG_ERR_PARAM, "arange: invalid start, end or step (produces an empty or invalid range).");
  mag_status_t status = mag_empty(err, &result, ctx, type, 1, &numel, device);
  if (mag_iserr(status)) return status;
  mag_op_params_t params = {
    .arange = {
      .start = start,
      .step = step
    }
  };
  status = mag_check_dtype_and_device_compat(err, MAG_OP_ARANGE, NULL, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_ARANGE, false, NULL, 0, &result, 1, &params);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_eye(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t n, int64_t m, mag_device_id_t device) {
  *out_result = NULL;
  if (mag_unlikely(n < 0))
      return mag_set_error(err, MAG_ERR_PARAM, "eye: n must be >= 0, but got %" PRIi64 ".", n);
  if (mag_unlikely(m < 0))
      return mag_set_error(err, MAG_ERR_PARAM, "eye: m must be >= 0, but got %" PRIi64 ".", m);
  mag_tensor_t *result = NULL;
  mag_status_t status = mag_empty(err, &result, ctx, type, 2, (int64_t[2]){n, m}, device);
  if (mag_iserr(status)) return status;
  if (mag_likely(n > 0 && m > 0)) {
    status = mag_check_dtype_and_device_compat(err, MAG_OP_EYE, NULL, 0);
    if (mag_iserr(status)) return status;
    status = mag_dispatch(err, MAG_OP_EYE, false, NULL, 0, &result, 1, NULL);
    if (mag_iserr(status)) return status;
  }
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_linspace(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t start, mag_scalar_t end, int64_t steps, mag_device_id_t device) {
  *out_result = NULL;
  if (mag_unlikely(steps <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "linspace: steps must be > 0, but got %" PRIi64 ".", steps);
  if (mag_unlikely(!mag_scalar_same_type(start, end)))
      return mag_set_error(err, MAG_ERR_PARAM, "linspace: start and end must have the same scalar type.");
  if (mag_unlikely(!(mag_dtype_bit(type) & MAG_DTYPE_MASK_NUMERIC)))
      return mag_set_error(err, MAG_ERR_PARAM, "linspace: requires a numeric dtype.");
  if (steps == 1) return mag_full(err, out_result, ctx, type, 1, &steps, start, device);
  mag_status_t status = MAG_OK;
  mag_tensor_t *idx = NULL;
  mag_tensor_t *scale = NULL;
  mag_tensor_t *start_t = NULL;
  mag_tensor_t *tmp = NULL;
  mag_tensor_t *result = NULL;
  status = mag_arange(err, &idx, ctx, type, mag_scalar_from_int64(0), mag_scalar_from_int64(steps), mag_scalar_from_int64(1), device);
  if (mag_iserr(status)) goto cleanup;
  status = mag_full(err, &scale, ctx, type, 1, &steps, mag_scalar_from_float64((mag_scalar_as_float64(end) - mag_scalar_as_float64(start))/(double)(steps - 1)), device);
  if (mag_iserr(status)) goto cleanup;
  status = mag_full(err, &start_t, ctx, type, 1, &steps, start, device);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &tmp, idx, scale);
  if (mag_iserr(status)) goto cleanup;
  status = mag_add(err, &result, tmp, start_t);
  if (mag_iserr(status)) goto cleanup;
  *out_result = result;
  result = NULL;
cleanup:
  if (result) mag_tensor_decref(result);
  if (tmp) mag_tensor_decref(tmp);
  if (start_t) mag_tensor_decref(start_t);
  if (scale) mag_tensor_decref(scale);
  if (idx) mag_tensor_decref(idx);
  return status;
}

mag_status_t mag_meshgrid(mag_error_t *err, mag_tensor_t **out_results, mag_tensor_t **tensors, size_t count) {
  if (mag_unlikely(!(out_results != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "meshgrid: out_results must not be NULL.");
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "meshgrid: tensors must not be NULL.");
  if (mag_unlikely(count <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "meshgrid: expected at least one tensor.");
  if (mag_unlikely(!(count <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_RANK, "meshgrid: tensor count %zu exceeds maximum rank %d.", count, MAG_MAX_DIMS);
  for (size_t i=0; i < count; ++i) {
    out_results[i] = NULL;
    if (mag_unlikely(!(tensors[i] != NULL)))
        return mag_set_error(err, MAG_ERR_PARAM, "meshgrid: tensors[%zu] must not be NULL.", i);
    if (mag_unlikely(!(tensors[i]->meta.coords.rank == 1)))
        return mag_set_error(err, MAG_ERR_RANK, "meshgrid: tensors[%zu] must be 1-D, but got rank %" PRIi64 ".", i, tensors[i]->meta.coords.rank);
  }
  int64_t full_shape[MAG_MAX_DIMS];
  for (size_t i=0; i < count; ++i)
    full_shape[i] = tensors[i]->meta.coords.shape[0];
  size_t filled = 0;
  mag_status_t status = MAG_OK;
  for (size_t i=0; i < count; ++i) {
    int64_t view_shape[MAG_MAX_DIMS];
    for (size_t dim=0; dim < count; ++dim)
      view_shape[dim] = 1;
    view_shape[i] = tensors[i]->meta.coords.shape[0];
    mag_tensor_t *view = NULL;
    mag_tensor_t *expanded = NULL;
    status = mag_view(err, &view, tensors[i], view_shape, (int64_t)count);
    if (mag_iserr(status)) {
      for (size_t j=0; j < filled; ++j) { mag_tensor_decref(out_results[j]); out_results[j] = NULL; }
      return status;
    }
    status = mag_expand(err, &expanded, view, (int64_t)count, full_shape);
    if (mag_iserr(status)) {
      mag_tensor_decref(view);
      for (size_t j=0; j < filled; ++j) { mag_tensor_decref(out_results[j]); out_results[j] = NULL; }
      return status;
    }
    mag_tensor_decref(view);
    out_results[i] = expanded;
    ++filled;
  }
  return MAG_OK;
}

mag_status_t mag_rand_perm(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t n, mag_device_id_t device) {
  *out_result = NULL;
  if (mag_unlikely(!(mag_dtype_bit(type) & MAG_DTYPE_MASK_INTEGER)))
      return mag_set_error(err, MAG_ERR_PARAM, "rand_perm: requires an integer dtype.");
  mag_tensor_t *result;
  mag_status_t status = mag_empty(err, &result, ctx, type, 1, &n, device);
  if (mag_iserr(status)) return status;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_RAND_PERM, NULL, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_RAND_PERM, false, NULL, 0, &result, 1, NULL);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_clone(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x) {
  *out_result = NULL;
  mag_tensor_t *result;
  mag_status_t status = mag_empty_like(err, &result, x);
  if (mag_iserr(status)) return status;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_CLONE, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_CLONE, false, &x, 1, &result, 1, NULL);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_cast(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_dtype_t dst_type) {
  if (x->meta.dtype == dst_type) return mag_clone(err, out_result, x); /* If dtypes match, we just clone */
  *out_result = NULL;
  mag_tensor_t *result;
  mag_status_t status = mag_empty(err, &result, x->ctx, dst_type, x->meta.coords.rank, x->meta.coords.shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_CAST, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_CAST, false, &x, 1, &result, 1, NULL);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_transfer(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_device_id_t device) {
  *out_result = NULL;
  mag_device_id_t src_id = mag_tensor_device_id(x);
  if (src_id.type == device.type && src_id.device_ordinal == device.device_ordinal) { /* If already on same device, bump refcount and do nothing */
    mag_rc_incref(x);
    *out_result = x;
    return MAG_OK;
  }
  mag_status_t status = MAG_OK;
  mag_tensor_t *xc = NULL;
  mag_tensor_t *out = NULL;
  status = mag_contiguous(err, &xc, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_empty(err, &out, x->ctx, xc->meta.dtype, xc->meta.coords.rank, xc->meta.coords.shape, device);
  if (mag_iserr(status)) goto cleanup;
  {
    mag_device_t *src_dvc = xc->meta.device;
    mag_device_t *dst_dvc = out->meta.device;
    bool src_hv = xc->storage->flags&MAG_STORAGE_FLAG_HOST_VISIBLE;
    bool dst_hv = out->storage->flags&MAG_STORAGE_FLAG_HOST_VISIBLE;
    if (src_hv && dst_hv) {
      size_t nb = mag_tensor_numbytes(xc);
      if (mag_unlikely(nb != mag_tensor_numbytes(out))) {
        status = mag_set_error(err, MAG_ERR_PARAM, "transfer: source and destination tensor sizes do not match.");
        goto cleanup;
      }
      if (mag_unlikely(!(mag_tensor_is_contiguous(xc) && mag_tensor_is_contiguous(out)))) {
        status = mag_set_error(err, MAG_ERR_PARAM, "transfer: both tensors must be contiguous.");
        goto cleanup;
      }
      memcpy((void *)mag_tensor_data_ptr_mut(out), (const void *)mag_tensor_data_ptr(xc), nb);
      mag_tensor_decref(xc);
      *out_result = out;
      return MAG_OK;
    }
    mag_device_t *exec = NULL;
    mag_transfer_dir_t dir;
    if (src_hv && !dst_hv) {
      exec = dst_dvc;
      dir = MAG_TRANSFER_DIR_H2D;
    } else if (!src_hv && dst_hv) {
      exec = src_dvc;
      dir = MAG_TRANSFER_DIR_D2H;
    } else {
      exec = dst_dvc;
      dir = MAG_TRANSFER_DIR_D2D;
    }
    if (mag_unlikely(!(exec->transfer != NULL))) {
      status = mag_set_error(err, MAG_ERR_STATE, "transfer: target device does not implement tensor transfer.");
      goto cleanup;
    }
    status = (*exec->transfer)(err, exec, dir, xc, out);
    if (mag_iserr(status))
        goto cleanup;
  }
  mag_tensor_decref(xc);
  *out_result = out;
  return MAG_OK;
cleanup:
  if (out) mag_tensor_decref(out);
  if (xc) mag_tensor_decref(xc);
  return status;
}

mag_status_t mag_view(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  if (mag_unlikely(!(rank >= 0 && rank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_RANK, "view: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  mag_status_t status;
  if (rank == 0) {
    if (mag_unlikely(x->meta.numel != 1))
        return mag_set_error(err, MAG_ERR_PARAM, "view: rank-0 view is only allowed on tensors with a single element, but got %" PRIi64 " elements.", x->meta.numel);
    status = mag_strided_view(err, &result, x->ctx, x, 0, NULL, NULL, x->meta.storage_offset);
    if (mag_iserr(status)) return status;
  } else {
    if (mag_unlikely(!(dims != NULL)))
        return mag_set_error(err, MAG_ERR_PARAM, "view: dims must not be NULL when rank > 0.");
    int64_t oshape[MAG_MAX_DIMS] = {0};
    memcpy(oshape, dims, rank*sizeof(*dims));
    int64_t shape[MAG_MAX_DIMS];
    status = mag_infer_missing_dim(err, &shape, oshape, rank, x->meta.numel);
    if (mag_iserr(status)) return status;
    int64_t strides[MAG_MAX_DIMS];
    if (rank == x->meta.coords.rank && !memcmp(shape, x->meta.coords.shape, rank*sizeof(*shape))) { /* Stride strategy: same shape as base */
      memcpy(strides, x->meta.coords.strides, rank*sizeof(*shape));
    } else if (rank == x->meta.coords.rank+1 && shape[rank-2]*shape[rank-1] == x->meta.coords.shape[x->meta.coords.rank-1]) { /* Stride strategy: last dim only */
      memcpy(strides, x->meta.coords.strides, (rank-2)*sizeof(*strides));
      strides[rank-2] = x->meta.coords.strides[x->meta.coords.rank-1]*shape[rank-1];
      strides[rank-1] = x->meta.coords.strides[x->meta.coords.rank-1];
    } else if (mag_tensor_is_contiguous(x)) { /* Stride strategy: contiguous row-major */
      strides[rank-1] = 1;
      for (int64_t i=rank-2; i >= 0; --i) {
        if (mag_unlikely(mag_mulov64(shape[i+1], strides[i+1], strides+i)))
            return mag_set_error(err, MAG_ERR_DIM, "view: stride computation overflowed at dim %" PRIi64 ".", i);
      }
    } else { /* Stride strategy: solve generic strides */
      status = mag_solve_view_strides(err, &strides, x->meta.coords.shape, x->meta.coords.strides, x->meta.coords.rank, shape, rank);
      if (mag_iserr(status)) return status;
    }
    status = mag_strided_view(err, &result, x->ctx, x, rank, shape, strides, x->meta.storage_offset);
    if (mag_iserr(status)) return status;
  }
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_reshape(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  int64_t shape[MAG_MAX_DIMS];
  mag_status_t status = mag_infer_missing_dim(err, &shape, dims, rank, x->meta.numel);
  if (mag_iserr(status)) return status;
  if (x->meta.coords.rank == rank && !memcmp(x->meta.coords.shape, shape, sizeof(*dims)*rank)) {
    mag_rc_incref(x);
    *out_result = x;
    return MAG_OK;
  }
  if (mag_tensor_is_contiguous(x) || mag_tensor_can_view(x, shape, rank))
    return mag_view(err, out_result, x, shape, rank);
  status = mag_contiguous(err, &result, x);
  if (mag_iserr(status)) return status;
  mag_tensor_t *reshaped = NULL;
  status = mag_view(err, &reshaped, result, shape, rank);
  mag_rc_decref(result);
  if (mag_iserr(status)) return status;
  *out_result = reshaped;
  return MAG_OK;
}

mag_status_t mag_view_slice(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t start, int64_t len, int64_t step) {
  *out_result = NULL;
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(rank <= 0))
      return mag_set_error(err, MAG_ERR_RANK, "slice: cannot slice a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "slice: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  if (mag_unlikely(step <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "slice: step must be > 0, but got %" PRIi64 ".", step);
  int64_t sz = x->meta.coords.shape[dim];
  mag_norm_axis(&start, sz);
  if (mag_unlikely(!(0 <= start && start < sz)))
      return mag_set_error(err, MAG_ERR_PARAM, "slice: start %" PRIi64 " is out of bounds for dim %" PRIi64 " of size %" PRIi64 ".", start, dim, sz);
  if (len < 0) len = (int64_t)mag_uint128_ceildiv((uint64_t)(sz-start), (uint64_t)step);
  if (mag_unlikely(len <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "slice: length must be > 0, but got %" PRIi64 ".", len);
  int64_t last = start + (len - 1)*step;
  if (mag_unlikely(!(0 <= last && last < sz)))
      return mag_set_error(err, MAG_ERR_PARAM, "slice: end index %" PRIi64 " exceeds size %" PRIi64 " on dim %" PRIi64 ".", last, sz, dim);
  int64_t shape[MAG_MAX_DIMS];
  int64_t strides[MAG_MAX_DIMS];
  memcpy(shape, x->meta.coords.shape, rank*sizeof(*shape));
  memcpy(strides, x->meta.coords.strides, rank*sizeof(*strides));
  shape[dim] = len;
  strides[dim] = x->meta.coords.strides[dim] * step;
  int64_t offset = x->meta.storage_offset + start*x->meta.coords.strides[dim];
  return mag_strided_view(err, out_result, x->ctx, x, rank, shape, strides, offset);
}

mag_status_t mag_transpose(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim1, int64_t dim2) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  if (mag_unlikely(x->meta.coords.rank < 2))
      return mag_set_error(err, MAG_ERR_PARAM, "transpose: requires rank >= 2, but got %" PRIi64 ".", x->meta.coords.rank);
  if (mag_unlikely(dim1 == dim2))
      return mag_set_error(err, MAG_ERR_PARAM, "transpose: axes must differ, but got dim1 == dim2 == %" PRIi64 ".", dim1);
  int64_t ra = x->meta.coords.rank;
  int64_t ax0 = dim1;
  int64_t ax1 = dim2;
  mag_norm_axis(&ax0, ra);
  mag_norm_axis(&ax1, ra);
  if (mag_unlikely(!(ax0 >= 0 && ax0 < ra)))
      return mag_set_error(err, MAG_ERR_PARAM, "transpose: axis %" PRIi64 " is out of range for rank %" PRIi64 ".", dim1, ra);
  if (mag_unlikely(!(ax1 >= 0 && ax1 < ra)))
      return mag_set_error(err, MAG_ERR_PARAM, "transpose: axis %" PRIi64 " is out of range for rank %" PRIi64 ".", dim2, ra);
  int64_t shape[MAG_MAX_DIMS];
  int64_t stride[MAG_MAX_DIMS];
  memcpy(shape, x->meta.coords.shape, sizeof shape);
  memcpy(stride, x->meta.coords.strides, sizeof stride);
  mag_swap(int64_t, shape[ax0], shape[ax1]);
  mag_swap(int64_t, stride[ax0], stride[ax1]);
  (void)result;
  return mag_strided_view(err, out_result, x->ctx, x, x->meta.coords.rank, shape, stride, x->meta.storage_offset);
}

mag_status_t mag_T(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x) {
  int64_t rank = mag_tensor_rank(x);
  if (rank < 2) {
    mag_rc_incref(x);
    *out_result = x;
    return MAG_OK;
  }
  if (rank == 2) return mag_transpose(err, out_result, x, 0, 1);
  int64_t dims[MAG_MAX_DIMS];
  for (int64_t i=0; i < rank; ++i)
    dims[i] = rank-1-i;
  return mag_permute(err, out_result, x, dims, rank);
}

mag_status_t mag_permute(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  if (mag_unlikely(!(rank >= 0 && rank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_RANK, "permute: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  int64_t axes[MAG_MAX_DIMS];
  for (int64_t i=0; i < rank; ++i) axes[i] = dims[i];

  for (int64_t i=0; i < rank; ++i)
  for (int64_t j=i+1; j < rank; ++j)
    if (mag_unlikely(!(axes[i] != axes[j])))
      return mag_set_error(err, MAG_ERR_PARAM, "permute: duplicate axis %" PRIi64 " at positions %" PRIi64 " and %" PRIi64 ".", axes[i], i, j);

  int64_t shape[MAG_MAX_DIMS] = {0};
  int64_t stride[MAG_MAX_DIMS] = {0};
  for (int64_t i=0; i < rank; ++i) {
    shape[i] = x->meta.coords.shape[axes[i]];
    stride[i] = x->meta.coords.strides[axes[i]];
  }
  (void)result;
  return mag_strided_view(err, out_result, x->ctx, x, x->meta.coords.rank, shape, stride, x->meta.storage_offset);
}

mag_status_t mag_flip(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t ndims) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  int64_t ra = x->meta.coords.rank;
  if (mag_unlikely(!(ndims >= 0 && ndims <= ra)))
      return mag_set_error(err, MAG_ERR_PARAM, "flip: number of axes must be in [0, %" PRIi64 "], but got %" PRIi64 ".", ra, ndims);
  int64_t axes[MAG_MAX_DIMS];
  bool seen[MAG_MAX_DIMS] = {0};
  for (int64_t i=0; i < ndims; ++i) {
    int64_t ax = dims[i];
    mag_norm_axis(&ax, ra);
    if (mag_unlikely(!(ax >= 0 && ax < ra)))
      return mag_set_error(err, MAG_ERR_PARAM, "flip: axis %" PRIi64 " is out of range for rank %" PRIi64 ".", dims[i], ra);
    if (mag_unlikely(seen[ax]))
      return mag_set_error(err, MAG_ERR_PARAM, "flip: axis %" PRIi64 " appears more than once.", ax);
    seen[ax] = true;
    axes[i] = ax;
  }
  int64_t shape[MAG_MAX_DIMS];
  int64_t stride[MAG_MAX_DIMS];
  memcpy(shape, x->meta.coords.shape, sizeof shape);
  memcpy(stride, x->meta.coords.strides, sizeof stride);
  int64_t offset = x->meta.storage_offset;
  for (int64_t i=0; i < ndims; ++i) {
    int64_t ax = axes[i];
    if (x->meta.coords.shape[ax] > 0)
      offset += (x->meta.coords.shape[ax]-1)*x->meta.coords.strides[ax];
    stride[ax] = -x->meta.coords.strides[ax];
  }
  (void)result;
  return mag_strided_view(err, out_result, x->ctx, x, ra, shape, stride, offset);
}

mag_status_t mag_contiguous(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x) {
  if (!x->meta.storage_offset && mag_tensor_is_contiguous(x)) {
    mag_rc_incref(x); /* Borrow +1 ref for caller; *out may alias x — caller must mag_tensor_decref(*out) once */
    *out_result = x;
    return MAG_OK;
  }
  return mag_clone(err, out_result, x);
}

mag_status_t mag_squeeze_all(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x) {
  *out_result = NULL;
  int64_t rank = x->meta.coords.rank;
  if (!rank) return mag_view(err, out_result, x, x->meta.coords.shape, 0);
  int64_t shape[MAG_MAX_DIMS];
  int64_t nrank = 0;
  for (int64_t i=0; i < rank; ++i) {
    int64_t sz = x->meta.coords.shape[i];
    if (sz != 1) shape[nrank++] = sz;
  }
  return nrank == rank ? mag_view(err, out_result, x, x->meta.coords.shape, rank) : mag_view(err, out_result, x, shape, nrank);
}

mag_status_t mag_squeeze_dim(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
  *out_result = NULL;
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(rank <= 0))
      return mag_set_error(err, MAG_ERR_RANK, "squeeze: cannot squeeze a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "squeeze: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  int64_t sz = x->meta.coords.shape[dim];
  if (sz != 1) return mag_view(err, out_result, x, x->meta.coords.shape, rank);
  int64_t shape[MAG_MAX_DIMS];
  int64_t nrank = 0;
  for (int64_t i=0; i < rank; ++i) {
    if (i == dim) continue;
    shape[nrank++] = x->meta.coords.shape[i];
  }
  return mag_view(err, out_result, x, shape, nrank);
}

mag_status_t mag_unsqueeze(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
  *out_result = NULL;
  int64_t rank = x->meta.coords.rank;
  int64_t nrank = rank+1;
  if (mag_unlikely(!(nrank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_RANK, "unsqueeze: result would exceed the maximum rank of %d.", MAG_MAX_DIMS);
  mag_norm_axis(&dim, nrank);
  if (mag_unlikely(!(0 <= dim && dim < nrank)))
      return mag_set_error(err, MAG_ERR_RANK, "unsqueeze: dim %" PRIi64 " is out of range for new rank %" PRIi64 ".", dim, nrank);
  int64_t shape[MAG_MAX_DIMS];
  for (int64_t i=0, j=0; i < nrank; ++i)
    shape[i] = i == dim ? 1 : x->meta.coords.shape[j++];
  return mag_view(err, out_result, x, shape, nrank);
}

mag_status_t mag_flatten(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t start_dim, int64_t end_dim) {
  *out_result = NULL;
  int64_t rank = x->meta.coords.rank;
  if (!rank) return mag_view(err, out_result, x, x->meta.coords.shape, 0);
  mag_norm_axis(&start_dim, rank);
  mag_norm_axis(&end_dim, rank);
  if (mag_unlikely(!(0 <= start_dim && start_dim < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "flatten: start_dim %" PRIi64 " is out of range for rank %" PRIi64 ".", start_dim, rank);
  if (mag_unlikely(!(0 <= end_dim && end_dim < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "flatten: end_dim %" PRIi64 " is out of range for rank %" PRIi64 ".", end_dim, rank);
  if (mag_unlikely(start_dim > end_dim))
      return mag_set_error(err, MAG_ERR_PARAM, "flatten: start_dim must be <= end_dim, but got %" PRIi64 " > %" PRIi64 ".", start_dim, end_dim);
  int64_t shape[MAG_MAX_DIMS];
  int64_t nrank = 0;
  for (int64_t i=0; i < start_dim; ++i) shape[nrank++] = x->meta.coords.shape[i];
  int64_t sz=1;
  for (int64_t i=start_dim; i <= end_dim; ++i) sz *= x->meta.coords.shape[i];
  shape[nrank++] = sz;
  for (int64_t i=end_dim+1; i < rank; ++i) shape[nrank++] = x->meta.coords.shape[i];
  if (mag_unlikely(!(nrank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_RANK, "flatten: result rank %" PRIi64 " exceeds the maximum rank of %d.", nrank, MAG_MAX_DIMS);
  mag_status_t status = mag_view(err, out_result, x, shape, nrank); /* Try view first */
  if (mag_iserr(status))
    status = mag_reshape(err, out_result, x, shape, nrank);
  return status;
}

mag_status_t mag_unflatten(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, const int64_t *sizes, int64_t sizes_rank) {
  *out_result = NULL;
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(sizes_rank <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "unflatten: sizes must contain at least one dimension.");
  if (mag_unlikely(!(sizes != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "unflatten: sizes must not be NULL.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "unflatten: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  if (mag_unlikely(!(sizes_rank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_RANK, "unflatten: sizes rank %" PRIi64 " exceeds the maximum rank of %d.", sizes_rank, MAG_MAX_DIMS);
  int64_t nr = rank-1+sizes_rank;
  if (mag_unlikely(!(nr <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_RANK, "unflatten: result rank %" PRIi64 " exceeds the maximum rank of %d.", nr, MAG_MAX_DIMS);
  int64_t resolved[MAG_MAX_DIMS];
  mag_status_t status = mag_infer_missing_dim(
    err,
    &resolved,
    sizes,
    sizes_rank,
    x->meta.coords.shape[dim]
  );
  if (mag_iserr(status)) return status;
  int64_t shape[MAG_MAX_DIMS];
  int64_t k=0;
  for (int64_t i=0; i < dim; ++i) shape[k++] = x->meta.coords.shape[i];
  for (int64_t i=0; i < sizes_rank; ++i) shape[k++] = resolved[i];
  for (int64_t i=dim+1; i < rank; ++i) shape[k++] = x->meta.coords.shape[i];
  status = mag_view(err, out_result, x, shape, nr);
  if (mag_iserr(status)) {
    mag_error_t ignored = {0};
    status = mag_reshape(&ignored, out_result, x, shape, nr);
    if (mag_iserr(status))
      status = mag_reshape(err, out_result, x, shape, nr);
  }
  return status;
}

mag_status_t mag_narrow(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t start, int64_t length) {
  *out_result = NULL;
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(rank <= 0))
      return mag_set_error(err, MAG_ERR_RANK, "narrow: cannot narrow a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "narrow: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  if (mag_unlikely(length < 0))
      return mag_set_error(err, MAG_ERR_PARAM, "narrow: length must be >= 0, but got %" PRIi64 ".", length);
  if (mag_unlikely(length <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "narrow: length 0 is not supported.");
  int64_t sz = x->meta.coords.shape[dim];
  mag_norm_axis(&start, sz);
  if (mag_unlikely(!(start >= 0 && start <= sz)))
      return mag_set_error(err, MAG_ERR_PARAM, "narrow: start %" PRIi64 " is out of bounds for dim of size %" PRIi64 ".", start, sz);
  int64_t end = start+length;
  if (mag_unlikely(!(end <= sz)))
      return mag_set_error(err, MAG_ERR_PARAM, "narrow: range [%" PRIi64 ", %" PRIi64 ") exceeds dim size %" PRIi64 ".", start, end, sz);
  return mag_view_slice(err, out_result, x, dim, start, length, 1);
}

mag_status_t mag_movedim(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t src, int64_t dst) {
  *out_result = NULL;
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(rank <= 0))
      return mag_set_error(err, MAG_ERR_RANK, "movedim: cannot apply movedim to a scalar tensor.");
  mag_norm_axis(&src, rank);
  mag_norm_axis(&dst, rank);
  if (mag_unlikely(!(0 <= src && src < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "movedim: source dim %" PRIi64 " is out of range for rank %" PRIi64 ".", src, rank);
  if (mag_unlikely(!(0 <= dst && dst < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "movedim: destination dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dst, rank);
  if (src == dst)
    return mag_view(err, out_result, x, x->meta.coords.shape, rank);
  int64_t perm[MAG_MAX_DIMS];
  for (int64_t i=0; i < rank; ++i) perm[i] = i;
  int64_t tmp = perm[src];
  if (src < dst) {
    for (int64_t i=src; i < dst; ++i) perm[i] = perm[i+1];
    perm[dst] = tmp;
  } else {
    for (int64_t i=src; i > dst; --i) perm[i] = perm[i-1];
    perm[dst] = tmp;
  }
  return mag_permute(err, out_result, x, perm, rank);
}

mag_status_t mag_select(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t index) {
  *out_result = NULL;
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(rank <= 0))
      return mag_set_error(err, MAG_ERR_RANK, "select: cannot select from a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "select: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  int64_t sz = x->meta.coords.shape[dim];
  mag_norm_axis(&index, sz);
  if (mag_unlikely(!(0 <= index && index < sz)))
      return mag_set_error(err, MAG_ERR_PARAM, "select: index %" PRIi64 " is out of bounds for dim of size %" PRIi64 ".", index, sz);
  mag_tensor_t *tmp = NULL;
  mag_status_t status = mag_view_slice(err, &tmp, x, dim, index, 1, 1);
  if (mag_iserr(status)) return status;
  status = mag_squeeze_dim(err, out_result, tmp, dim);
  if (mag_iserr(status)) {
    mag_tensor_decref(tmp);
    return status;
  }
  return MAG_OK;
}

mag_status_t mag_split(mag_error_t *err, mag_tensor_t **outs, int64_t num_splits, mag_tensor_t *x, int64_t split_size, int64_t dim) {
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(!(split_size > 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "split: split_size must be > 0, but got %" PRIi64 ".", split_size);
  if (mag_unlikely(rank <= 0))
      return mag_set_error(err, MAG_ERR_RANK, "split: cannot split a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_ERR_RANK, "split: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  int64_t sz = x->meta.coords.shape[dim];
  int64_t expected_chunks = 0;
  if (sz > 0) expected_chunks = (int64_t)mag_uint128_ceildiv((uint64_t)sz, (uint64_t)split_size);
  if (mag_unlikely(!(num_splits >= 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "split: number of splits must be >= 0, but got %" PRIi64 ".", num_splits);
  if (mag_unlikely(!(num_splits == expected_chunks)))
      return mag_set_error(err, MAG_ERR_PARAM, "split: number of splits (%" PRIi64 ") does not match the expected chunk count (%" PRIi64 ").", num_splits, expected_chunks);
  if (!num_splits) return MAG_OK;
  for (int64_t i=0; i < num_splits; ++i) outs[i] = NULL;
  int64_t start = 0;
  for (int64_t i=0; i < num_splits; ++i) {
    int64_t remaining = sz - start;
    int64_t length = remaining < split_size ? remaining : split_size;  /* min */
    mag_status_t status = mag_view_slice(err, outs+i, x, dim, start, length, 1);
    if (mag_iserr(status)) {
      for (int64_t j=0; j < i; ++j) {
        mag_tensor_decref(outs[j]);
        outs[j] = NULL;
      }
      return status;
    }
    start += length;
  }
  return MAG_OK;
}

mag_status_t mag_mean(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  return mag_op_stub_reduction(err, out_result, MAG_OP_MEAN, x, dims, rank, keepdim);
}

mag_status_t mag_minima(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  return mag_op_stub_reduction(err, out_result, MAG_OP_MINIMA, x, dims, rank, keepdim);
}

mag_status_t mag_maxima(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  return mag_op_stub_reduction(err, out_result, MAG_OP_MAXIMA, x, dims, rank, keepdim);
}

mag_status_t mag_argmin(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  return mag_op_stub_reduction(err, out_result, MAG_OP_ARGMIN, x, dims, rank, keepdim);
}

mag_status_t mag_argmax(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  return mag_op_stub_reduction(err, out_result, MAG_OP_ARGMAX, x, dims, rank, keepdim);
}

mag_status_t mag_sum(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  return mag_op_stub_reduction(err, out_result, MAG_OP_SUM, x, dims, rank, keepdim);
}

mag_status_t mag_prod(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  return mag_op_stub_reduction(err, out_result, MAG_OP_PROD, x, dims, rank, keepdim);
}

mag_status_t mag_all(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  return mag_op_stub_reduction(err, out_result, MAG_OP_ALL, x, dims, rank, keepdim);
}

mag_status_t mag_any(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  return mag_op_stub_reduction(err, out_result, MAG_OP_ANY, x, dims, rank, keepdim);
}

mag_status_t mag_topk(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t k, int64_t dim, bool largest, bool sorted) {
  *out_values = NULL;
  *out_indices = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "topk: input tensor must not be NULL.");
  mag_context_t *ctx = x->ctx;
  if (mag_unlikely(k <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "topk: k must be > 0, but got %" PRIi64 ".", k);
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(rank <= 0))
      return mag_set_error(err, MAG_ERR_RANK, "topk: requires a tensor with rank > 0.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_ERR_DIM, "topk: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  int64_t dim_size = x->meta.coords.shape[dim];
  if (mag_unlikely(k > dim_size))
      return mag_set_error(err, MAG_ERR_PARAM, "topk: k (%" PRIi64 ") must be <= the size of dim %" PRIi64 " (%" PRIi64 ").", k, dim, dim_size);
  int64_t shape[MAG_MAX_DIMS];
  memcpy(shape, x->meta.coords.shape, sizeof(*shape)*rank);
  shape[dim] = k;
  mag_tensor_t *values = NULL;
  mag_tensor_t *indices = NULL;
  mag_status_t status = mag_empty(err, &values, ctx, x->meta.dtype, rank, shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &indices, ctx, MAG_DTYPE_INT64, rank, shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) {
    mag_tensor_decref(values);
    return status;
  }
  mag_op_params_t params = {
    .topk = {
      .k = k,
      .dim = dim,
      .largest = largest,
      .sorted = sorted
    }
  };
  status = mag_check_dtype_and_device_compat(err, MAG_OP_TOPK, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_TOPK, false, &x, 1, (mag_tensor_t*[2]){values, indices}, 2, &params);
  if (mag_iserr(status)) return status;
  *out_values = values;
  *out_indices = indices;
  return MAG_OK;
}

mag_status_t mag_cusum(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
  return mag_op_stub_cu(err, out_result, MAG_OP_CUSUM, "sum", x, dim);
}

mag_status_t mag_cuprod(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
  return mag_op_stub_cu(err, out_result, MAG_OP_CUPROD, "prod", x, dim);
}

mag_status_t mag_cumax(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t dim) {
  return mag_op_stub_cu_ex(err, out_values, out_indices, MAG_OP_CUMAX, "max", x, dim);
}

mag_status_t mag_cumin(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t dim) {
  return mag_op_stub_cu_ex(err, out_values, out_indices, MAG_OP_CUMIN, "min", x, dim);
}

mag_status_t mag_outer(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *a, mag_tensor_t *b) {
  *out_result = NULL;
  if (mag_unlikely(!(a != NULL)))
    return mag_set_error(err, MAG_ERR_PARAM, "outer: first input tensor must not be NULL.");
  if (mag_unlikely(!(b != NULL)))
    return mag_set_error(err, MAG_ERR_PARAM, "outer: second input tensor must not be NULL.");
  if (mag_unlikely(a->meta.coords.rank != 1))
    return mag_set_error(err, MAG_ERR_RANK, "outer: first input must be 1D, but got rank %" PRIi64 ".", a->meta.coords.rank);
  if (mag_unlikely(b->meta.coords.rank != 1))
    return mag_set_error(err, MAG_ERR_RANK, "outer: second input must be 1D, but got rank %" PRIi64 ".", b->meta.coords.rank);
  if (mag_unlikely(!mag_device_id_eq(mag_tensor_device_id(a), mag_tensor_device_id(b))))
    return mag_set_error(err, MAG_ERR_DEVICE, "outer: input tensors must be on the same device.");
  mag_tensor_t *av = NULL;
  mag_tensor_t *bv = NULL;
  mag_status_t status = mag_unsqueeze(err, &av, a, 1);
  if (mag_iserr(status)) return status;
  status = mag_unsqueeze(err, &bv, b, 0);
  if (mag_iserr(status)) {
    mag_tensor_decref(av);
    return status;
  }
  status = mag_mul(err, out_result, av, bv);
  mag_tensor_decref(av);
  mag_tensor_decref(bv);
  return status;
}

#define mag_impl_unary_pair(name, op) \
  mag_status_t mag_##name(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t* x) { return mag_op_stub_unary(err, out_result, MAG_OP_##op, x, NULL, false); } \
  mag_status_t mag_##name##_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t* x) { return mag_op_stub_unary(err, out_result, MAG_OP_##op, x, NULL, true); }

mag_impl_unary_pair(not, NOT)
mag_impl_unary_pair(abs, ABS)
mag_impl_unary_pair(sgn, SGN)
mag_impl_unary_pair(neg, NEG)
mag_impl_unary_pair(log, LOG)
mag_impl_unary_pair(log10, LOG10)
mag_impl_unary_pair(log1p, LOG1P)
mag_impl_unary_pair(log2, LOG2)
mag_impl_unary_pair(sqr, SQR)
mag_impl_unary_pair(rcp, RCP)
mag_impl_unary_pair(sqrt, SQRT)
mag_impl_unary_pair(rsqrt, RSQRT)
mag_impl_unary_pair(sin, SIN)
mag_impl_unary_pair(cos, COS)
mag_impl_unary_pair(tan, TAN)
mag_impl_unary_pair(sinh, SINH)
mag_impl_unary_pair(cosh, COSH)
mag_impl_unary_pair(tanh, TANH)
mag_impl_unary_pair(asin, ASIN)
mag_impl_unary_pair(acos, ACOS)
mag_impl_unary_pair(atan, ATAN)
mag_impl_unary_pair(asinh, ASINH)
mag_impl_unary_pair(acosh, ACOSH)
mag_impl_unary_pair(atanh, ATANH)
mag_impl_unary_pair(step, STEP)
mag_impl_unary_pair(erf, ERF)
mag_impl_unary_pair(erfc, ERFC)
mag_impl_unary_pair(exp, EXP)
mag_impl_unary_pair(exp2, EXP2)
mag_impl_unary_pair(expm1, EXPM1)
mag_impl_unary_pair(floor, FLOOR)
mag_impl_unary_pair(ceil, CEIL)
mag_impl_unary_pair(round, ROUND)
mag_impl_unary_pair(trunc, TRUNC)
mag_impl_unary_pair(softmax, SOFTMAX)
mag_impl_unary_pair(softmax_dv, SOFTMAX_DV)
mag_impl_unary_pair(sigmoid, SIGMOID)
mag_impl_unary_pair(sigmoid_dv, SIGMOID_DV)
mag_impl_unary_pair(hard_sigmoid, HARD_SIGMOID)
mag_impl_unary_pair(silu, SILU)
mag_impl_unary_pair(silu_dv, SILU_DV)
mag_impl_unary_pair(tanh_dv, TANH_DV)
mag_impl_unary_pair(relu, RELU)
mag_impl_unary_pair(relu_dv, RELU_DV)
mag_impl_unary_pair(gelu, GELU)
mag_impl_unary_pair(gelu_approx, GELU_APPROX)
mag_impl_unary_pair(gelu_dv, GELU_DV)

#undef mag_impl_unary_pair

mag_status_t mag_pad(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *pad, int64_t pad_len, const char *mode, mag_scalar_t value) {
  *out_result = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "pad: input tensor must not be NULL.");
  if (mag_unlikely(!(pad != NULL || pad_len == 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "pad: padding array must not be NULL.");
  if (mag_unlikely(pad_len < 0))
      return mag_set_error(err, MAG_ERR_PARAM, "pad: pad_len must be >= 0.");
  if (mag_unlikely(!(mode && *mode)))
      return mag_set_error(err, MAG_ERR_PARAM, "pad: invalid mode string");
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(pad_len > (rank<<1)))
      return mag_set_error(err, MAG_ERR_PARAM, "pad: expected at most %" PRIi64 " padding values for rank %" PRIi64 ", but got %" PRIi64 ".", 2*rank, rank, pad_len);
  mag_op_params_t params = {0};
  params.pad.rank = rank;
  if (!strcmp(mode, "constant")) params.pad.mode = MAG_PAD_MODE_CONSTANT;
  else if (!strcmp(mode, "reflect")) params.pad.mode = MAG_PAD_MODE_REFLECT;
  else if (!strcmp(mode, "replicate")) params.pad.mode = MAG_PAD_MODE_REPLICATE;
  else return mag_set_error(err, MAG_ERR_PARAM, "pad: invalid mode string '%s'.", mode);
  params.pad.value = value;
  for (int64_t d=0; d < rank; ++d) {
    int64_t idx = (rank - 1 - d)<<1;
    params.pad.pad_before[d] = idx < pad_len ? pad[idx] : 0;
    params.pad.pad_after[d] = idx + 1 < pad_len ? pad[idx+1] : 0;
    if (mag_unlikely(!(params.pad.pad_before[d] >= 0 && params.pad.pad_after[d] >= 0)))
        return mag_set_error(err, MAG_ERR_PARAM, "pad: padding values must be >= 0.");
    if (params.pad.mode == MAG_PAD_MODE_REFLECT) {
      int64_t dim = x->meta.coords.shape[d];
      if (mag_unlikely(!(params.pad.pad_before[d] < dim && params.pad.pad_after[d] < dim)))
          return mag_set_error(err, MAG_ERR_PARAM, "pad: reflect padding on dim %" PRIi64 " must be less than input size %" PRIi64 ".", d, dim);
    }
  }
  int64_t shape[MAG_MAX_DIMS];
  for (int64_t dim=0; dim < rank; ++dim)
    shape[dim] = x->meta.coords.shape[dim] + params.pad.pad_before[dim] + params.pad.pad_after[dim];
  mag_tensor_t *result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_PAD, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &result, x->ctx, x->meta.dtype, rank, shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_PAD, false, &x, 1, &result, 1, &params);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_tril(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag) {
  *out_result = NULL;
  if (mag_unlikely(tensor->meta.coords.rank < 2))
      return mag_set_error(err, MAG_ERR_PARAM, "tril: requires rank >= 2, but got %" PRIi64 ".", tensor->meta.coords.rank);
  mag_op_params_t params = {
    .trilu = {.diag = diag}
  };
  return mag_op_stub_unary(err, out_result, MAG_OP_TRIL, tensor, &params, false);
}

mag_status_t mag_tril_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag) {
  *out_result = NULL;
  if (mag_unlikely(tensor->meta.coords.rank < 2))
      return mag_set_error(err, MAG_ERR_PARAM, "tril_: requires rank >= 2, but got %" PRIi64 ".", tensor->meta.coords.rank);
  mag_op_params_t params = {
    .trilu = {.diag = diag}
  };
  return mag_op_stub_unary(err, out_result, MAG_OP_TRIL, tensor, &params, true);
}

mag_status_t mag_triu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag) {
  *out_result = NULL;
  if (mag_unlikely(tensor->meta.coords.rank < 2))
      return mag_set_error(err, MAG_ERR_PARAM, "triu: requires rank >= 2, but got %" PRIi64 ".", tensor->meta.coords.rank);
  mag_op_params_t params = {
    .trilu = {.diag = diag}
  };
  return mag_op_stub_unary(err, out_result, MAG_OP_TRIU, tensor, &params, false);
}

mag_status_t mag_triu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag) {
  *out_result = NULL;
  if (mag_unlikely(tensor->meta.coords.rank < 2))
      return mag_set_error(err, MAG_ERR_PARAM, "triu_: requires rank >= 2, but got %" PRIi64 ".", tensor->meta.coords.rank);
  mag_op_params_t params = {
    .trilu = {.diag = diag}
  };
  return mag_op_stub_unary(err, out_result, MAG_OP_TRIU, tensor, &params, true);
}

mag_status_t mag_multinomial(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t num_samples, bool replacement) {
  *out_result = NULL;
  if (mag_unlikely(!(tensor->meta.coords.rank == 1 || tensor->meta.coords.rank == 2)))
      return mag_set_error(err, MAG_ERR_PARAM, "multinomial: requires rank 1 or 2, but got %" PRIi64 ".", tensor->meta.coords.rank);
  if (mag_unlikely(!mag_tensor_is_contiguous(tensor)))
      return mag_set_error(err, MAG_ERR_PARAM, "multinomial: input tensor must be contiguous row-major.");
  if (mag_unlikely(num_samples <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "multinomial: num_samples must be > 0, but got %" PRIi64 ".", num_samples);
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_MULTINOMIAL, &tensor, 0);
  if (mag_iserr(status)) return status;
  int64_t shape[MAG_MAX_DIMS] = {0};
  if (tensor->meta.coords.rank > 1) memcpy(shape, tensor->meta.coords.shape, (tensor->meta.coords.rank - 1)*sizeof(*shape));
  shape[tensor->meta.coords.rank-1] = num_samples;
  mag_tensor_t *result;
  status = mag_empty(err, &result, tensor->ctx, MAG_DTYPE_INT64, tensor->meta.coords.rank, shape, mag_tensor_device_id(tensor));
  if (mag_iserr(status)) return status;
  mag_op_params_t params = {
    .multinomial = {
      .samples = num_samples,
      .replacement = replacement
    }
  };
  status = mag_dispatch(err, MAG_OP_MULTINOMIAL, false, &tensor, 1, &result, 1, &params);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_cat(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count, int64_t dim) {
  *out_result = NULL;
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "cat: tensors array must not be NULL.");
  mag_tensor_t *result = NULL;
  if (mag_unlikely(count <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "cat: tensor count must be > 0.");
  mag_tensor_t *t0 = tensors[0];
  if (mag_unlikely(!(t0 != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "cat: first tensor must not be NULL.");
  int64_t rank = t0->meta.coords.rank;
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(dim >= 0 && dim < MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_PARAM, "cat: dim must be in [0, %d), but got %" PRIi64 ".", MAG_MAX_DIMS, dim);
  if (mag_unlikely(!(rank > 0 && dim < rank)))
      return mag_set_error(err, MAG_ERR_DIM, "cat: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", rank, dim);
  mag_dtype_t dtype = t0->meta.dtype;
  int64_t shape[MAG_MAX_DIMS];
  memcpy(shape, t0->meta.coords.shape, rank*sizeof(*shape));
  shape[dim] = 0;
  mag_tensor_t **tmp = (*mag_try_alloc)(NULL, count*sizeof(*tmp), 0);
  if (mag_unlikely(!tmp))
    return mag_set_error(err, MAG_ERR_OOM, "cat: failed to allocate temporary array for %zu tensors.", count);
  for (size_t i=0; i < count; ++i) {
    mag_tensor_t *tensor = tensors[i];
    if (mag_unlikely(!(tensor != NULL))) {
      for (size_t j=0; j < i; ++j) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return mag_set_error(err, MAG_ERR_PARAM, "cat: tensor at index %" PRIu64 " is NULL.", (uint64_t)i);
    }
    if (mag_unlikely(tensor->meta.coords.rank != rank)) {
      for (size_t j=0; j < i; ++j) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return mag_set_error(err, MAG_ERR_PARAM, "cat: all tensors must have the same rank (got %" PRIi64 " and %" PRIi64 ").", tensor->meta.coords.rank, rank);
    }
    if (mag_unlikely(tensor->meta.dtype != dtype)) {
      for (size_t j=0; j < i; ++j) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return mag_set_error(err, MAG_ERR_PARAM, "cat: all tensors must have the same dtype (got %s and %s).", mag_type_trait(tensor->meta.dtype)->name, mag_type_trait(dtype)->name);
    }
    for (int64_t j=0; j < rank; ++j) {
      if (j == dim) continue;
      if (mag_unlikely(tensor->meta.coords.shape[j] != t0->meta.coords.shape[j])) {
        for (size_t k=0; k < i; ++k) mag_tensor_decref(tmp[k]);
        (*mag_alloc)(tmp, 0, 0);
        return mag_set_error(err, MAG_ERR_PARAM, "cat: shapes must match on non-concat dimensions (mismatch on axis %" PRIi64 ").", j);
      }
    }
    mag_status_t cont_status = mag_contiguous(err, tmp+i, tensor); /* TODO: kernel requires all tensors to be contiguous for now, add strided path */
    if (mag_iserr(cont_status)) {
      for (size_t j=0; j < i; ++j) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return cont_status;
    }
    shape[dim] += tensor->meta.coords.shape[dim];
  }
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_CAT, tmp, (uint32_t)count);
  if (mag_iserr(status)) { for (size_t i=0; i < count; ++i) mag_tensor_decref(tmp[i]); (*mag_alloc)(tmp, 0, 0); return status; }
  status = mag_empty(err, &result, t0->ctx, dtype, rank, shape, mag_tensor_device_id(*tmp));
  if (mag_iserr(status)) { for (size_t i=0; i < count; ++i) mag_tensor_decref(tmp[i]); (*mag_alloc)(tmp, 0, 0); return status; }
  mag_op_params_t params = {
    .cat = {.dim = dim}
  };
  status = mag_dispatch(err, MAG_OP_CAT, false, tmp, count, &result, 1, &params);
  if (mag_iserr(status)) { for (size_t i=0; i < count; ++i) mag_tensor_decref(tmp[i]); (*mag_alloc)(tmp, 0, 0); return status; }
  for (size_t i=0; i < count; ++i)
    mag_tensor_decref(tmp[i]);
  (*mag_alloc)(tmp, 0, 0);
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_stack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count, int64_t dim) {
  *out_result = NULL;
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "stack: tensors array must not be NULL.");
  if (mag_unlikely(count <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "stack: tensor count must be > 0.");
  if (mag_unlikely(!(tensors[0] != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "stack: first tensor must not be NULL.");
  int64_t rank = tensors[0]->meta.coords.rank;
  if (mag_unlikely(!(dim >= 0 && dim <= rank)))
      return mag_set_error(err, MAG_ERR_DIM, "stack: dim must be in [0, %" PRIi64 "], but got %" PRIi64 ".", rank, dim);
  if (mag_unlikely(!(rank + 1 <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_DIM, "stack: result rank would exceed MAG_MAX_DIMS.");
  mag_tensor_t **tmp = (*mag_try_alloc)(NULL, count*sizeof(*tmp), 0);
  if (mag_unlikely(!tmp))
    return mag_set_error(err, MAG_ERR_OOM, "stack: failed to allocate temporary array for %zu tensors.", count);
  for (size_t i=0; i < count; ++i) {
    tmp[i] = NULL;
    mag_status_t status = mag_unsqueeze(err, &tmp[i], tensors[i], dim);
    if (mag_iserr(status)) {
      for (size_t j=0; j < i; ++j) if (tmp[j]) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return status;
    }
  }
  mag_status_t status = mag_cat(err, out_result, tmp, count, dim);
  for (size_t i=0; i < count; ++i)
    mag_tensor_decref(tmp[i]);
  (*mag_alloc)(tmp, 0, 0);
  return status;
}

mag_status_t mag_hstack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count) {
  *out_result = NULL;
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "hstack: tensors array must not be NULL.");
  if (mag_unlikely(!(count > 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "hstack: tensor count must be > 0.");
  if (mag_unlikely(!(tensors[0] != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "hstack: first tensor must not be NULL.");
  int64_t rank = tensors[0]->meta.coords.rank;
  return mag_cat(err, out_result, tensors, count, rank == 1 ? 0 : 1);
}

mag_status_t mag_vstack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count) {
  *out_result = NULL;
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "vstack: tensors array must not be NULL.");
  if (mag_unlikely(!(count > 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "vstack: tensor count must be > 0.");
  if (mag_unlikely(!(tensors[0] != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "vstack: first tensor must not be NULL.");
  int64_t rank = tensors[0]->meta.coords.rank;
  return rank == 1 ? mag_stack(err, out_result, tensors, count, 0) : mag_cat(err, out_result, tensors, count, 0);
}

mag_status_t mag_dstack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count) {
  *out_result = NULL;
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "dstack: tensors array must not be NULL.");
  if (mag_unlikely(!(count > 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "dstack: tensor count must be > 0.");
  if (mag_unlikely(!(tensors[0] != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "dstack: first tensor must not be NULL.");
  int64_t rank = tensors[0]->meta.coords.rank;
  if (rank >= 3)
    return mag_cat(err, out_result, tensors, count, 2);
  mag_tensor_t **tmp = (*mag_try_alloc)(NULL, count*sizeof(*tmp), 0);
  if (mag_unlikely(!tmp))
    return mag_set_error(err, MAG_ERR_OOM, "dstack: failed to allocate temporary array for %zu tensors.", count);
  for (size_t i = 0; i < count; ++i) {
    tmp[i] = NULL;
    if (rank == 1) {
      mag_tensor_t *a = NULL;
      mag_tensor_t *b = NULL;
      mag_status_t st = mag_unsqueeze(err, &a, tensors[i], 0);
      if (mag_iserr(st)) {
        for (size_t j=0; j < i; ++j) if (tmp[j]) mag_tensor_decref(tmp[j]);
        (*mag_alloc)(tmp, 0, 0);
        return st;
      }
      st = mag_unsqueeze(err, &b, a, 2);
      mag_tensor_decref(a);
      if (mag_iserr(st)) {
        for (size_t j=0; j < i; ++j) if (tmp[j]) mag_tensor_decref(tmp[j]);
        (*mag_alloc)(tmp, 0, 0);
        return st;
      }
      tmp[i] = b;
    } else {
      mag_status_t st = mag_unsqueeze(err, &tmp[i], tensors[i], 2);
      if (mag_iserr(st)) {
        for (size_t j=0; j < i; ++j) if (tmp[j]) mag_tensor_decref(tmp[j]);
        (*mag_alloc)(tmp, 0, 0);
        return st;
      }
    }
  }
  mag_status_t status = mag_cat(err, out_result, tmp, count, 2);
  for (size_t i=0; i < count; ++i)
    mag_tensor_decref(tmp[i]);
  (*mag_alloc)(tmp, 0, 0);
  return status;
}

mag_status_t mag_chunk(mag_error_t *err, mag_tensor_t ***out_chunks, size_t *out_count, mag_tensor_t *x, int64_t chunks, int64_t dim) {
  *out_chunks = NULL;
  *out_count = 0;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "chunk: input tensor must not be NULL.");
  if (mag_unlikely(!(chunks > 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "chunk: chunks must be > 0.");
  int64_t rank = x->meta.coords.rank;
  if (mag_unlikely(rank <= 0))
      return mag_set_error(err, MAG_ERR_DIM, "chunk: input rank must be > 0.");
  if (mag_unlikely(!(dim >= 0 && dim < rank)))
      return mag_set_error(err, MAG_ERR_DIM, "chunk: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", rank, dim);
  int64_t n = x->meta.coords.shape[dim];
  if (n == 0) {
    *out_chunks = NULL;
    *out_count = 0;
    return MAG_OK;
  }
  int64_t chunk_size = (int64_t)mag_uint128_ceildiv((uint64_t)n, (uint64_t)chunks);
  int64_t actual = (int64_t)mag_uint128_ceildiv((uint64_t)n, (uint64_t)chunk_size);
  mag_tensor_t **res = (*mag_try_alloc)(NULL, (size_t)actual*sizeof(*res), 0);
  if (mag_unlikely(!res))
    return mag_set_error(err, MAG_ERR_OOM, "chunk: failed to allocate result array for %" PRIi64 " chunks.", actual);
  memset(res, 0, (size_t)actual*sizeof(*res));
  for (int64_t i=0; i < actual; ++i) {
    int64_t start = i*chunk_size;
    int64_t len = chunk_size;
    if (start+len > n) len = n-start;
    mag_status_t status = mag_narrow(err, res+i, x, dim, start, len);
    if (status != MAG_OK) {
      for (int64_t j=0; j < actual; ++j)
        if (res[j]) mag_tensor_decref(res[j]);
      (*mag_alloc)(res, 0, 0);
      return status;
    }
  }
  *out_chunks = res;
  *out_count = (size_t)actual;
  return MAG_OK;
}

mag_status_t mag_einsum(mag_error_t *err, mag_tensor_t **out_result, const char *equation, mag_tensor_t **args, size_t num_args) {
  return mag_einsum_eval(err, out_result, equation, (const mag_tensor_t **)args, num_args);
}

mag_status_t mag_one_hot(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *indices, int64_t num_classes) {
  *out_result = NULL;
  mag_context_t *ctx = indices->ctx;
  if (mag_unlikely(!(indices->meta.dtype == MAG_DTYPE_INT64)))
      return mag_set_error(err, MAG_ERR_PARAM, "one_hot: indices must have dtype int64, but got %s.", mag_type_trait(indices->meta.dtype)->name);
  if (mag_unlikely(!(num_classes >= -1)))
      return mag_set_error(err, MAG_ERR_PARAM, "one_hot: num_classes must be >= -1, but got %" PRIi64 ".", num_classes);
  if (num_classes == -1) {
    mag_tensor_t *maxv = NULL;
    mag_status_t status = mag_maxima(err, &maxv, indices, NULL, 0, false);
    if (mag_iserr(status)) return status;
    mag_scalar_t max_scalar;
    status = mag_tensor_item(err, maxv, &max_scalar);
    if (mag_iserr(status)) {
      mag_tensor_decref(maxv);
      return status;
    }
    int64_t max_class = mag_scalar_as_int64(max_scalar);
    mag_tensor_decref(maxv);
    num_classes = max_class >= 0 ? 1+max_class : 0;
  }
  if (mag_unlikely(!(num_classes > 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "one_hot: inferred num_classes must be > 0, but got %" PRIi64 ".", num_classes);
  int64_t rank = indices->meta.coords.rank;
  if (mag_unlikely(!(rank + 1 <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_RANK, "one_hot: result rank (rank(indices)+1) exceeds the maximum rank of %d.", MAG_MAX_DIMS);
  int64_t orank = rank+1;
  int64_t oshape[MAG_MAX_DIMS];
  for (int64_t i=0; i < rank; ++i)
    oshape[i] = indices->meta.coords.shape[i];
  oshape[rank] = num_classes;
  mag_tensor_t *result;
  mag_status_t status = mag_zeros(err, &result, ctx, MAG_DTYPE_INT64, orank, oshape, mag_tensor_device_id(indices));
  if (mag_iserr(status)) return status;
  mag_op_params_t params = {
    .one_hot = { .num_classes = num_classes }
  };
  status = mag_check_dtype_and_device_compat(err, MAG_OP_ONE_HOT, &indices, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_ONE_HOT, false, &indices, 1, &result, 1, &params);
  if (mag_iserr(status)) {
    mag_tensor_decref(result);
    return status;
  }
  *out_result = result;
  return MAG_OK;
}

#define mag_impl_binary_pair(name, op, logical) \
  mag_status_t mag_##name(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t* x, mag_tensor_t* y) { return mag_op_stub_binary(err, out_result, MAG_OP_##op, x, y, logical ? MAG_BINOP_LOGICAL : 0); } \
  mag_status_t mag_##name##_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t* x, mag_tensor_t* y) { return mag_op_stub_binary(err, out_result, MAG_OP_##op, x, y, (logical ? MAG_BINOP_LOGICAL : 0)+MAG_BINOP_INPLACE); }

mag_impl_binary_pair(add, ADD, false)
mag_impl_binary_pair(sub, SUB, false)
mag_impl_binary_pair(mul, MUL, false)
mag_impl_binary_pair(div, DIV, false)
mag_impl_binary_pair(floordiv, FLOORDIV, false)
mag_impl_binary_pair(mod, MOD, false)
mag_impl_binary_pair(pow, POW, false)
mag_impl_binary_pair(and, AND, false)
mag_impl_binary_pair(or, OR, false)
mag_impl_binary_pair(xor, XOR, false)
mag_impl_binary_pair(shl, SHL, false)
mag_impl_binary_pair(shr, SHR, false)
mag_impl_binary_pair(eq, EQ, true)
mag_impl_binary_pair(ne, NE, true)
mag_impl_binary_pair(le, LE, true)
mag_impl_binary_pair(ge, GE, true)
mag_impl_binary_pair(lt, LT, true)
mag_impl_binary_pair(gt, GT, true)

#undef mag_impl_binary_pair

mag_status_t mag_min(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y) {
  return mag_op_stub_binary(err, out_result, MAG_OP_MIN, x, y, 0);
}

mag_status_t mag_max(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y) {
  return mag_op_stub_binary(err, out_result, MAG_OP_MAX, x, y, 0);
}

mag_status_t mag_where(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *cond, mag_tensor_t *x, mag_tensor_t *y) {
  *out_result = NULL;
  if (mag_unlikely(cond->meta.dtype != MAG_DTYPE_BOOLEAN))
      return mag_set_error(err, MAG_ERR_PARAM, "where: condition tensor must have dtype bool, but got %s.", mag_type_trait(cond->meta.dtype)->name);
  if (mag_unlikely(x->meta.dtype != y->meta.dtype))
      return mag_set_error(err, MAG_ERR_PARAM, "where: x and y must have the same dtype, but got %s and %s.", mag_type_trait(x->meta.dtype)->name, mag_type_trait(y->meta.dtype)->name);
  int64_t dims[MAG_MAX_DIMS];
  int64_t rank;
  const mag_coords_t *coords[3] = {&cond->meta.coords, &x->meta.coords, &y->meta.coords};
  if (mag_unlikely(!mag_coords_broadcast_multi_shape(coords, sizeof(coords)/sizeof(*coords), dims, &rank))) {
    char sc[MAG_FMT_DIM_BUF_SIZE];
    char sx[MAG_FMT_DIM_BUF_SIZE];
    char sy[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&sc, &cond->meta.coords.shape, cond->meta.coords.rank);
    mag_fmt_shape(&sx, &x->meta.coords.shape, x->meta.coords.rank);
    mag_fmt_shape(&sy, &y->meta.coords.shape, y->meta.coords.rank);
    return mag_set_error(err, MAG_ERR_BROADCAST,
      "where: cannot broadcast shapes %s, %s and %s.\n"
      "    Hint: ensure that cond, x and y are broadcast-compatible.",
      sc, sx, sy);
  }
  mag_tensor_t *result = NULL;
  mag_status_t status = rank ? mag_empty(err, &result, x->ctx, x->meta.dtype, rank, dims, mag_tensor_device_id(cond)) : mag_empty_scalar(err, &result, x->ctx, x->meta.dtype, mag_tensor_device_id(cond));
  if (mag_iserr(status)) return status;
  mag_tensor_t *in[3] = {cond, x, y};
  status = mag_check_dtype_and_device_compat(err, MAG_OP_WHERE, in, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_WHERE, false, in, sizeof(in)/sizeof(*in), &result, 1, NULL);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_clamp(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *min, mag_tensor_t *max) {
  *out_result = NULL;
  if (mag_unlikely(!(x->meta.dtype == min->meta.dtype && min->meta.dtype == max->meta.dtype)))
      return mag_set_error(err, MAG_ERR_PARAM, "clamp: x, min and max must have the same dtype, but got %s, %s and %s.", mag_type_trait(x->meta.dtype)->name, mag_type_trait(min->meta.dtype)->name, mag_type_trait(max->meta.dtype)->name);
  int64_t dims[MAG_MAX_DIMS];
  int64_t rank;
  const mag_coords_t *coords[3] = {&x->meta.coords, &min->meta.coords, &max->meta.coords};
  if (mag_unlikely(!mag_coords_broadcast_multi_shape(coords, sizeof(coords)/sizeof(*coords), dims, &rank))) {
    char sc[MAG_FMT_DIM_BUF_SIZE];
    char sx[MAG_FMT_DIM_BUF_SIZE];
    char sy[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&sc, &x->meta.coords.shape, x->meta.coords.rank);
    mag_fmt_shape(&sx, &min->meta.coords.shape, min->meta.coords.rank);
    mag_fmt_shape(&sy, &max->meta.coords.shape, max->meta.coords.rank);
    return mag_set_error(err, MAG_ERR_BROADCAST,
      "clamp: cannot broadcast shapes %s, %s and %s.\n"
      "    Hint: ensure that x, min and max are broadcast-compatible.",
      sc, sx, sy);
  }
  mag_tensor_t *result = NULL;
  mag_status_t status = rank ? mag_empty(err, &result, x->ctx, x->meta.dtype, rank, dims, mag_tensor_device_id(x)) : mag_empty_scalar(err, &result, x->ctx, x->meta.dtype, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  mag_tensor_t *in[3] = {x, min, max};
  status = mag_check_dtype_and_device_compat(err, MAG_OP_CLAMP, in, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_CLAMP, false, in, sizeof(in)/sizeof(*in), &result, 1, NULL);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_clamp_min(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *min) {
  return mag_max(err, out_result, x, min);
}

mag_status_t mag_clamp_max(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *max) {
  return mag_min(err, out_result, x, max);
}

mag_status_t mag_lerp(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *start, mag_tensor_t *end, mag_tensor_t *weight) { /* TODO: this op deserves dedicated kernel */
  *out_result = NULL;
  mag_tensor_t *delta = NULL;
  mag_tensor_t *scaled = NULL;
  mag_tensor_t *result = NULL;
  mag_status_t status = mag_sub(err, &delta, end, start);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &scaled, delta, weight);
  if (mag_iserr(status)) goto cleanup;
  status = mag_add(err, &result, start, scaled);
  if (mag_iserr(status)) goto cleanup;
  *out_result = result;
  result = NULL; /* ownership transferred */
  cleanup:
    if (delta) mag_tensor_decref(delta);
  if (scaled) mag_tensor_decref(scaled);
  if (result) mag_tensor_decref(result);
  return status;
}

mag_status_t mag_lerp_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *start, mag_tensor_t *end, mag_tensor_t *weight) { /* TODO: this op deserves dedicated kernel */
  *out_result = NULL;
  mag_tensor_t *delta = NULL;
  mag_tensor_t *scaled = NULL;
  mag_status_t status = mag_sub(err, &delta, end, start);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &scaled, delta, weight);
  if (mag_iserr(status)) goto cleanup;
  status = mag_add_(err, out_result, start, scaled);
  cleanup:
    if (delta) mag_tensor_decref(delta);
    if (scaled) mag_tensor_decref(scaled);
  return status;

}

mag_status_t mag_matmul(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y) {
  *out_result = NULL;
  if (mag_unlikely(!(mag_tensor_is_floating_point_typed(x) && mag_tensor_is_floating_point_typed(y))))
      return mag_set_error(err, MAG_ERR_PARAM, "matmul: requires floating-point tensors, but got dtypes %s and %s.", mag_type_trait(x->meta.dtype)->name, mag_type_trait(y->meta.dtype)->name);
  if (mag_unlikely(!(x->meta.coords.rank >= 1 && y->meta.coords.rank >= 1)))
      return mag_set_error(err, MAG_ERR_PARAM, "matmul: both tensors must have rank >= 1, but got %" PRIi64 " and %" PRIi64 ".", x->meta.coords.rank, y->meta.coords.rank);
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_MATMUL, (mag_tensor_t *[]){x, y}, 0);
  if (mag_iserr(status)) return status;
  mag_tensor_t *result = NULL;
  int64_t rb, xb, yb;
  status = mag_matmul_verify_shapes(err, &rb, &xb, &yb, x, y);
  if (mag_iserr(status)) return status;
  status = mag_matmul_alloc_res(err, &result, rb, &xb, &yb, x, y);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_MATMUL, false, (mag_tensor_t *[2]){x, y}, 2, &result, 1, NULL);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_repeat_back(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_REPEAT_BACK, (mag_tensor_t *[]) {x, y}, 0);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &result, x->ctx, x->meta.dtype, y->meta.coords.rank, y->meta.coords.shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  /* TODO: Check for broadcastability of x and y */
  status = mag_dispatch(err, MAG_OP_REPEAT_BACK, false, (mag_tensor_t *[2]) {x, y}, 2, &result, 1, NULL);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_repeat(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *repeats, int64_t repeats_len) {
  *out_result = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "repeat: input tensor must not be NULL.");
  if (mag_unlikely(!(repeats != NULL && repeats_len > 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "repeat: repeats must be a non-empty sequence.");
  int64_t in_rank = x->meta.coords.rank;
  int64_t out_rank = in_rank > repeats_len ? in_rank : repeats_len;
  if (mag_unlikely(!(out_rank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_PARAM, "repeat: result rank would exceed MAG_MAX_DIMS.");
  mag_op_params_t params = {0};
  params.repeat.in_rank = in_rank;
  params.repeat.rank = out_rank;
  int64_t lead_x = out_rank - in_rank;
  int64_t lead_r = out_rank - repeats_len;
  for (int64_t d=0; d < out_rank; ++d) {
    int64_t is = d >= lead_x ? x->meta.coords.shape[d - lead_x] : 1;
    int64_t rs = d >= lead_r ? repeats[d - lead_r] : 1;
    if (mag_unlikely(!(rs >= 0)))
        return mag_set_error(err, MAG_ERR_PARAM, "repeat: repeat counts must be >= 0.");
    params.repeat.in_shape[d] = is;
    params.repeat.out_shape[d] = is*rs;
  }
  mag_tensor_t *result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_REPEAT, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &result, x->ctx, x->meta.dtype, out_rank, params.repeat.out_shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_REPEAT, false, &x, 1, &result, 1, &params);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_repeat_interleave(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, bool flatten, int64_t dim, const int64_t *counts, int64_t count_len) {
  *out_result = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "repeat_interleave: input tensor must not be NULL.");
  if (mag_unlikely(!(counts != NULL && count_len > 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "repeat_interleave: counts must be a non-empty sequence.");
  for (int64_t i=0; i < count_len; ++i)
    if (mag_unlikely(!(counts[i] >= 0)))
        return mag_set_error(err, MAG_ERR_PARAM, "repeat_interleave: counts must be >= 0.");
  mag_op_params_t params = {0};
  params.repeat_interleave.flatten = flatten;
  params.repeat_interleave.counts = counts;
  params.repeat_interleave.count_len = count_len;
  mag_tensor_t *xin = NULL;
  int64_t shape[MAG_MAX_DIMS];
  mag_status_t status = MAG_OK;
  if (flatten) {
    if (mag_unlikely(!(count_len == 1 || count_len == x->meta.numel)))
        return mag_set_error(err, MAG_ERR_PARAM, "repeat_interleave: counts length (%" PRIi64 ") must match input numel (%" PRIi64 ") when dim is None.", count_len, x->meta.numel);
    int64_t out_n = 0;
    if (count_len == 1)
      out_n = x->meta.numel*counts[0];
    else
      for (int64_t i=0; i < count_len; ++i) out_n += counts[i];
    params.repeat_interleave.rank = 1;
    params.repeat_interleave.out_shape[0] = out_n;
    status = mag_contiguous(err, &xin, x);
    if (mag_iserr(status)) return status;
  } else {
    if (mag_unlikely(!(x->meta.coords.rank > 0)))
        return mag_set_error(err, MAG_ERR_DIM, "repeat_interleave: input must have rank >= 1.");
    mag_norm_axis(&dim, x->meta.coords.rank);
    if (mag_unlikely(!(dim >= 0 && dim < x->meta.coords.rank)))
        return mag_set_error(err, MAG_ERR_DIM, "repeat_interleave: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", x->meta.coords.rank, dim);
    params.repeat_interleave.dim = dim;
    params.repeat_interleave.rank = x->meta.coords.rank;
    memcpy(shape, x->meta.coords.shape, params.repeat_interleave.rank*sizeof(*shape));
    int64_t axis_len = x->meta.coords.shape[dim];
    if (mag_unlikely(!(count_len == 1 || count_len == axis_len)))
        return mag_set_error(err, MAG_ERR_PARAM, "repeat_interleave: counts length (%" PRIi64 ") must match size of dim (%" PRIi64 ").", count_len, axis_len);
    if (count_len == 1)
      shape[dim] *= counts[0];
    else {
      int64_t sum = 0;
      for (int64_t i=0; i < count_len; ++i) sum += counts[i];
      shape[dim] = sum;
    }
    memcpy(params.repeat_interleave.out_shape, shape, params.repeat_interleave.rank*sizeof(*params.repeat_interleave.out_shape));
    status = mag_contiguous(err, &xin, x);
    if (mag_iserr(status)) return status;
  }
  mag_tensor_t *result = NULL;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_REPEAT_INTERLEAVE, &xin, 0);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &result, x->ctx, x->meta.dtype, params.repeat_interleave.rank, flatten ? params.repeat_interleave.out_shape : shape, mag_tensor_device_id(xin));
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_REPEAT_INTERLEAVE, false, &xin, 1, &result, 1, &params);
  if (mag_iserr(status)) return status;
  mag_tensor_decref(xin);
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_gather(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t dim, mag_tensor_t *idx) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  if (mag_unlikely(!(idx->meta.dtype == MAG_DTYPE_INT64)))
      return mag_set_error(err, MAG_ERR_PARAM, "gather: index tensor must have dtype int64, but got %s.", mag_type_trait(idx->meta.dtype)->name);
  if (mag_unlikely(!(dim >= 0 && dim < tensor->meta.coords.rank)))
      return mag_set_error(err, MAG_ERR_PARAM, "gather: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", tensor->meta.coords.rank, dim);
  if (mag_unlikely(!(idx->meta.coords.rank == tensor->meta.coords.rank)))
      return mag_set_error(err, MAG_ERR_PARAM, "gather: index rank (%" PRIi64 ") must equal input rank (%" PRIi64 "). Use embedding() for row-select indexing.", idx->meta.coords.rank, tensor->meta.coords.rank);
  mag_norm_axis(&dim, tensor->meta.coords.rank);
  if (mag_unlikely(!(dim >= 0 && dim < tensor->meta.coords.rank)))
      return mag_set_error(err, MAG_ERR_DIM, "gather: normalized dim %" PRIi64 " is out of range [0, %" PRIi64 ").", dim, tensor->meta.coords.rank);
  /* Output shape equals index shape (same as torch.gather). */
  int64_t ork = idx->meta.coords.rank;
  int64_t ax[MAG_MAX_DIMS];
  for (int64_t i = 0; i < ork; ++i) ax[i] = idx->meta.coords.shape[i];
  mag_status_t status = mag_empty(err, &result, tensor->ctx, tensor->meta.dtype, ork, ax, mag_tensor_device_id(tensor));
  if (mag_iserr(status)) return status;
  mag_op_params_t params = {
    .gather = {.dim = dim}
  };
  status = mag_check_dtype_and_device_compat(err, MAG_OP_GATHER, (mag_tensor_t *[2]){tensor, idx}, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_GATHER, false, (mag_tensor_t *[2]) {tensor, idx}, 2, &result, 1, &params);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_embedding(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *weight, mag_tensor_t *indices) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  if (mag_unlikely(!(indices->meta.dtype == MAG_DTYPE_INT64)))
      return mag_set_error(err, MAG_ERR_PARAM, "embedding: indices tensor must have dtype int64, but got %s.", mag_type_trait(indices->meta.dtype)->name);
  mag_dtype_mask_t fp_mask = MAG_DTYPE_MASK_FP;
  if (mag_unlikely(!((fp_mask & mag_dtype_bit(weight->meta.dtype)) != 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "embedding: weight tensor must have a floating-point dtype, but got %s.", mag_type_trait(weight->meta.dtype)->name);
  if (mag_unlikely(!(weight->meta.coords.rank >= 1)))
      return mag_set_error(err, MAG_ERR_PARAM, "embedding: weight must have rank >= 1.");
  if (mag_unlikely(!(indices->meta.coords.rank >= 1)))
      return mag_set_error(err, MAG_ERR_PARAM, "embedding: indices must have rank >= 1.");
  /* Output shape: indices.shape + weight.shape[1:] */
  int64_t ork = 0;
  int64_t ax[MAG_MAX_DIMS];
  for (int64_t i = 0; i < indices->meta.coords.rank; ++i) ax[ork++] = indices->meta.coords.shape[i];
  for (int64_t i = 1; i < weight->meta.coords.rank; ++i)  ax[ork++] = weight->meta.coords.shape[i];
  if (mag_unlikely(!(ork >= 1 && ork <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_ERR_RANK, "embedding: output rank must be in [1, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, ork);
  mag_status_t status = mag_empty(err, &result, weight->ctx, weight->meta.dtype, ork, ax, mag_tensor_device_id(weight));
  if (mag_iserr(status)) return status;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_EMBEDDING, (mag_tensor_t *[2]){weight, indices}, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_EMBEDDING, false, (mag_tensor_t *[2]) {weight, indices}, 2, &result, 1, NULL);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_index_add_(mag_error_t *err, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *source, double alpha) {
  if (mag_unlikely(!(self != NULL && index != NULL && source != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "index_add_: tensors must not be NULL.");
  if (mag_unlikely(index->meta.dtype != MAG_DTYPE_INT64))
      return mag_set_error(err, MAG_ERR_PARAM, "index_add_: index must have dtype int64, but got %s.", mag_type_trait(index->meta.dtype)->name);
  if (mag_unlikely(index->meta.coords.rank != 1))
      return mag_set_error(err, MAG_ERR_PARAM, "index_add_: index must be 1-D, but got rank %" PRIi64 ".", index->meta.coords.rank);
  if (mag_unlikely(self->meta.coords.rank <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "index_add_: self must have rank >= 1.");
  if (mag_unlikely(source->meta.coords.rank != self->meta.coords.rank))
      return mag_set_error(err, MAG_ERR_PARAM, "index_add_: source rank (%" PRIi64 ") must match self rank (%" PRIi64 ").", source->meta.coords.rank, self->meta.coords.rank);
  if (mag_unlikely(self->meta.dtype != source->meta.dtype))
      return mag_set_error(err, MAG_ERR_PARAM, "index_add_: self and source must have the same dtype, but got %s and %s.", mag_type_trait(self->meta.dtype)->name, mag_type_trait(source->meta.dtype)->name);
  mag_norm_axis(&dim, self->meta.coords.rank);
  if (mag_unlikely(!(dim >= 0 && dim < self->meta.coords.rank)))
      return mag_set_error(err, MAG_ERR_DIM, "index_add_: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", self->meta.coords.rank, dim);
  int64_t idx_len = index->meta.coords.shape[0];
  for (int64_t d=0; d < self->meta.coords.rank; ++d) {
    if (d == dim) {
      if (mag_unlikely(source->meta.coords.shape[d] != idx_len))
          return mag_set_error(err, MAG_ERR_PARAM, "index_add_: source size along dim (%" PRIi64 ") must match index length (%" PRIi64 ").", source->meta.coords.shape[d], idx_len);
    } else {
      if (mag_unlikely(source->meta.coords.shape[d] != self->meta.coords.shape[d]))
          return mag_set_error(err, MAG_ERR_PARAM, "index_add_: source shape must match self on non-index dimensions (mismatch on dim %" PRIi64 ").", d);
    }
  }
  mag_status_t status = mag_check_inplace_grad_ok(err, self);
  if (mag_iserr(status)) return status;
  mag_op_params_t params = {
    .index_add = {
      .dim = dim,
      .alpha = alpha
    }
  };
  mag_tensor_t *inputs[3] = {self, source, index};
  status = mag_check_dtype_and_device_compat(err, MAG_OP_INDEX_ADD, inputs, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_INDEX_ADD, true, inputs, 3, &self, 1, &params);
  if (mag_iserr(status)) return status;
  return MAG_OK;
}

static mag_status_t mag_scatter_validate(mag_error_t *err, const char *name, mag_tensor_t *self, int64_t *dim, mag_tensor_t *index, mag_tensor_t *src) {
  if (mag_unlikely(!(self != NULL && index != NULL && src != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "%s: tensors must not be NULL.", name);
  if (mag_unlikely(index->meta.dtype != MAG_DTYPE_INT64))
      return mag_set_error(err, MAG_ERR_PARAM, "%s: index must have dtype int64, but got %s.", name, mag_type_trait(index->meta.dtype)->name);
  if (mag_unlikely(self->meta.dtype != src->meta.dtype))
      return mag_set_error(err, MAG_ERR_PARAM, "%s: self and src must have the same dtype, but got %s and %s.", name, mag_type_trait(self->meta.dtype)->name, mag_type_trait(src->meta.dtype)->name);
  if (mag_unlikely(self->meta.coords.rank <= 0))
      return mag_set_error(err, MAG_ERR_PARAM, "%s: self must have rank >= 1.", name);
  if (mag_unlikely(index->meta.coords.rank != self->meta.coords.rank || src->meta.coords.rank != self->meta.coords.rank))
      return mag_set_error(err, MAG_ERR_PARAM, "%s: self, index and src must have the same rank (got %" PRIi64 ", %" PRIi64 ", %" PRIi64 ").", name, self->meta.coords.rank, index->meta.coords.rank, src->meta.coords.rank);
  mag_norm_axis(dim, self->meta.coords.rank);
  if (mag_unlikely(!(*dim >= 0 && *dim < self->meta.coords.rank)))
      return mag_set_error(err, MAG_ERR_DIM, "%s: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", name, self->meta.coords.rank, *dim);
  for (int64_t d=0; d < self->meta.coords.rank; ++d) {
    if (mag_unlikely(index->meta.coords.shape[d] > src->meta.coords.shape[d]))
        return mag_set_error(err, MAG_ERR_PARAM, "%s: index size (%" PRIi64 ") must be <= src size (%" PRIi64 ") on dim %" PRIi64 ".", name, index->meta.coords.shape[d], src->meta.coords.shape[d], d);
    if (d != *dim && mag_unlikely(index->meta.coords.shape[d] > self->meta.coords.shape[d]))
        return mag_set_error(err, MAG_ERR_PARAM, "%s: index size (%" PRIi64 ") must be <= self size (%" PRIi64 ") on dim %" PRIi64 ".", name, index->meta.coords.shape[d], self->meta.coords.shape[d], d);
  }
  return MAG_OK;
}

static mag_status_t mag_scatter_impl(mag_error_t *err, mag_opcode_t op, const char *name, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *src) {
  mag_status_t status = mag_scatter_validate(err, name, self, &dim, index, src);
  if (mag_iserr(status)) return status;
  mag_op_params_t params = {
    .scatter = {.dim = dim}
  };
  mag_tensor_t *inputs[3] = {self, src, index};
  status = mag_check_dtype_and_device_compat(err, op, inputs, 0);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, op, true, inputs, 3, &self, 1, &params);
}

mag_status_t mag_scatter_(mag_error_t *err, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *src) {
  mag_status_t status = mag_check_inplace_grad_ok(err, self);
  if (mag_iserr(status)) return status;
  return mag_scatter_impl(err, MAG_OP_SCATTER, "scatter_", self, dim, index, src);
}

mag_status_t mag_scatter(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *src) {
  *out_result = NULL;
  mag_status_t status = mag_clone(err, out_result, self);
  if (mag_iserr(status)) return status;
  return mag_scatter_impl(err, MAG_OP_SCATTER, "scatter", *out_result, dim, index, src);
}

mag_status_t mag_scatter_add_(mag_error_t *err, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *src) {
  mag_status_t status = mag_check_inplace_grad_ok(err, self);
  if (mag_iserr(status)) return status;
  return mag_scatter_impl(err, MAG_OP_SCATTER_ADD, "scatter_add_", self, dim, index, src);
}

mag_status_t mag_scatter_add(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *src) {
  *out_result = NULL;
  mag_status_t status = mag_clone(err, out_result, self);
  if (mag_iserr(status)) return status;
  return mag_scatter_impl(err, MAG_OP_SCATTER_ADD, "scatter_add", *out_result, dim, index, src);
}

mag_status_t mag_copy_(mag_error_t *err, mag_tensor_t *dst, mag_tensor_t *src) {
  if (mag_unlikely(!(dst && src)))
      return mag_set_error(err, MAG_ERR_PARAM, "copy: source and destination tensors must not be NULL.");
  if (mag_unlikely(!mag_tensor_is_shape_eq(dst, src)))
      return mag_set_error(err, MAG_ERR_PARAM, "copy: source and destination must have the same shape.");
  if (mag_unlikely(dst->meta.dtype != src->meta.dtype))
      return mag_set_error(err, MAG_ERR_PARAM,
        "copy: source and destination must have the same dtype, but got %s and %s.",
        mag_type_trait(src->meta.dtype)->name,
        mag_type_trait(dst->meta.dtype)->name
      );
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_CLONE, (mag_tensor_t *[2]){src, dst}, 2);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_CLONE, true, &src, 1, &dst, 1, NULL);
  if (mag_iserr(status)) return status;
  return MAG_OK;
}

mag_status_t mag_copy_raw_(mag_error_t *err, mag_tensor_t *tensor, const void *data, size_t size_bytes) {
  if (mag_unlikely(!(data != NULL && size_bytes > 0)))
      return mag_set_error(err, MAG_ERR_PARAM, "copy_raw: data pointer must not be NULL and size_bytes must be > 0.");
  if (mag_unlikely(tensor->meta.device->id.type != MAG_BACKEND_TYPE_CPU))
      return mag_set_error(err, MAG_ERR_PARAM, "copy_raw: tensor storage must reside on CPU, but got %s.", mag_backend_type_to_str(tensor->meta.device->id.type));
  if (mag_unlikely(mag_tensor_numbytes(tensor) != size_bytes))
      return mag_set_error(err, MAG_ERR_PARAM, "copy_raw: buffer size (%" PRIu64 " bytes) does not match the tensor size (%" PRIu64 " bytes).", (uint64_t)size_bytes, (uint64_t)mag_tensor_numbytes(tensor));
  if (mag_unlikely(!mag_tensor_is_contiguous(tensor)))
      return mag_set_error(err, MAG_ERR_PARAM, "copy_raw: tensor must be contiguous to load from a raw buffer.");
  void *dst = (void *)mag_tensor_data_ptr_mut(tensor);
  memcpy(dst, data, size_bytes);
  return MAG_OK;
}

mag_status_t mag_zeros_(mag_error_t *err, mag_tensor_t *tensor) {
  return mag_fill_(err, tensor, mag_scalar_from_uint64(0));
}

mag_status_t mag_ones_(mag_error_t *err, mag_tensor_t *tensor) {
  return mag_fill_(err, tensor, mag_scalar_from_uint64(1));
}

mag_status_t mag_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t value) {
  mag_op_params_t params = {
    .fill = {.value = value}
  };
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_FILL, NULL, 0);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_FILL, false, NULL, 0, &tensor, 1, &params);
}

mag_status_t mag_masked_fill(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value) {
  *out_result = NULL;
  mag_op_params_t params = {
    .fill = {.value = value}
  };
  mag_tensor_t *inputs[2] = {tensor, mask};
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_MASKED_FILL, inputs, 2);
  if (mag_iserr(status)) return status;
  mag_tensor_t *result = NULL;
  status = mag_empty_like(err, &result, tensor);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_MASKED_FILL, false, inputs, 2, &result, 1, &params);
  if (mag_iserr(status)) {
    mag_tensor_decref(result);
    return status;
  }
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_masked_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value) {
  mag_op_params_t params = {
    .fill = {.value = value}
  };
  mag_tensor_t *inputs[2] = {tensor, mask};
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_MASKED_FILL, inputs, 2);
  if (mag_iserr(status)) return status;
  status = mag_check_inplace_grad_ok(err, tensor);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_MASKED_FILL, true, inputs, 2, &tensor, 1, &params);
}

mag_status_t mag_uniform_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t low, mag_scalar_t high) {
  if (mag_unlikely(!mag_scalar_same_type(low, high)))
      return mag_set_error(err, MAG_ERR_PARAM, "uniform_: low and high must have the same scalar type.");
  if (mag_unlikely(!mag_tensor_is_numeric_typed(tensor)))
      return mag_set_error(err, MAG_ERR_PARAM, "uniform_: requires a numeric tensor dtype, but got %s.", mag_type_trait(tensor->meta.dtype)->name);
  if (mag_scalar_is_float64(low)) {
    if (mag_unlikely(mag_scalar_as_float64(low) >= mag_scalar_as_float64(high)))
        return mag_set_error(err, MAG_ERR_PARAM, "uniform_: low must be less than high (got low=%f, high=%f).", mag_scalar_as_float64(low), mag_scalar_as_float64(high));
  } else if (mag_scalar_is_int64(low)) {
    if (mag_unlikely(mag_scalar_as_int64(low) >= mag_scalar_as_int64(high)))
        return mag_set_error(err, MAG_ERR_PARAM, "uniform_: low must be less than high (got low=%" PRIi64 ", high=%" PRIi64 ").", mag_scalar_as_int64(low), mag_scalar_as_int64(high));
  } else if (mag_scalar_is_uint64(low)) {
    if (mag_unlikely(mag_scalar_as_uint64(low) >= mag_scalar_as_uint64(high)))
        return mag_set_error(err, MAG_ERR_PARAM, "uniform_: low must be less than high (got low=%" PRIu64 ", high=%" PRIu64 ").", mag_scalar_as_uint64(low), mag_scalar_as_uint64(high));
  } else {
    return mag_set_error(err, MAG_ERR_PARAM, "uniform_: unsupported scalar type for low/high.");
  }
  mag_op_params_t params = {
    .uniform = {.low = low, .high = high}
  };
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_RAND_UNIFORM, NULL, 0);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_RAND_UNIFORM, false, NULL, 0, &tensor, 1, &params);
}

mag_status_t mag_normal_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t mean, mag_scalar_t stddev) {
  if (mag_unlikely(!(mag_scalar_is_float64(mean) && mag_scalar_is_float64(stddev))))
      return mag_set_error(err, MAG_ERR_PARAM, "normal_: mean and stddev must be floating-point scalars.");
  if (mag_unlikely(!mag_tensor_is_floating_point_typed(tensor)))
      return mag_set_error(err, MAG_ERR_PARAM, "normal_: requires a floating-point tensor dtype, but got %s.", mag_type_trait(tensor->meta.dtype)->name);
  if (mag_unlikely(!(mag_scalar_as_float64(stddev) >= 0.0)))
      return mag_set_error(err, MAG_ERR_PARAM, "normal_: stddev must be >= 0, but got %f.", mag_scalar_as_float64(stddev));
  mag_op_params_t params = {
    .normal = {.mean = mean, .std = stddev}
  };
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_RAND_NORMAL, NULL, 0);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_RAND_NORMAL, false, NULL, 0, &tensor, 1, &params);
}

mag_status_t mag_bernoulli_(mag_error_t *err, mag_tensor_t *tensor, double p) {
  if (mag_unlikely(!(tensor->meta.dtype == MAG_DTYPE_BOOLEAN)))
      return mag_set_error(err, MAG_ERR_PARAM, "bernoulli_: requires a bool tensor dtype, but got %s.", mag_type_trait(tensor->meta.dtype)->name);
  if (mag_unlikely(!(p >= 0.0 && p <= 1.0)))
      return mag_set_error(err, MAG_ERR_PARAM, "bernoulli_: probability p must be in [0.0, 1.0], but got %f.", p);
  mag_op_params_t params = {
    .bernoulli = {.p = p}
  };
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_RAND_BERNOULLI, NULL, 0);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_RAND_BERNOULLI, false, NULL, 0, &tensor, 1, &params);
}

mag_status_t mag_detach(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor) {
  *out_result = NULL;
  mag_status_t status = mag_view(err, out_result, tensor, tensor->meta.coords.shape, tensor->meta.coords.rank);
  if (mag_iserr(status)) return status;
  mag_tensor_t *target = *out_result;
  if (target->au_state) {
    mag_rc_decref(target->au_state);
    target->au_state = NULL;
  }
  return MAG_OK;
}
