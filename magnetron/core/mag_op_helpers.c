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

#include "mag_op_helpers.h"
#include "mag_tensor.h"
#include "mag_context.h"
#include "mag_op_dispatch.h"
#include "mag_u128.h"

bool mag_arange_numel_i64(int64_t start, int64_t stop, int64_t step, int64_t *numel) {
  if (step == 0) {
    *numel = 0;
    return false;
  }
  bool ascending = step > 0;
  if ((ascending && stop <= start) || (!ascending && stop >= start)) {
    *numel = 0;
    return true;
  }
  uint64_t delta = ascending ? (uint64_t)stop-(uint64_t)start : (uint64_t)start-(uint64_t)stop;
  uint64_t step_mag = ascending ? (uint64_t)step : 0-(uint64_t)step;
  uint64_t count = mag_uint128_ceildiv(delta, step_mag);
  if (mag_unlikely(count > (uint64_t)INT64_MAX)) return false;
  *numel = (int64_t)count;
  return true;
}

bool mag_arange_numel_u64(uint64_t start, uint64_t stop, uint64_t step, int64_t *numel) {
  if (step == 0) {
    *numel = 0;
    return false;
  }
  if (stop <= start) {
    *numel = 0;
    return true;
  }
  uint64_t count = mag_uint128_ceildiv(stop-start, step);
  if (mag_unlikely(count > (uint64_t)INT64_MAX)) return false;
  *numel = (int64_t)count;
  return true;
}

bool mag_arange_numel_float(double start, double end, double step, int64_t *numel) {
  if (step == 0.0) {
    *numel = 0;
    return false;
  }
  double delta = end - start;
  if ((step > 0.0 && delta <= 0.0) || (step < 0.0 && delta >= 0.0)) {
    *numel = 0;
    return true;
  }
  double nc = ceil(delta/step - 1e-12);
  if (nc <= 0.0) {
    *numel = 0;
    return true;
  }
  if (nc > (double)INT64_MAX) return false;
  *numel = (int64_t)nc;
  return true;
}

mag_status_t mag_op_stub_reduction(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_opcode_t op,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
) {
  *out_result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, op, &x, 0);
  if (mag_iserr(status)) return status;
  mag_op_params_t params = {0};
  mag_reduce_plan_t *plan = &params.reduction.red_plan;
  status = mag_reduce_plan_init(err, plan, &x->coords, dims, rank, keepdim);
  if (mag_iserr(status)) return status;
  mag_tensor_t *result = NULL;
  mag_dtype_t type= x->dtype;
  if ((op == MAG_OP_SUM || op == MAG_OP_PROD) && mag_tensor_is_integer_typed(x))
    type = mag_dtype_bit(x->dtype) & MAG_DTYPE_MASK_UINT ? MAG_DTYPE_UINT64 : MAG_DTYPE_INT64;  /* For integral types sum/prod use wide int64/uint64 */
  else if (op == MAG_OP_ANY || op == MAG_OP_ALL) type = MAG_DTYPE_BOOLEAN; /* For logical reductions, use boolean dtype */
  else if (op == MAG_OP_ARGMIN || op == MAG_OP_ARGMAX) type = MAG_DTYPE_INT64; /* For argmin/argmax, use int64 dtype */
  if (!keepdim && !plan->out_rank) status = mag_empty_scalar(err, &result, x->ctx, type, mag_tensor_device_id(x));
  else status = mag_empty(err, &result, x->ctx, type, plan->out_rank, plan->out_shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, op, false, &x, 1, &result, 1, &params);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_op_stub_unary(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_opcode_t op,
  mag_tensor_t *x,
  const mag_op_params_t *params,
  bool inplace
) {
  *out_result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, op, &x, 0);
  if (mag_iserr(status)) return status;
  mag_tensor_t *result = NULL;
  if (inplace) {
    result = x;
    mag_rc_incref(result);
    status = mag_check_inplace_grad_ok(err, x);
    if (mag_iserr(status)) return status;
  } else {
    status = mag_empty_like(err, &result, x); /* Allocate a new tensor for the result */
    if (mag_iserr(status)) return status;
  }
  status = mag_dispatch(err, op, inplace, &x, 1, &result, 1, params);
  if (mag_iserr(status)) {
    mag_tensor_decref(result);
    return status;
  }
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_op_stub_cu(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_opcode_t op,
  const char *ext,
  mag_tensor_t *x,
  int64_t dim
) {
  *out_result = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_ERR_PARAM, "cu%s: input tensor must not be NULL.", ext);
  if (mag_unlikely(!(x->coords.rank > 0)))
      return mag_set_error(err, MAG_ERR_RANK, "cu%s: requires a tensor with rank > 0.", ext);
  mag_norm_axis(&dim, x->coords.rank);
  if (mag_unlikely(!(dim >= 0 && dim < x->coords.rank)))
      return mag_set_error(err, MAG_ERR_DIM, "cu%s: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", ext, dim, x->coords.rank);
  mag_op_params_t params = {
    .cumu = {.dim = dim}
  };
  return mag_op_stub_unary(err, out_result, op, x, &params, false);
}

mag_status_t mag_op_stub_cu_ex(
  mag_error_t *err,
  mag_tensor_t **out_values,
  mag_tensor_t **out_indices,
  mag_opcode_t op,
  const char *ext,
  mag_tensor_t *x,
  int64_t dim
) {
  *out_values = NULL;
  *out_indices = NULL;
  if (mag_unlikely(!(x != NULL)))
    return mag_set_error(err, MAG_ERR_PARAM, "cu%s: input tensor must not be NULL.", ext);
  if (mag_unlikely(!(x->coords.rank > 0)))
    return mag_set_error(err, MAG_ERR_RANK, "cu%s: requires a tensor with rank > 0.", ext);
  mag_norm_axis(&dim, x->coords.rank);
  if (mag_unlikely(!(dim >= 0 && dim < x->coords.rank)))
    return mag_set_error(err, MAG_ERR_DIM, "cu%s: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", ext, dim, x->coords.rank);
  mag_tensor_t *values = NULL;
  mag_tensor_t *indices = NULL;
  mag_status_t status = mag_empty_like(err, &values, x);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &indices, x->ctx, MAG_DTYPE_INT64, x->coords.rank, x->coords.shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) {
    mag_tensor_decref(values);
    return status;
  }
  mag_op_params_t params = {
    .cumu = {.dim = dim}
  };
  status = mag_check_dtype_and_device_compat(err, op, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, op, false, &x, 1, (mag_tensor_t*[2]){values, indices}, 2, &params);
  if (mag_iserr(status)) return status;
  *out_values = values;
  *out_indices = indices;
  return MAG_OK;
}

mag_status_t mag_op_stub_binary(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_opcode_t op,
  mag_tensor_t *x,
  mag_tensor_t *y,
  mag_binop_flags_t flags
) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  mag_context_t *ctx = x->ctx;
  mag_dtype_t prom_type; /* common compute dtype for x,y */
  mag_dtype_t res_type;  /* dtype of 'result' tensor */
  bool x_int = mag_tensor_is_integer_typed(x);
  bool y_int = mag_tensor_is_integer_typed(y);
  if (flags & MAG_BINOP_INPLACE) {
    switch (op) {
      case MAG_OP_DIV: {
        if (mag_unlikely(!(!x_int)))
            return mag_set_error(err, MAG_ERR_PARAM, "binary_op: in-place true division is not allowed on integer tensors (got dtype %s).", mag_type_trait(x->dtype)->name);
      } break;
      case MAG_OP_FLOORDIV: {
        if (mag_unlikely(!(x_int && y_int)))
            return mag_set_error(err, MAG_ERR_PARAM, "binary_op: in-place floor division requires integer tensors, but got dtypes %s and %s.", mag_type_trait(x->dtype)->name, mag_type_trait(y->dtype)->name);
      } break;
      default: { /* Inplace ops must keep x's dtype */
        mag_dtype_t prom;
        bool prom_ok = mag_promote_type(&prom, x->dtype, y->dtype);
        if (mag_unlikely(!(prom_ok && prom == x->dtype)))
            return mag_set_error(err, MAG_ERR_PARAM, "binary_op: in-place '%s' would change the dtype of x from %s to %s.", mag_op_trait(op)->mnemonic, mag_type_trait(x->dtype)->name, mag_type_trait(prom)->name);
      } break;
    }
    prom_type = x->dtype;
    res_type = x->dtype;
  } else if (flags & MAG_BINOP_LOGICAL) { /* Inplace keeps x's dtype, but cast y to x's dtype if needed */
    bool prom_ok = mag_promote_type(&prom_type, x->dtype, y->dtype);
    if (mag_unlikely(!prom_ok))
        return mag_set_error(err, MAG_ERR_PARAM, "binary_op: logical operator '%s' does not support dtypes %s and %s.",
          mag_op_trait(op)->mnemonic,
          mag_type_trait(x->dtype)->name,
          mag_type_trait(y->dtype)->name
        );
    res_type = MAG_DTYPE_BOOLEAN; /* logical ops always yield boolean result */
    mag_assert2(!(flags & MAG_BINOP_INPLACE));
  } else { /* Pure out of place -> full promotion */
    switch (op) {
      case MAG_OP_DIV: { /* Special case for truediv */
        if (x_int && y_int) { /* Integer division always promotes to default float dtype */
          prom_type = res_type = ctx->default_dtype;
        } else {
          bool prom_ok = mag_promote_type(&prom_type, x->dtype, y->dtype);
          if (mag_unlikely(!prom_ok))
              return mag_set_error(err, MAG_ERR_PARAM,
                "binary_op: operator '%s' does not support dtypes %s and %s.",
                mag_op_trait(op)->mnemonic,
                mag_type_trait(x->dtype)->name,
                mag_type_trait(y->dtype)->name
              );
          res_type = prom_type;  /* will be a floating dtype */
        }
      } break;
      case MAG_OP_FLOORDIV: {
        bool prom_ok = mag_promote_type(&prom_type, x->dtype, y->dtype);
        if (mag_unlikely(!prom_ok))
            return mag_set_error(err, MAG_ERR_PARAM, "binary_op: operator '%s' does not support dtypes %s and %s.",
              mag_op_trait(op)->mnemonic,
              mag_type_trait(x->dtype)->name,
              mag_type_trait(y->dtype)->name
            );
        if (x_int && y_int) { /* Integer floor division keeps integer dtype */
          res_type = prom_type;
        } else { /* Non-integer floor division promotes to floating dtype */
          if (!(mag_dtype_bit(prom_type) & MAG_DTYPE_MASK_FP))
            prom_type = ctx->default_dtype;
          res_type = prom_type;
        }
      } break;
      default: {
        bool prom_ok = mag_promote_type(&prom_type, x->dtype, y->dtype);
        if (mag_unlikely(!prom_ok))
            return mag_set_error(err, MAG_ERR_PARAM, "binary_op: operator '%s' does not support dtypes %s and %s.",
              mag_op_trait(op)->mnemonic,
              mag_type_trait(x->dtype)->name,
              mag_type_trait(y->dtype)->name
            );
        res_type = prom_type;
      } break;
    }
  }
  mag_status_t status;
  if (flags & MAG_BINOP_INPLACE) {
    mag_assert2(!(flags & MAG_BINOP_LOGICAL));
    status = mag_check_inplace_grad_ok(err, x);
    if (mag_iserr(status)) return status;
    result = x;
    mag_tensor_incref(result);
  } else {
    int64_t dims[MAG_MAX_DIMS];
    int64_t rank;
    if (mag_unlikely(!mag_coords_broadcast_shape(&x->coords, &y->coords, dims, &rank))) {
      char sx[MAG_FMT_DIM_BUF_SIZE];
      char sy[MAG_FMT_DIM_BUF_SIZE];
      mag_fmt_shape(&sx, &x->coords.shape, x->coords.rank);
      mag_fmt_shape(&sy, &y->coords.shape, y->coords.rank);
      return mag_set_error(err, MAG_ERR_BROADCAST,
        "binary_op: cannot broadcast shapes %s and %s for operator '%s'.\n"
        "    Hint: ensure the shapes are broadcast-compatible.",
        sx, sy, mag_op_trait(op)->mnemonic
      );
    }
    status = rank ? mag_empty(err, &result, x->ctx, res_type, rank, dims, mag_tensor_device_id(x)) : mag_empty_scalar(err, &result, x->ctx, res_type, mag_tensor_device_id(x));
    if (mag_iserr(status)) return status;
  }
  mag_tensor_t *prom_x = x;
  mag_tensor_t *prom_y = y;
  mag_tensor_t *tmp_x = NULL;
  mag_tensor_t *tmp_y = NULL;
  if (x->dtype != prom_type) { /* Cast x only if its dtype != promote_dtype and the op semantics say so */
    status = mag_cast(err, &tmp_x, x, prom_type); /* For inplace, x->dtype == promote_dtype, so this is skipped */
    if (mag_iserr(status)) {
      if (!(flags & MAG_BINOP_INPLACE) && result) mag_tensor_decref(result);
      return status;
    }
    prom_x = tmp_x;
  }
  if (y->dtype != prom_type) { /* Cast y if needed */
    status = mag_cast(err, &tmp_y, y, prom_type);
    if (mag_iserr(status)) {
      if (tmp_x) mag_tensor_decref(tmp_x);
      if (!(flags & MAG_BINOP_INPLACE) && result) mag_tensor_decref(result);
      return status;
    }
    prom_y = tmp_y;
  }
  mag_tensor_t *in[2] = {prom_x, prom_y};
  status = mag_check_dtype_and_device_compat(err, op, in, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, op, flags & MAG_BINOP_INPLACE, in, sizeof(in)/sizeof(*in), &result, 1, NULL);
  if (mag_iserr(status)) return status;
  if (tmp_x) mag_tensor_decref(tmp_x);
  if (tmp_y) mag_tensor_decref(tmp_y);
  *out_result = result;
  return MAG_OK;
}

mag_status_t mag_matmul_verify_shapes(
  mag_error_t *err,
  int64_t *rb,
  int64_t *xb,
  int64_t *yb,
  const mag_tensor_t *x,
  const mag_tensor_t *y
) {
  int64_t kx = x->coords.shape[x->coords.rank-1];
  int64_t ky = y->coords.rank == 1 ? *y->coords.shape : y->coords.rank == 2 && x->coords.rank == 1 ? *y->coords.shape : y->coords.shape[y->coords.rank-2];
  *xb = x->coords.rank > 2 ? x->coords.rank-2 : 0;
  *yb = y->coords.rank > 2 ? y->coords.rank-2 : 0;
  *rb = mag_xmax(*xb, *yb);
  if (kx != ky) {
    char sx[MAG_FMT_DIM_BUF_SIZE];
    char sy[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&sx, &x->coords.shape, x->coords.rank);
    mag_fmt_shape(&sy, &y->coords.shape, y->coords.rank);
    return mag_set_error(err, MAG_ERR_OP,
      "matmul: incompatible shapes %s and %s, "
      "last dim of x (%" PRIi64 ") must match dim -2 of y (%" PRIi64 ").",
      sx, sy, kx, ky
    );
  }
  for (int64_t i=0; i < *rb; ++i) {
    int64_t xd = i < *rb-*xb ? 1 : x->coords.shape[i-(*rb-*xb)];
    int64_t yd = i < *rb-*yb ? 1 : y->coords.shape[i-(*rb-*yb)];
    if (xd != yd && xd != 1 && yd != 1) {
      char sx[MAG_FMT_DIM_BUF_SIZE];
      char sy[MAG_FMT_DIM_BUF_SIZE];
      mag_fmt_shape(&sx, &x->coords.shape, x->coords.rank);
      mag_fmt_shape(&sy, &y->coords.shape, y->coords.rank);
      return mag_set_error(err, MAG_ERR_OP,
        "matmul: batch dim %" PRIi64 " (%" PRIi64 ") of x cannot broadcast with y (%" PRIi64 ") for shapes %s and %s.",
        i, xd, yd, sx, sy
      );
    }
  }
  return MAG_OK;
}

mag_status_t mag_matmul_alloc_res(
  mag_error_t *err,
  mag_tensor_t **res,
  int64_t rb,
  int64_t *xb,
  int64_t *yb,
  mag_tensor_t *x,
  mag_tensor_t *y
) {
  mag_matmul_type_t type = mag_matmul_type_detect(x, y);
  switch (type) {
    case MAG_MATMUL_TYPE_INVALID: return mag_set_error(err, MAG_ERR_OP, "matmul: unsupported tensor shapes.");
    case MAG_MATMUL_TYPE_DOT: return mag_empty_scalar(err, res, x->ctx, x->dtype, mag_tensor_device_id(x));
    case MAG_MATMUL_TYPE_GEMV_VEC_MAT: {
        int64_t N = y->coords.shape[1];
      return mag_empty(err, res, x->ctx, x->dtype, 1, (int64_t[1]){N}, mag_tensor_device_id(x));
    } case MAG_MATMUL_TYPE_GEMV_MAT_VEC: {
        int64_t M = x->coords.shape[0];
      return mag_empty(err, res, x->ctx, x->dtype, 1, (int64_t[1]){M}, mag_tensor_device_id(x));
    } case MAG_MATMUL_TYPE_GEMM: {
        int64_t M = x->coords.shape[0];
        int64_t N = y->coords.shape[1];
      return mag_empty(err, res, x->ctx, x->dtype, 2, (int64_t[2]){M, N}, mag_tensor_device_id(x));
    } case MAG_MATMUL_TYPE_BMM_DOT:
      case MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT:
      case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC:
      case MAG_MATMUL_TYPE_BMM_GEMM: {
        *xb = x->coords.rank-2;
        *yb = y->coords.rank-2;
        int64_t shape[MAG_MAX_DIMS] = {0};
        for (int64_t i=0; i < rb; ++i) {
          int64_t da = i < rb-*xb ? 1 : x->coords.shape[i-(rb-*xb)];
          int64_t db = i < rb-*yb ? 1 : y->coords.shape[i-(rb-*yb)];
          shape[i] = da > db ? da : db;
        }
        shape[rb] = x->coords.shape[x->coords.rank-2];
        shape[rb+1] = y->coords.shape[y->coords.rank-1];
        return mag_empty(err, res, x->ctx, x->dtype, rb+2, shape, mag_tensor_device_id(x));
    } default: return mag_set_error(err, MAG_ERR_OP, "matmul: invalid BMM matmul type '%s'.", mag_matmul_type_name(type));
  }
}

void MAG_COLDPROC mag_dbg_trace_op_ir(
  mag_opcode_t op,
  bool inplace,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out
) {
  const mag_op_traits_t *meta = mag_op_trait(op);
  const mag_device_id_t *dvc = in && num_in ? &in[0]->storage->device->id : &out[0]->storage->device->id;
  bool cont = true;
  for (uint32_t i=0; i < num_in;  ++i) cont &= mag_tensor_is_contiguous(in[i]);
  for (uint32_t i=0; i < num_out; ++i) cont &= mag_tensor_is_contiguous(out[i]);
  char opcode[64];
  snprintf(opcode, sizeof(opcode), "%s", meta->mnemonic);
  for (char *p = opcode; *p; ++p) if (*p >= 'A' && *p <= 'Z') *p |= 0x20;
  char dvcname[64];
  snprintf(dvcname, sizeof(dvcname), "%s", mag_backend_type_to_str(dvc->type));
  for (char *p = dvcname; *p; ++p) if (*p >= 'A' && *p <= 'Z') *p |= 0x20;
  const mag_tensor_t *tin = num_in ? in[0] : NULL;
  const mag_tensor_t *tout = num_out ? out[0] : NULL;
  int64_t rank = tin ? tin->coords.rank : tout->coords.rank;
  printf("%s.%s:%u.%s%s.", opcode, dvcname, dvc->device_ordinal, cont ? "cont" : "stri", inplace ? ".inl" : "");
  if (op == MAG_OP_CAST && num_in && num_out) {
    printf("%s.%s.", mag_type_trait(out[0]->dtype)->short_name, mag_type_trait(in[0]->dtype)->short_name);
  } else if (num_out == 1) {
    printf("%s.", mag_type_trait(out[0]->dtype)->short_name);
  } else {
    putchar('(');
    for (uint32_t i=0; i < num_out; ++i) {
      fputs(mag_type_trait(out[i]->dtype)->short_name, stdout);
      if (i+1 < num_out) putchar(',');
    }
    printf(").");
  }
  printf("%zud", (size_t)rank);
  if (rank > 0) putchar('.');
  if (tin && rank > 0) {
    for (int64_t i=0; i < tin->coords.rank; ++i) {
      printf("%" PRIi64, tin->coords.shape[i]);
      if (i+1 < tin->coords.rank) putchar('x');
    }
  }
  if (tout && rank > 0) {
    putchar('.');
    for (int64_t i=0; i < tout->coords.rank; ++i) {
      printf("%" PRIi64, tout->coords.shape[i]);
      if (i+1 < tout->coords.rank) putchar('x');
    }
  }
  putchar('\n');
}

mag_status_t mag_check_dtype_and_device_compat(mag_error_t *err, mag_opcode_t op, mag_tensor_t **inputs, uint32_t num_in_dyn) {
  const mag_op_traits_t *meta = mag_op_trait(op);
  uint32_t n;
  if (meta->in == MAG_OP_INOUT_DYN) {
    n = num_in_dyn;
    if (mag_unlikely(!(inputs && n > 0)))
        return mag_set_error(err, MAG_ERR_PARAM, "op_validate: operator '%s' requires a non-empty input tensor list.", meta->mnemonic);
  } else {
    n = meta->in;
    (void)num_in_dyn;
  }
  mag_device_id_t dev0 = {0};
  for (uint32_t i=0; i < n; ++i) { /* Check dtype support and that all inputs share one device. */
    bool supported = meta->dtype_mask & mag_dtype_bit(inputs[i]->dtype);
    if (mag_unlikely(!supported)) {
      const char *dtype = mag_type_trait(inputs[i]->dtype)->name;
      return mag_set_error(err, MAG_ERR_PARAM,
        "op_validate: operator '%s' does not support dtype '%s'.\n"
        "    Hint: cast the tensor to a supported dtype.",
        meta->mnemonic, dtype
      );
    }
    if (i == 0) {
      dev0 = mag_tensor_device_id(inputs[0]);
    } else {
      mag_device_id_t devi = mag_tensor_device_id(inputs[i]);
      if (mag_unlikely(devi.type != dev0.type || devi.device_ordinal != dev0.device_ordinal)) {
        char b0[32], bi[32];
        mag_device_id_to_str(dev0, &b0);
        mag_device_id_to_str(devi, &bi);
        return mag_set_error(err, MAG_ERR_PARAM,
          "op_validate: all input tensors for operator '%s' must be on the same device, but found '%s' and '%s'.\n"
          "    Hint: transfer tensors to a single device before calling this operator.",
          meta->mnemonic, b0, bi
        );
      }
    }
  }
  if (op == MAG_OP_GATHER || op == MAG_OP_EMBEDDING) {
    if (mag_unlikely(!(inputs[1]->dtype == MAG_DTYPE_INT64)))
        return mag_set_error(err, MAG_ERR_PARAM,
          "op_validate: index tensor for operator '%s' must have dtype int64, but got '%s'.\n"
          "    Hint: cast the indices to int64.",
          meta->mnemonic, mag_type_trait(inputs[1]->dtype)->name
        );
    if (op == MAG_OP_EMBEDDING) {
      mag_dtype_mask_t fp_mask = MAG_DTYPE_MASK_FP;
      if (mag_unlikely(!((fp_mask & mag_dtype_bit(inputs[0]->dtype)) != 0)))
          return mag_set_error(err, MAG_ERR_PARAM,
            "op_validate: weight tensor for 'embedding' must have a floating-point dtype, but got '%s'.",
            mag_type_trait(inputs[0]->dtype)->name
          );
    }
    return MAG_OK;
  }
  if (op == MAG_OP_MASKED_FILL) {
    if (mag_unlikely(!(inputs[1]->dtype == MAG_DTYPE_BOOLEAN)))
        return mag_set_error(err, MAG_ERR_PARAM,
          "op_validate: mask tensor for operator '%s' must have dtype bool, but got '%s'.\n"
          "    Hint: cast the mask to bool.",
          meta->mnemonic, mag_type_trait(inputs[1]->dtype)->name
        );
    return MAG_OK;
  }
  if (mag_unlikely(meta->in == 2 && n == 2 && inputs[0]->dtype != inputs[1]->dtype)) { /* For binary operators, check that both inputs have the same data type. */
    const char *dtype_x = mag_type_trait(inputs[0]->dtype)->name;
    const char *dtype_y = mag_type_trait(inputs[1]->dtype)->name;
    return mag_set_error(err, MAG_ERR_PARAM,
      "op_validate: input dtypes for operator '%s' must match, but got '%s' and '%s'.\n"
      "    Hint: cast both inputs to the same dtype.",
      meta->mnemonic, dtype_x, dtype_y
    );
  }
  return MAG_OK;
}

mag_status_t mag_check_inplace_grad_ok(mag_error_t *err, const mag_tensor_t *result) {
  if (mag_unlikely((result->ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && (result->flags & MAG_TFLAG_REQUIRES_GRAD)))
    return mag_set_error(err, MAG_ERR_PARAM,
      "op_validate: in-place operations are not allowed on tensors that require gradients.\n"
      "    Hint: disable gradient tracking or use the out-of-place variant."
    );
  return MAG_OK;
}
