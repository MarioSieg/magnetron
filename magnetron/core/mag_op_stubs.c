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

#include <string.h>

#include "mag_alloc.h"

mag_scalar_t mag_scalar_from_f64(double value) {
  return (mag_scalar_t){.type = MAG_SCALAR_TYPE_F64, .value.f64 = value};
}

mag_scalar_t mag_scalar_from_i64(int64_t value) {
  return (mag_scalar_t){.type = MAG_SCALAR_TYPE_I64, .value.i64 = value};
}

mag_scalar_t mag_scalar_from_u64(uint64_t value) {
  return (mag_scalar_t){.type = MAG_SCALAR_TYPE_U64, .value.u64 = value};
}

bool mag_scalar_is_f64(mag_scalar_t s) { return s.type == MAG_SCALAR_TYPE_F64; }

bool mag_scalar_is_i64(mag_scalar_t s) { return s.type == MAG_SCALAR_TYPE_I64; }

bool mag_scalar_is_u64(mag_scalar_t s) { return s.type == MAG_SCALAR_TYPE_U64; }

double mag_scalar_as_f64(mag_scalar_t s) {
  switch (s.type) {
    case MAG_SCALAR_TYPE_F64: return s.value.f64;
    case MAG_SCALAR_TYPE_I64: return (double)s.value.i64;
    case MAG_SCALAR_TYPE_U64: return (double)s.value.u64;
    default: mag_panic("scalar: invalid type tag %d.", s.type);
  }
}

int64_t mag_scalar_as_i64(mag_scalar_t s) {
  switch (s.type) {
    case MAG_SCALAR_TYPE_I64: return s.value.i64;
    case MAG_SCALAR_TYPE_U64: return (int64_t)s.value.u64;
    case MAG_SCALAR_TYPE_F64: return (int64_t)s.value.f64;
    default: mag_panic("scalar: invalid type tag %d.", s.type);
  }
}

uint64_t mag_scalar_as_u64(mag_scalar_t s) {
  switch (s.type) {
    case MAG_SCALAR_TYPE_U64: return s.value.u64;
    case MAG_SCALAR_TYPE_I64: return (uint64_t)s.value.i64;
    case MAG_SCALAR_TYPE_F64: return (uint64_t)s.value.f64;
    default: mag_panic("scalar: invalid type tag %d.", s.type);
  }
}

static mag_op_attr_t mag_scalar_to_op_attr(mag_dtype_t dtype, mag_scalar_t x) {
  mag_dtype_mask_t dtb = mag_dtype_bit(dtype);
  if (dtb & MAG_DTYPE_MASK_FP) return mag_op_attr_float64(mag_scalar_as_f64(x));
  if (dtb & MAG_DTYPE_MASK_SINT) return mag_op_attr_int64(mag_scalar_as_i64(x));
  if (dtb & MAG_DTYPE_MASK_UINT || dtype == MAG_DTYPE_BOOLEAN) return mag_op_attr_uint64(mag_scalar_as_u64(x));
  mag_panic("scalar: unsupported dtype '%s' for conversion.", mag_type_trait(dtype)->name);
}

static bool mag_scalar_same_type(mag_scalar_t a, mag_scalar_t b) {
  return a.type == b.type;
}

static void mag_norm_axis(int64_t *ax, int64_t ra) {
  if (*ax < 0) *ax += ra;
}

static bool mag_op_requires_op_params(mag_opcode_t op) { /* Returns true if the op requires any op params and thus requires validation of them. */
  const mag_op_traits_t *meta = mag_op_trait(op);
  for (int i=0; i < MAG_MAX_OP_PARAMS; ++i) {
    if (meta->op_attr_types[i] != MAG_OP_ATTR_TYPE_EMPTY) {
      return true;
    }
  }
  return false;
}

static void mag_assert_correct_op_data(
  mag_opcode_t op,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out,
  const mag_op_attr_t *op_params,
  uint32_t num_op_params
) {
  mag_assert(op != MAG_OP_NOP, "op_validate: invalid opcode %d.", op);
  const mag_op_traits_t *meta = mag_op_trait(op);

  /* Check input/output tensors */
  if (meta->in) mag_assert(in != NULL, "op_validate: input tensors for operator '%s' are NULL.", meta->mnemonic);
  if (meta->out) mag_assert(out != NULL, "op_validate: output tensors for operator '%s' are NULL.", meta->mnemonic);
  if (meta->in != MAG_OP_INOUT_DYN) {
    mag_assert(meta->in == num_in, "op_validate: operator '%s' expected %u input tensors but got %u.", meta->mnemonic, meta->in, num_in);
    mag_assert(meta->out == num_out, "op_validate: operator '%s' expected %u output tensors but got %u.", meta->mnemonic, meta->out, num_out);
  }
  for (uint32_t i=0; i < num_in; ++i)
    mag_assert(in[i] != NULL, "op_validate: input tensor %u for operator '%s' is NULL.", i, meta->mnemonic);
  for (uint32_t i=0; i < num_out; ++i)
    mag_assert(out[i] != NULL, "op_validate: output tensor %u for operator '%s' is NULL.", i, meta->mnemonic);

  /* Check op params if required */
  if (mag_op_requires_op_params(op)) {
    mag_assert(op_params != NULL, "op_validate: operator '%s' requires parameters but none were provided.", meta->mnemonic);
    mag_assert(num_op_params <= MAG_MAX_OP_PARAMS, "op_validate: operator '%s' has too many parameters (%u > %u).", meta->mnemonic, num_op_params, MAG_MAX_OP_PARAMS);
    for (uint32_t i=0; i < num_op_params; ++i) {
      if (meta->op_attr_types[i] != MAG_OP_ATTR_TYPE_EMPTY) { /* Only check for type equality if op param is required */
        mag_assert(op_params[i].tag == meta->op_attr_types[i], "op_validate: operator '%s' got invalid parameter type %d (expected %d).", meta->mnemonic, op_params[i].tag, meta->op_attr_types[i]);
      }
    }
  }
}

extern void mag_tensor_detach_inplace(mag_tensor_t *target);
static void mag_bump_version(mag_tensor_t *t) {
  if (t->flags & MAG_TFLAG_IS_VIEW) /* If this is a view, bump the version of the base tensor */
    t = t->view_meta->base;
  ++t->version;
}

static mag_status_t mag_tensor_strided_view(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *base) {
  return mag_as_strided(err, out_result, base->ctx, base, base->coords.rank, base->coords.shape, base->coords.strides, base->storage_offset);
}

static void MAG_COLDPROC mag_dbg_trace_op_ir(mag_opcode_t op, bool inplace, mag_tensor_t **in,  uint32_t num_in, mag_tensor_t **out, uint32_t num_out) {
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

/* Execute an operator on the active compute device and return result tensor. */
static mag_status_t mag_dispatch(mag_error_t *err, mag_opcode_t op, bool inplace, const mag_op_attr_registry_t *layout, mag_tensor_t **in, uint32_t num_in, mag_tensor_t **out, uint32_t num_out) {
  const mag_op_traits_t *meta = mag_op_trait(op);
  mag_assert2((in && num_in) || (out && num_out));
  mag_assert2(op != MAG_OP_NOP);
#if 0 /* Debug: print dispatched ops */
  mag_dbg_trace_op_ir(op, inplace, in, num_in, out, num_out);
#endif
  mag_context_t *ctx = in ? (*in)->ctx : (*out)->ctx;
  mag_device_t *device = in ? (*in)->storage->device : (*out)->storage->device;
  const mag_op_attr_t *params = layout ? layout->slots : NULL;
  uint32_t num_params = layout ? layout->count : 0;
  mag_assert_correct_op_data(op, in, num_in, out, num_out, params, num_params);
  mag_status_t status;
  if (!!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && meta->backward) {
    for (uint32_t i=0; i < num_out; ++i) {
      mag_tensor_t *r = out[i];
      mag_au_state_t *au = mag_au_state_lazy_alloc(&r->au_state, r->ctx);
      if (mag_unlikely(!au))
        return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "dispatch: failed to allocate autodiff state for gradient recording.");
      au->op = op;
      for (uint32_t j=0; j < num_in; ++j) {
        mag_tensor_t *input = in[j];
        au->op_inputs[j] = input;
        if (input->flags & MAG_TFLAG_REQUIRES_GRAD && !(r->flags & MAG_TFLAG_REQUIRES_GRAD)) {
          status = mag_tensor_set_requires_grad(err, r, true);
          if (mag_iserr(status)) return status;
        }
        mag_rc_incref(input);
      }
      if (params)
        memcpy(au->op_attrs, params, num_params * sizeof(*params));
    }
  }
  mag_command_t cmd = {
    .op = op,
    .in = in,
    .out = out,
    .num_in = num_in,
    .num_out = num_out,
  };
  if (params) memcpy(cmd.attrs, params, num_params*sizeof(*params));
  mag_status_t (*submit)(mag_device_t *, mag_error_t *, const mag_command_t *) = device->submit;
  mag_status_t stat = (*submit)(device, err, &cmd);
  for (uint32_t i=0; i < num_out; ++i)
    if (inplace) mag_bump_version(out[i]);  /* Result aliases the modified storage */
  ++ctx->telemetry.ops_dispatched;
  return stat;
}

/* num_in_dyn: for MAG_OP_INOUT_DYN (e.g. CAT) pass the runtime input count else 0 */
static mag_status_t mag_check_dtype_and_device_compat(mag_error_t *err, mag_opcode_t op, mag_tensor_t **inputs, uint32_t num_in_dyn) {
  const mag_op_traits_t *meta = mag_op_trait(op);
  uint32_t n;
  if (meta->in == MAG_OP_INOUT_DYN) {
    n = num_in_dyn;
    if (mag_unlikely(!(inputs && n > 0)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "op_validate: operator '%s' requires a non-empty input tensor list.", meta->mnemonic);
  } else {
    n = meta->in;
    (void)num_in_dyn;
  }
  mag_device_id_t dev0 = {0};
  for (uint32_t i=0; i < n; ++i) { /* Check dtype support and that all inputs share one device. */
    bool supported = meta->dtype_mask & mag_dtype_bit(inputs[i]->dtype);
    if (mag_unlikely(!supported)) {
      const char *dtype = mag_type_trait(inputs[i]->dtype)->name;
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
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
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
          "op_validate: all input tensors for operator '%s' must be on the same device, but found '%s' and '%s'.\n"
          "    Hint: transfer tensors to a single device before calling this operator.",
          meta->mnemonic, b0, bi
        );
      }
    }
  }
  if (op == MAG_OP_GATHER || op == MAG_OP_EMBEDDING) {
    if (mag_unlikely(!(inputs[1]->dtype == MAG_DTYPE_INT64)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
          "op_validate: index tensor for operator '%s' must have dtype int64, but got '%s'.\n"
          "    Hint: cast the indices to int64.",
          meta->mnemonic, mag_type_trait(inputs[1]->dtype)->name
        );
    if (op == MAG_OP_EMBEDDING) {
      mag_dtype_mask_t fp_mask = MAG_DTYPE_MASK_FP;
      if (mag_unlikely(!((fp_mask & mag_dtype_bit(inputs[0]->dtype)) != 0)))
          return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
            "op_validate: weight tensor for 'embedding' must have a floating-point dtype, but got '%s'.",
            mag_type_trait(inputs[0]->dtype)->name
          );
    }
    return MAG_STATUS_OK;
  }
  if (mag_unlikely(meta->in == 2 && n == 2 && inputs[0]->dtype != inputs[1]->dtype)) { /* For binary operators, check that both inputs have the same data type. */
    const char *dtype_x = mag_type_trait(inputs[0]->dtype)->name;
    const char *dtype_y = mag_type_trait(inputs[1]->dtype)->name;
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
      "op_validate: input dtypes for operator '%s' must match, but got '%s' and '%s'.\n"
      "    Hint: cast both inputs to the same dtype.",
      meta->mnemonic, dtype_x, dtype_y
    );
  }
  return MAG_STATUS_OK;
}

static mag_status_t mag_check_inplace_grad_ok(mag_error_t *err, const mag_tensor_t *result) {
  if (mag_unlikely((result->ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && (result->flags & MAG_TFLAG_REQUIRES_GRAD)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
      "op_validate: in-place operations are not allowed on tensors that require gradients.\n"
      "    Hint: disable gradient tracking or use the out-of-place variant."
    );
  return MAG_STATUS_OK;
}

mag_status_t mag_empty_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like) {
  return mag_empty(err, out_result, like->ctx, like->dtype, like->coords.rank, like->coords.shape, mag_tensor_device_id(like));
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
  return mag_full(err, out_result, ctx, type, rank, shape, mag_scalar_from_u64(0), device);
}

mag_status_t mag_zeros_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like) {
  return mag_full_like(err, out_result, like, mag_scalar_from_u64(0));
}

mag_status_t mag_ones(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device) {
  return mag_full(err, out_result, ctx, type, rank, shape, mag_scalar_from_u64(1), device);
}

mag_status_t mag_ones_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like) {
  return mag_full_like(err, out_result, like, mag_scalar_from_u64(1));
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

mag_status_t mag_bernoulli(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, int64_t rank, const int64_t *shape, mag_scalar_t p, mag_device_id_t device) {
  mag_status_t status = mag_empty(err, out_result, ctx, MAG_DTYPE_BOOLEAN, rank, shape, device);
  if (mag_iserr(status)) return status;
  return mag_bernoulli_(err, *out_result, p);
}

mag_status_t mag_bernoulli_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t p) {
  mag_status_t status = mag_empty(err, out_result, like->ctx, MAG_DTYPE_BOOLEAN, like->coords.rank, like->coords.shape, mag_tensor_device_id(like));
  if (mag_iserr(status)) return status;
  return mag_bernoulli_(err, *out_result, p);
}

static bool mag_arange_numel_int(int64_t start, int64_t stop, int64_t step, int64_t *numel) {
  if (step == 0) {
    *numel = 0;
    return false;
  }
  int64_t delta = stop - start;
  if (step > 0) {
    if (delta <= 0) {
      *numel = 0;
      return true;
    }
    *numel = (delta + step - 1)/step;
    return true;
  }
  if (delta >= 0) {
    *numel = 0;
    return true;
  }
  int64_t step_pos = -step;
  int64_t diff_pos = -delta;
  *numel = (diff_pos + step_pos - 1)/step_pos;
  return true;
}

static bool mag_arange_numel_float(double start, double end, double step, int64_t *numel) {
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

mag_status_t mag_arange(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t start, mag_scalar_t end, mag_scalar_t step, mag_device_id_t device) {
  *out_result = NULL;
  if (mag_unlikely(!(mag_scalar_same_type(start, end) && mag_scalar_same_type(start, step))))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "arange: start, end and step must have the same scalar type.");
  if (mag_unlikely(!(mag_dtype_bit(type) & MAG_DTYPE_MASK_NUMERIC)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "arange: requires a numeric dtype.");
  mag_tensor_t *result;
  int64_t numel = 0;
  bool ok = false;
  if (mag_dtype_bit(type) & MAG_DTYPE_MASK_INTEGER) ok = mag_arange_numel_int(mag_scalar_as_i64(start), mag_scalar_as_i64(end), mag_scalar_as_i64(step), &numel);
  else ok = mag_arange_numel_float(mag_scalar_as_f64(start), mag_scalar_as_f64(end), mag_scalar_as_f64(step), &numel);
  if (mag_unlikely(!ok) || numel <= 0)
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "arange: invalid start, end or step (produces an empty or invalid range).");
  mag_status_t status = mag_empty(err, &result, ctx, type, 1, &numel, device);
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_float64(mag_scalar_as_f64(start))); /* TODO: this looses information for int64/uint64 ranges that exceed f64 precision */
  mag_op_attr_registry_insert(&layout, mag_op_attr_float64(mag_scalar_as_f64(step)));
  status = mag_check_dtype_and_device_compat(err, MAG_OP_ARANGE, NULL, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_ARANGE, false, &layout, NULL, 0, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_eye(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t n, int64_t m, mag_device_id_t device) {
  *out_result = NULL;
  if (mag_unlikely(!(n >= 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "eye: n must be >= 0, but got %" PRIi64 ".", n);
  if (mag_unlikely(!(m >= 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "eye: m must be >= 0, but got %" PRIi64 ".", m);
  mag_tensor_t *result = NULL;
  int64_t shape[2] = {n, m};
  mag_status_t status = mag_empty(err, &result, ctx, type, 2, shape, device);
  if (mag_iserr(status)) return status;
  if (n > 0 && m > 0) {
    status = mag_check_dtype_and_device_compat(err, MAG_OP_EYE, NULL, 0);
    if (mag_iserr(status)) return status;
    status = mag_dispatch(err, MAG_OP_EYE, false, NULL, NULL, 0, &result, 1);
    if (mag_iserr(status)) return status;
  }
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_linspace(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t start, mag_scalar_t end, int64_t steps, mag_device_id_t device) {
  *out_result = NULL;
  if (mag_unlikely(!(steps > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "linspace: steps must be > 0, but got %" PRIi64 ".", steps);
  if (mag_unlikely(!mag_scalar_same_type(start, end)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "linspace: start and end must have the same scalar type.");
  if (mag_unlikely(!(mag_dtype_bit(type) & MAG_DTYPE_MASK_NUMERIC)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "linspace: requires a numeric dtype.");
  if (steps == 1) return mag_full(err, out_result, ctx, type, 1, &steps, start, device);
  mag_status_t status = MAG_STATUS_OK;
  mag_tensor_t *idx = NULL, *scale = NULL, *start_t = NULL, *tmp = NULL, *result = NULL;
  status = mag_arange(err, &idx, ctx, type, mag_scalar_from_i64(0), mag_scalar_from_i64(steps), mag_scalar_from_i64(1), device);
  if (mag_iserr(status))
      goto cleanup;
  status = mag_full(err, &scale, ctx, type, 1, &steps, mag_scalar_from_f64((mag_scalar_as_f64(end) - mag_scalar_as_f64(start))/(double)(steps - 1)), device);
  if (mag_iserr(status))
      goto cleanup;
  status = mag_full(err, &start_t, ctx, type, 1, &steps, start, device);
  if (mag_iserr(status))
      goto cleanup;
  status = mag_mul(err, &tmp, idx, scale);
  if (mag_iserr(status))
      goto cleanup;
  status = mag_add(err, &result, tmp, start_t);
  if (mag_iserr(status))
      goto cleanup;
  *out_result = result;
  result = NULL;
cleanup:
  if (result)  mag_tensor_decref(result);
  if (tmp)     mag_tensor_decref(tmp);
  if (start_t) mag_tensor_decref(start_t);
  if (scale)   mag_tensor_decref(scale);
  if (idx)     mag_tensor_decref(idx);
  return status;
}

mag_status_t mag_meshgrid(mag_error_t *err, mag_tensor_t **out_results, mag_tensor_t **tensors, size_t count) {
  if (mag_unlikely(!(out_results != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "meshgrid: out_results must not be NULL.");
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "meshgrid: tensors must not be NULL.");
  if (mag_unlikely(!(count > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "meshgrid: expected at least one tensor.");
  if (mag_unlikely(!(count <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "meshgrid: tensor count %zu exceeds maximum rank %d.", count, MAG_MAX_DIMS);
  for (size_t i=0; i < count; ++i) {
    out_results[i] = NULL;
    if (mag_unlikely(!(tensors[i] != NULL)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "meshgrid: tensors[%zu] must not be NULL.", i);
    if (mag_unlikely(!(tensors[i]->coords.rank == 1)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "meshgrid: tensors[%zu] must be 1-D, but got rank %" PRIi64 ".", i, tensors[i]->coords.rank);
  }
  int64_t full_shape[MAG_MAX_DIMS];
  for (size_t i=0; i < count; ++i)
    full_shape[i] = tensors[i]->coords.shape[0];
  size_t filled = 0;
  mag_status_t status = MAG_STATUS_OK;
  for (size_t i=0; i < count; ++i) {
    int64_t view_shape[MAG_MAX_DIMS];
    for (size_t dim=0; dim < count; ++dim)
      view_shape[dim] = 1;
    view_shape[i] = tensors[i]->coords.shape[0];
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
  return MAG_STATUS_OK;
}

mag_status_t mag_rand_perm(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t n, mag_device_id_t device) {
  *out_result = NULL;
  if (mag_unlikely(!(mag_dtype_bit(type) & MAG_DTYPE_MASK_INTEGER)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "rand_perm: requires an integer dtype.");
  mag_tensor_t *result;
  mag_status_t status = mag_empty(err, &result, ctx, type, 1, &n, device);
  if (mag_iserr(status)) return status;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_RAND_PERM, NULL, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_RAND_PERM, false, NULL, NULL, 0, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_clone(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x) {
  *out_result = NULL;
  mag_tensor_t *result;
  mag_status_t status = mag_empty_like(err, &result, x);
  if (mag_iserr(status)) return status;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_CLONE, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_CLONE, false, NULL, &x, 1, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_cast(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_dtype_t dst_type) {
  if (x->dtype == dst_type) return mag_clone(err, out_result, x); /* If dtypes match, we just clone */
  *out_result = NULL;
  mag_tensor_t *result;
  mag_status_t status = mag_empty(err, &result, x->ctx, dst_type, x->coords.rank, x->coords.shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_CAST, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_CAST, false, NULL, &x, 1, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_transfer(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_device_id_t device) {
  *out_result = NULL;
  mag_device_id_t src_id = mag_tensor_device_id(x);
  if (src_id.type == device.type && src_id.device_ordinal == device.device_ordinal) { /* If already on same device, bump refcount and do nothing */
    mag_rc_incref(x);
    *out_result = x;
    return MAG_STATUS_OK;
  }
  mag_status_t status = MAG_STATUS_OK;
  mag_tensor_t *xc = NULL;
  mag_tensor_t *out = NULL;
  status = mag_contiguous(err, &xc, x);
  if (mag_iserr(status))
      goto cleanup;
  status = mag_empty(err, &out, x->ctx, xc->dtype, xc->coords.rank, xc->coords.shape, device);
  if (mag_iserr(status))
      goto cleanup;
  {
    mag_device_t *src_dvc = xc->storage->device;
    mag_device_t *dst_dvc = out->storage->device;
    bool src_hv = xc->storage->flags&MAG_STORAGE_FLAG_HOST_VISIBLE;
    bool dst_hv = out->storage->flags&MAG_STORAGE_FLAG_HOST_VISIBLE;
    if (src_hv && dst_hv) {
      size_t nb = mag_tensor_numbytes(xc);
      if (mag_unlikely(!(nb == mag_tensor_numbytes(out)))) {
        status = mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "transfer: source and destination tensor sizes do not match.");
        goto cleanup;
      }
      if (mag_unlikely(!(mag_tensor_is_contiguous(xc) && mag_tensor_is_contiguous(out)))) {
        status = mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "transfer: both tensors must be contiguous.");
        goto cleanup;
      }
      memcpy((void *)mag_tensor_data_ptr_mut(out), (const void *)mag_tensor_data_ptr(xc), nb);
      mag_tensor_decref(xc);
      *out_result = out;
      return MAG_STATUS_OK;
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
      status = mag_set_error(err, MAG_STATUS_ERR_INVALID_STATE, "transfer: target device does not implement tensor transfer.");
      goto cleanup;
    }
    status = (*exec->transfer)(exec, err, dir, xc, out);
    if (mag_iserr(status))
        goto cleanup;
  }
  mag_tensor_decref(xc);
  *out_result = out;
  return MAG_STATUS_OK;
cleanup:
  if (out) mag_tensor_decref(out);
  if (xc)  mag_tensor_decref(xc);
  return status;
}

mag_status_t mag_view(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  if (mag_unlikely(!(rank >= 0 && rank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "view: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  mag_status_t status;
  if (rank == 0) {
    if (mag_unlikely(!(x->numel == 1)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "view: rank-0 view is only allowed on tensors with a single element, but got %" PRIi64 " elements.", x->numel);
    status = mag_as_strided(err, &result, x->ctx, x, 0, NULL, NULL, x->storage_offset);
    if (mag_iserr(status)) return status;
  } else {
    if (mag_unlikely(!(dims != NULL)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "view: dims must not be NULL when rank > 0.");
    int64_t oshape[MAG_MAX_DIMS] = {0};
    memcpy(oshape, dims, rank*sizeof(*dims));
    int64_t shape[MAG_MAX_DIMS];
    status = mag_infer_missing_dim(err, &shape, oshape, rank, x->numel);
    if (mag_iserr(status)) return status;
    int64_t strides[MAG_MAX_DIMS];
    if (rank == x->coords.rank && !memcmp(shape, x->coords.shape, rank*sizeof(*shape))) { /* Stride strategy: same shape as base */
      memcpy(strides, x->coords.strides, rank*sizeof(*shape));
    } else if (rank == x->coords.rank+1 && shape[rank-2]*shape[rank-1] == x->coords.shape[x->coords.rank-1]) { /* Stride strategy: last dim only */
      memcpy(strides, x->coords.strides, (rank-2)*sizeof(*strides));
      strides[rank-2] = x->coords.strides[x->coords.rank-1]*shape[rank-1];
      strides[rank-1] = x->coords.strides[x->coords.rank-1];
    } else if (mag_tensor_is_contiguous(x)) { /* Stride strategy: contiguous row-major */
      strides[rank-1] = 1;
      for (int64_t i=rank-2; i >= 0; --i) {
        if (mag_unlikely(mag_mulov64(shape[i+1], strides[i+1], strides+i)))
            return mag_set_error(err, MAG_STATUS_ERR_DIM_OVERFLOW, "view: stride computation overflowed at dim %" PRIi64 ".", i);
      }
    } else { /* Stride strategy: solve generic strides */
      status = mag_solve_view_strides(err, &strides, x->coords.shape, x->coords.strides, x->coords.rank, shape, rank);
      if (mag_iserr(status)) return status;
    }
    status = mag_as_strided(err, &result, x->ctx, x, rank, shape, strides, x->storage_offset);
    if (mag_iserr(status)) return status;
  }
  status = mag_check_dtype_and_device_compat(err, MAG_OP_VIEW, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_VIEW, false, NULL, &x, 1, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_reshape(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  int64_t shape[MAG_MAX_DIMS];
  mag_status_t status = mag_infer_missing_dim(err, &shape, dims, rank, x->numel);
  if (mag_iserr(status)) return status;
  if (x->coords.rank == rank && !memcmp(x->coords.shape, shape, sizeof(*dims)*rank)) {
    mag_rc_incref(x);
    *out_result = x;
    return MAG_STATUS_OK;
  }
  if (mag_tensor_is_contiguous(x)) {
    int64_t strides[MAG_MAX_DIMS];
    strides[rank-1] = 1;
    for (int64_t i=rank-2; i >= 0; --i) {
      if (mag_unlikely(mag_mulov64(shape[i+1], strides[i+1], strides+i)))
          return mag_set_error(err, MAG_STATUS_ERR_DIM_OVERFLOW, "reshape: stride computation overflowed at dim %" PRIi64 ".", i);
    }
    status = mag_as_strided(err, &result, x->ctx, x, rank, shape, strides, x->storage_offset);
    if (mag_iserr(status)) return status;
    *out_result = result;
    return MAG_STATUS_OK;
  }
  if (mag_tensor_can_view(x, shape, rank)) {
    status = mag_view(err, &result, x, shape, rank);
    if (mag_iserr(status)) return status;
    *out_result = result;
    return MAG_STATUS_OK;
  }
  status = mag_contiguous(err, &result, x);
  if (mag_iserr(status)) return status;
  int64_t strides[MAG_MAX_DIMS];
  strides[rank-1] = 1;
  for (int64_t i=rank-2; i >= 0; --i) {
    if (mag_unlikely(mag_mulov64(shape[i+1], strides[i+1], strides+i))) {
      mag_rc_decref(result);
      return mag_set_error(err, MAG_STATUS_ERR_DIM_OVERFLOW, "reshape: stride computation overflowed at dim %" PRIi64 ".", i);
    }
  }
  mag_tensor_t *reshaped;
  status = mag_as_strided(err, &reshaped, result->ctx, result, rank, shape, strides, result->storage_offset);
  if (mag_iserr(status)) {
    mag_rc_decref(result);
    return status;
  }
  mag_rc_decref(result);
  *out_result = reshaped;
  return MAG_STATUS_OK;
}

mag_status_t mag_view_slice(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t start, int64_t len, int64_t step) {
  *out_result = NULL;
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "slice: cannot slice a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "slice: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  if (mag_unlikely(!(step > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "slice: step must be > 0, but got %" PRIi64 ".", step);
  int64_t sz = x->coords.shape[dim];
  mag_norm_axis(&start, sz);
  if (mag_unlikely(!(0 <= start && start < sz)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "slice: start %" PRIi64 " is out of bounds for dim %" PRIi64 " of size %" PRIi64 ".", start, dim, sz);
  if (len < 0) len = (sz - start + step - 1)/step;
  if (mag_unlikely(!(len > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "slice: length must be > 0, but got %" PRIi64 ".", len);
  int64_t last = start + (len - 1)*step;
  if (mag_unlikely(!(0 <= last && last < sz)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "slice: end index %" PRIi64 " exceeds size %" PRIi64 " on dim %" PRIi64 ".", last, sz, dim);
  int64_t shape[MAG_MAX_DIMS];
  int64_t strides[MAG_MAX_DIMS];
  memcpy(shape, x->coords.shape, rank*sizeof(*shape));
  memcpy(strides, x->coords.strides, rank*sizeof(*strides));
  shape[dim] = len;
  strides[dim] = x->coords.strides[dim] * step;
  int64_t offset = x->storage_offset + start*x->coords.strides[dim];
  return mag_as_strided(err, out_result, x->ctx, x, rank, shape, strides, offset);
}

mag_status_t mag_transpose(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim1, int64_t dim2) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  if (mag_unlikely(!(x->coords.rank >= 2)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "transpose: requires rank >= 2, but got %" PRIi64 ".", x->coords.rank);
  if (mag_unlikely(!(dim1 != dim2)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "transpose: axes must differ, but got dim1 == dim2 == %" PRIi64 ".", dim1);
  int64_t ra = x->coords.rank;
  int64_t ax0 = dim1;
  int64_t ax1 = dim2;
  mag_norm_axis(&ax0, ra);
  mag_norm_axis(&ax1, ra);
  if (mag_unlikely(!(ax0 >= 0 && ax0 < ra)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "transpose: axis %" PRIi64 " is out of range for rank %" PRIi64 ".", dim1, ra);
  if (mag_unlikely(!(ax1 >= 0 && ax1 < ra)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "transpose: axis %" PRIi64 " is out of range for rank %" PRIi64 ".", dim2, ra);
  int64_t shape[MAG_MAX_DIMS];
  int64_t stride[MAG_MAX_DIMS];
  memcpy(shape, x->coords.shape, sizeof shape);
  memcpy(stride, x->coords.strides, sizeof stride);
  mag_swap(int64_t, shape[ax0], shape[ax1]);
  mag_swap(int64_t, stride[ax0], stride[ax1]);
  mag_status_t status = mag_as_strided(err, &result, x->ctx, x, x->coords.rank, shape, stride, x->storage_offset);
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(ax0));
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(ax1));
  status = mag_check_dtype_and_device_compat(err, MAG_OP_TRANSPOSE, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_TRANSPOSE, false, &layout, &x, 1, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_T(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x) {
  int64_t rank = mag_tensor_rank(x);
  if (rank < 2) {
    mag_rc_incref(x);
    *out_result = x;
    return MAG_STATUS_OK;
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
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "permute: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  int64_t axes[MAG_MAX_DIMS];
  for (int64_t i=0; i < rank; ++i) axes[i] = dims[i];
  for (int64_t i=0; i < rank; ++i) {
    for (int64_t j = i+1; j < rank; ++j) {
      if (mag_unlikely(!(axes[i] != axes[j])))
          return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "permute: duplicate axis %" PRIi64 " at positions %" PRIi64 " and %" PRIi64 ".", axes[i], i, j);
    }
  }
  int64_t shape[MAG_MAX_DIMS];
  int64_t stride[MAG_MAX_DIMS];
  for (int64_t i=0; i < rank; ++i) {
    shape[i] = x->coords.shape[axes[i]];
    stride[i] = x->coords.strides[axes[i]];
  }
  mag_status_t status = mag_as_strided(err, &result, x->ctx, x, x->coords.rank, shape, stride, x->storage_offset);
  if (mag_iserr(status)) return status;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_PERMUTE, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_PERMUTE, false, NULL, &x, 1, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_contiguous(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x) {
  if (!x->storage_offset && mag_tensor_is_contiguous(x)) {
    mag_rc_incref(x); /* Borrow +1 ref for caller; *out may alias x — caller must mag_tensor_decref(*out) once */
    *out_result = x;
    return MAG_STATUS_OK;
  }
  return mag_clone(err, out_result, x);
}

mag_status_t mag_squeeze_all(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x) {
  *out_result = NULL;
  int64_t rank = x->coords.rank;
  if (!rank) return mag_view(err, out_result, x, x->coords.shape, 0);
  int64_t shape[MAG_MAX_DIMS];
  int64_t nrank = 0;
  for (int64_t i=0; i < rank; ++i) {
    int64_t sz = x->coords.shape[i];
    if (sz != 1) shape[nrank++] = sz;
  }
  return nrank == rank ? mag_view(err, out_result, x, x->coords.shape, rank) : mag_view(err, out_result, x, shape, nrank);
}

mag_status_t mag_squeeze_dim(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
  *out_result = NULL;
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "squeeze: cannot squeeze a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "squeeze: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  int64_t sz = x->coords.shape[dim];
  if (sz != 1) return mag_view(err, out_result, x, x->coords.shape, rank);
  int64_t shape[MAG_MAX_DIMS];
  int64_t nrank = 0;
  for (int64_t i=0; i < rank; ++i) {
    if (i == dim) continue;
    shape[nrank++] = x->coords.shape[i];
  }
  return mag_view(err, out_result, x, shape, nrank);
}

mag_status_t mag_unsqueeze(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
  *out_result = NULL;
  int64_t rank = x->coords.rank;
  int64_t nrank = rank+1;
  if (mag_unlikely(!(nrank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "unsqueeze: result would exceed the maximum rank of %d.", MAG_MAX_DIMS);
  mag_norm_axis(&dim, nrank);
  if (mag_unlikely(!(0 <= dim && dim < nrank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "unsqueeze: dim %" PRIi64 " is out of range for new rank %" PRIi64 ".", dim, nrank);
  int64_t shape[MAG_MAX_DIMS];
  for (int64_t i=0, j=0; i < nrank; ++i)
    shape[i] = i == dim ? 1 : x->coords.shape[j++];
  return mag_view(err, out_result, x, shape, nrank);
}

mag_status_t mag_flatten(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t start_dim, int64_t end_dim) {
  *out_result = NULL;
  int64_t rank = x->coords.rank;
  if (!rank) return mag_view(err, out_result, x, x->coords.shape, 0);
  mag_norm_axis(&start_dim, rank);
  mag_norm_axis(&end_dim, rank);
  if (mag_unlikely(!(0 <= start_dim && start_dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "flatten: start_dim %" PRIi64 " is out of range for rank %" PRIi64 ".", start_dim, rank);
  if (mag_unlikely(!(0 <= end_dim && end_dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "flatten: end_dim %" PRIi64 " is out of range for rank %" PRIi64 ".", end_dim, rank);
  if (mag_unlikely(!(start_dim <= end_dim)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "flatten: start_dim must be <= end_dim, but got %" PRIi64 " > %" PRIi64 ".", start_dim, end_dim);
  int64_t shape[MAG_MAX_DIMS];
  int64_t nrank = 0;
  for (int64_t i=0; i < start_dim; ++i)
    shape[nrank++] = x->coords.shape[i];
  int64_t sz=1;
  for (int64_t i=start_dim; i <= end_dim; ++i)
    sz *= x->coords.shape[i];
  shape[nrank++] = sz;
  for (int64_t i=end_dim+1; i < rank; ++i)
    shape[nrank++] = x->coords.shape[i];
  if (mag_unlikely(!(nrank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "flatten: result rank %" PRIi64 " exceeds the maximum rank of %d.", nrank, MAG_MAX_DIMS);
  mag_status_t stat = mag_view(err, out_result, x, shape, nrank); /* Try view first */
  if (mag_iserr(stat))
    stat = mag_reshape(err, out_result, x, shape, nrank);
  return stat;
}

mag_status_t mag_unflatten(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, const int64_t *sizes, int64_t sizes_rank) {
  *out_result = NULL;
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(sizes_rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "unflatten: sizes must contain at least one dimension.");
  if (mag_unlikely(!(sizes != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "unflatten: sizes must not be NULL.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK,
        "unflatten: dim %" PRIi64 " is out of range for rank %" PRIi64 ".",
        dim, rank
      );
  if (mag_unlikely(!(sizes_rank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK,
        "unflatten: sizes rank %" PRIi64 " exceeds the maximum rank of %d.",
        sizes_rank, MAG_MAX_DIMS
      );
  int64_t nr = rank - 1 + sizes_rank;
  if (mag_unlikely(!(nr <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK,
        "unflatten: result rank %" PRIi64 " exceeds the maximum rank of %d.",
        nr, MAG_MAX_DIMS
      );
  int64_t resolved[MAG_MAX_DIMS];
  mag_status_t status = mag_infer_missing_dim(
    err,
    &resolved,
    sizes,
    sizes_rank,
    x->coords.shape[dim]
  );
  if (mag_iserr(status)) return status;
  int64_t shape[MAG_MAX_DIMS];
  int64_t k = 0;
  for (int64_t i=0; i < dim; ++i)
    shape[k++] = x->coords.shape[i];
  for (int64_t i=0; i < sizes_rank; ++i)
    shape[k++] = resolved[i];
  for (int64_t i=dim+1; i < rank; ++i)
    shape[k++] = x->coords.shape[i];
  mag_status_t stat = mag_view(err, out_result, x, shape, nr);
  if (mag_iserr(stat)) {
    mag_error_t ignored = {0};
    stat = mag_reshape(&ignored, out_result, x, shape, nr);
    if (mag_iserr(stat))
      stat = mag_reshape(err, out_result, x, shape, nr);
  }
  return stat;
}

mag_status_t mag_narrow(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t start, int64_t length) {
  *out_result = NULL;
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "narrow: cannot narrow a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK,
        "narrow: dim %" PRIi64 " is out of range for rank %" PRIi64 ".",
        dim, rank);
  if (mag_unlikely(!(length >= 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
        "narrow: length must be >= 0, but got %" PRIi64 ".",
        length);
  if (mag_unlikely(!(length > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "narrow: length 0 is not supported.");
  int64_t sz = x->coords.shape[dim];
  mag_norm_axis(&start, sz);
  if (mag_unlikely(!(start >= 0 && start <= sz)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
        "narrow: start %" PRIi64 " is out of bounds for dim of size %" PRIi64 ".",
        start, sz);
  int64_t end = start+length;
  if (mag_unlikely(!(end <= sz)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
        "narrow: range [%" PRIi64 ", %" PRIi64 ") exceeds dim size %" PRIi64 ".",
        start, end, sz);
  return mag_view_slice(err, out_result, x, dim, start, length, 1);
}

mag_status_t mag_movedim(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t src, int64_t dst) {
  *out_result = NULL;
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "movedim: cannot apply movedim to a scalar tensor.");
  mag_norm_axis(&src, rank);
  mag_norm_axis(&dst, rank);
  if (mag_unlikely(!(0 <= src && src < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "movedim: source dim %" PRIi64 " is out of range for rank %" PRIi64 ".", src, rank);
  if (mag_unlikely(!(0 <= dst && dst < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "movedim: destination dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dst, rank);
  if (src == dst)
    return mag_view(err, out_result, x, x->coords.shape, rank);
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
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "select: cannot select from a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "select: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  int64_t sz = x->coords.shape[dim];
  mag_norm_axis(&index, sz);
  if (mag_unlikely(!(0 <= index && index < sz)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "select: index %" PRIi64 " is out of bounds for dim of size %" PRIi64 ".", index, sz);
  mag_tensor_t *tmp = NULL;
  mag_status_t status = mag_view_slice(err, &tmp, x, dim, index, 1, 1);
  if (mag_iserr(status)) return status;
  status = mag_squeeze_dim(err, out_result, tmp, dim);
  if (mag_iserr(status)) {
    mag_tensor_decref(tmp);
    return status;
  }
  return MAG_STATUS_OK;
}

mag_status_t mag_split(mag_error_t *err, mag_tensor_t **outs, int64_t num_splits, mag_tensor_t *x, int64_t split_size, int64_t dim) {
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(split_size > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "split: split_size must be > 0, but got %" PRIi64 ".", split_size);
  if (mag_unlikely(!(rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "split: cannot split a scalar tensor.");
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "split: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  int64_t sz = x->coords.shape[dim];
  int64_t expected_chunks = 0;
  if (sz > 0) expected_chunks = (sz + split_size-1)/split_size;
  if (mag_unlikely(!(num_splits >= 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "split: number of splits must be >= 0, but got %" PRIi64 ".", num_splits);
  if (mag_unlikely(!(num_splits == expected_chunks)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "split: number of splits (%" PRIi64 ") does not match the expected chunk count (%" PRIi64 ").", num_splits, expected_chunks);
  if (!num_splits) return MAG_STATUS_OK;
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
  return MAG_STATUS_OK;
}

static mag_status_t mag_op_stub_reduction(mag_error_t *err, mag_tensor_t **out_result, mag_opcode_t op, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
  *out_result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, op, &x, 0);
  if (mag_iserr(status)) return status;
  mag_reduce_plan_t plan;
  status = mag_reduce_plan_init(err, &plan, &x->coords, dims, rank, keepdim);
  if (mag_iserr(status)) return status;
  mag_tensor_t *result = NULL;
  mag_dtype_t otype;
  if ((op == MAG_OP_SUM || op == MAG_OP_PROD) && mag_tensor_is_integer_typed(x)) {
    /* For sum/prod use large int64/uint64 as result dtype to store big accumulators */
    otype = mag_dtype_bit(x->dtype) & MAG_DTYPE_MASK_UINT ? MAG_DTYPE_UINT64 : MAG_DTYPE_INT64;
  } else if (op == MAG_OP_ANY || op == MAG_OP_ALL) { /* For logical reductions, use boolean dtype */
    otype = MAG_DTYPE_BOOLEAN;
  } else if (op == MAG_OP_ARGMIN || op == MAG_OP_ARGMAX) { /* For argmin/argmax, use int64 dtype */
    otype = MAG_DTYPE_INT64;
  } else { /* For other reductions, use same dtype as input */
    otype = x->dtype;
  }
  if (!keepdim && !plan.out_rank) status = mag_empty_scalar(err, &result, x->ctx, otype, mag_tensor_device_id(x));
  else status = mag_empty(err, &result, x->ctx, otype, plan.out_rank, plan.out_shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_ptr(&plan));
  status = mag_dispatch(err, op, false, &layout, &x, 1, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
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
  *out_values  = NULL;
  *out_indices = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "topk: input tensor must not be NULL.");
  mag_context_t *ctx = x->ctx;
  if (mag_unlikely(!(k > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "topk: k must be > 0, but got %" PRIi64 ".", k);
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "topk: requires a tensor with rank > 0.");
  if (dim < 0) dim += rank;
  if (mag_unlikely(!(0 <= dim && dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "topk: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", dim, rank);
  int64_t dim_size = x->coords.shape[dim];
  if (mag_unlikely(!(k <= dim_size)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "topk: k (%" PRIi64 ") must be <= the size of dim %" PRIi64 " (%" PRIi64 ").", k, dim, dim_size);
  int64_t shape[MAG_MAX_DIMS];
  memcpy(shape, x->coords.shape, sizeof(*shape)*rank);
  shape[dim] = k;
  mag_tensor_t *values  = NULL;
  mag_tensor_t *indices = NULL;
  mag_status_t status = mag_empty(err, &values, ctx, x->dtype, rank, shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &indices, ctx, MAG_DTYPE_INT64, rank, shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) {
    mag_tensor_decref(values);
    return status;
  }
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(k));
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(dim));
  mag_op_attr_registry_insert(&layout, mag_op_attr_bool(largest));
  mag_op_attr_registry_insert(&layout, mag_op_attr_bool(sorted));
  status = mag_check_dtype_and_device_compat(err, MAG_OP_TOPK, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_TOPK, false, &layout, &x, 1, (mag_tensor_t*[2]){values, indices}, 2);
  if (mag_iserr(status)) return status;
  *out_values = values;
  *out_indices = indices;
  return MAG_STATUS_OK;
}

static mag_status_t mag_op_stub_unary(mag_error_t *err, mag_tensor_t **out_result, mag_opcode_t op, mag_tensor_t *x, const mag_op_attr_registry_t *layout, bool inplace);

static mag_status_t mag_op_stub_cumulative(mag_error_t *err, mag_tensor_t **out_result, mag_opcode_t op, const char *ext, mag_tensor_t *x, int64_t dim) {
  *out_result = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cu%s: input tensor must not be NULL.", ext);
  if (mag_unlikely(!(x->coords.rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "cu%s: requires a tensor with rank > 0.", ext);
  mag_norm_axis(&dim, x->coords.rank);
  if (mag_unlikely(!(dim >= 0 && dim < x->coords.rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "cu%s: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", ext, dim, x->coords.rank);
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(dim));
  return mag_op_stub_unary(err, out_result, op, x, &layout, false);
}

mag_status_t mag_cusum(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
  return mag_op_stub_cumulative(err, out_result, MAG_OP_CUSUM, "sum", x, dim);
}

mag_status_t mag_cuprod(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
  return mag_op_stub_cumulative(err, out_result, MAG_OP_CUPROD, "prod", x, dim);
}

static mag_status_t mag_op_stub_cumulative_ext(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_opcode_t op, const char *ext, mag_tensor_t *x, int64_t dim) {
  *out_values = NULL;
  *out_indices = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cu%s: input tensor must not be NULL.", ext);
  if (mag_unlikely(!(x->coords.rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "cu%s: requires a tensor with rank > 0.", ext);
  mag_norm_axis(&dim, x->coords.rank);
  if (mag_unlikely(!(dim >= 0 && dim < x->coords.rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "cu%s: dim %" PRIi64 " is out of range for rank %" PRIi64 ".", ext, dim, x->coords.rank);
  mag_tensor_t *values = NULL;
  mag_tensor_t *indices = NULL;
  mag_status_t status = mag_empty_like(err, &values, x);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &indices, x->ctx, MAG_DTYPE_INT64, x->coords.rank, x->coords.shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) {
    mag_tensor_decref(values);
    return status;
  }
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(dim));
  status = mag_check_dtype_and_device_compat(err, op, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, op, false, &layout, &x, 1, (mag_tensor_t*[2]){values, indices}, 2);
  if (mag_iserr(status)) return status;
  *out_values = values;
  *out_indices = indices;
  return MAG_STATUS_OK;
}

mag_status_t mag_cumax(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t dim) {
  return mag_op_stub_cumulative_ext(err, out_values, out_indices, MAG_OP_CUMAX, "max", x, dim);
}

mag_status_t mag_cumin(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t dim) {
  return mag_op_stub_cumulative_ext(err, out_values, out_indices, MAG_OP_CUMIN, "min", x, dim);
}

static mag_status_t mag_op_stub_unary(mag_error_t *err, mag_tensor_t **out_result, mag_opcode_t op, mag_tensor_t *x, const mag_op_attr_registry_t *layout, bool inplace) {
  *out_result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, op, &x, 0);
  if (mag_iserr(status)) return status;
  mag_tensor_t *result = NULL;
  if (inplace) {
    status = mag_tensor_strided_view(err, &result, x); /* Use the same storage as x */
    if (mag_iserr(status)) return status;
    status = mag_check_inplace_grad_ok(err, x);
    if (mag_iserr(status)) return status;
  } else {
    status = mag_empty_like(err, &result, x); /* Allocate a new tensor for the result */
    if (mag_iserr(status)) return status;
  }
  status = mag_dispatch(err, op, inplace, layout, &x, 1, &result, 1);
  if (mag_iserr(status)) {
    mag_tensor_decref(result);
    return status;
  }
  *out_result = result;
  return MAG_STATUS_OK;
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
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "pad: input tensor must not be NULL.");
  if (mag_unlikely(!(pad != NULL || pad_len == 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "pad: padding array must not be NULL.");
  if (mag_unlikely(!(pad_len >= 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "pad: pad_len must be >= 0.");
  if (mag_unlikely(!(mode && *mode)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "pad: invalid mode string");
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(pad_len <= 2*rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "pad: expected at most %" PRIi64 " padding values for rank %" PRIi64 ", but got %" PRIi64 ".", 2*rank, rank, pad_len);
  mag_pad_plan_t plan = {0};
  plan.rank = rank;
  if (!strcmp(mode, "constant")) plan.mode = MAG_PAD_MODE_CONSTANT;
  else if (!strcmp(mode, "reflect")) plan.mode = MAG_PAD_MODE_REFLECT;
  else if (!strcmp(mode, "replicate")) plan.mode = MAG_PAD_MODE_REPLICATE;
  else return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "pad: invalid mode string '%s'.", mode);
  plan.value = value;
  for (int64_t d=0; d < rank; ++d) {
    int64_t idx = (rank - 1 - d)<<1;
    plan.pad_before[d] = idx < pad_len ? pad[idx] : 0;
    plan.pad_after[d] = idx + 1 < pad_len ? pad[idx+1] : 0;
    if (mag_unlikely(!(plan.pad_before[d] >= 0 && plan.pad_after[d] >= 0)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "pad: padding values must be >= 0.");
    if (plan.mode == MAG_PAD_MODE_REFLECT) {
      int64_t dim = x->coords.shape[d];
      if (mag_unlikely(!(plan.pad_before[d] < dim && plan.pad_after[d] < dim)))
          return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "pad: reflect padding on dim %" PRIi64 " must be less than input size %" PRIi64 ".", d, dim);
    }
  }
  int64_t shape[MAG_MAX_DIMS];
  for (int64_t dim=0; dim < rank; ++dim)
    shape[dim] = x->coords.shape[dim] + plan.pad_before[dim] + plan.pad_after[dim];
  mag_tensor_t *result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_PAD, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &result, x->ctx, x->dtype, rank, shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_ptr(&plan));
  status = mag_dispatch(err, MAG_OP_PAD, false, &layout, &x, 1, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_tril(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag) {
  *out_result = NULL;
  if (mag_unlikely(!(tensor->coords.rank >= 2)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "tril: requires rank >= 2, but got %" PRIi64 ".", tensor->coords.rank);
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(diag));
  return mag_op_stub_unary(err, out_result, MAG_OP_TRIL, tensor, &layout, false);
}

mag_status_t mag_tril_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag) {
  *out_result = NULL;
  if (mag_unlikely(!(tensor->coords.rank >= 2)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "tril_: requires rank >= 2, but got %" PRIi64 ".", tensor->coords.rank);
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(diag));
  return mag_op_stub_unary(err, out_result, MAG_OP_TRIL, tensor, &layout, true);
}

mag_status_t mag_triu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag) {
  *out_result = NULL;
  if (mag_unlikely(!(tensor->coords.rank >= 2)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "triu: requires rank >= 2, but got %" PRIi64 ".", tensor->coords.rank);
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(diag));
  return mag_op_stub_unary(err, out_result, MAG_OP_TRIU, tensor, &layout, false);
}

mag_status_t mag_triu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag) {
  *out_result = NULL;
  if (mag_unlikely(!(tensor->coords.rank >= 2)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "triu_: requires rank >= 2, but got %" PRIi64 ".", tensor->coords.rank);
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(diag));
  return mag_op_stub_unary(err, out_result, MAG_OP_TRIU, tensor, &layout, true);
}

mag_status_t mag_multinomial(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t num_samples, bool replacement) {
  *out_result = NULL;
  if (mag_unlikely(!(tensor->coords.rank == 1 || tensor->coords.rank == 2)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "multinomial: requires rank 1 or 2, but got %" PRIi64 ".", tensor->coords.rank);
  if (mag_unlikely(!mag_tensor_is_contiguous(tensor)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "multinomial: input tensor must be contiguous row-major.");
  if (mag_unlikely(!(num_samples > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "multinomial: num_samples must be > 0, but got %" PRIi64 ".", num_samples);
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_MULTINOMIAL, &tensor, 0);
  if (mag_iserr(status)) return status;
  int64_t shape[MAG_MAX_DIMS] = {0};
  if (tensor->coords.rank > 1) memcpy(shape, tensor->coords.shape, (tensor->coords.rank - 1)*sizeof(*shape));
  shape[tensor->coords.rank-1] = num_samples;
  mag_tensor_t *result;
  status = mag_empty(err, &result, tensor->ctx, MAG_DTYPE_INT64, tensor->coords.rank, shape, mag_tensor_device_id(tensor));
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(num_samples));
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(!!replacement));
  status = mag_dispatch(err, MAG_OP_MULTINOMIAL, false, &layout, &tensor, 1, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_cat(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count, int64_t dim) {
  *out_result = NULL;
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cat: tensors array must not be NULL.");
  mag_tensor_t *result = NULL;
  if (mag_unlikely(!(count > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cat: tensor count must be > 0.");
  mag_tensor_t *t0 = tensors[0];
  if (mag_unlikely(!(t0 != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cat: first tensor must not be NULL.");
  int64_t rank = t0->coords.rank;
  mag_norm_axis(&dim, rank);
  if (mag_unlikely(!(dim >= 0 && dim < MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cat: dim must be in [0, %d), but got %" PRIi64 ".", MAG_MAX_DIMS, dim);
  if (mag_unlikely(!(rank > 0 && dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "cat: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", rank, dim);
  mag_dtype_t dtype = t0->dtype;
  int64_t shape[MAG_MAX_DIMS];
  memcpy(shape, t0->coords.shape, rank*sizeof(*shape));
  shape[dim] = 0;
  mag_tensor_t **tmp = (*mag_try_alloc)(NULL, count*sizeof(*tmp), 0);
  if (mag_unlikely(!tmp))
    return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "cat: failed to allocate temporary array for %zu tensors.", count);
  for (size_t i=0; i < count; ++i) {
    mag_tensor_t *tensor = tensors[i];
    if (mag_unlikely(!(tensor != NULL))) {
      for (size_t j=0; j < i; ++j) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cat: tensor at index %" PRIu64 " is NULL.", (uint64_t)i);
    }
    if (mag_unlikely(!(tensor->coords.rank == rank))) {
      for (size_t j=0; j < i; ++j) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cat: all tensors must have the same rank (got %" PRIi64 " and %" PRIi64 ").", tensor->coords.rank, rank);
    }
    if (mag_unlikely(!(tensor->dtype == dtype))) {
      for (size_t j=0; j < i; ++j) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cat: all tensors must have the same dtype (got %s and %s).", mag_type_trait(tensor->dtype)->name, mag_type_trait(dtype)->name);
    }
    for (int64_t j=0; j < rank; ++j) {
      if (j == dim) continue;
      if (mag_unlikely(!(tensor->coords.shape[j] == t0->coords.shape[j]))) {
        for (size_t k=0; k < i; ++k) mag_tensor_decref(tmp[k]);
        (*mag_alloc)(tmp, 0, 0);
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "cat: shapes must match on non-concat dimensions (mismatch on axis %" PRIi64 ").", j);
      }
    }
    mag_status_t cont_status = mag_contiguous(err, tmp+i, tensor); /* TODO: kernel requires all tensors to be contiguous for now, add strided path */
    if (mag_iserr(cont_status)) {
      for (size_t j=0; j < i; ++j) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return cont_status;
    }
    shape[dim] += tensor->coords.shape[dim];
  }
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(dim));
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_CAT, tmp, (uint32_t)count);
  if (mag_iserr(status)) { for (size_t i=0; i < count; ++i) mag_tensor_decref(tmp[i]); (*mag_alloc)(tmp, 0, 0); return status; }
  status = mag_empty(err, &result, t0->ctx, dtype, rank, shape, mag_tensor_device_id(*tmp));
  if (mag_iserr(status)) { for (size_t i=0; i < count; ++i) mag_tensor_decref(tmp[i]); (*mag_alloc)(tmp, 0, 0); return status; }
  status = mag_dispatch(err, MAG_OP_CAT, false, &layout, tmp, count, &result, 1);
  if (mag_iserr(status)) { for (size_t i=0; i < count; ++i) mag_tensor_decref(tmp[i]); (*mag_alloc)(tmp, 0, 0); return status; }
  for (size_t i=0; i < count; ++i)
    mag_tensor_decref(tmp[i]);
  (*mag_alloc)(tmp, 0, 0);
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_stack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count, int64_t dim) {
  *out_result = NULL;
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "stack: tensors array must not be NULL.");
  if (mag_unlikely(!(count > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "stack: tensor count must be > 0.");
  if (mag_unlikely(!(tensors[0] != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "stack: first tensor must not be NULL.");
  int64_t rank = tensors[0]->coords.rank;
  if (mag_unlikely(!(dim >= 0 && dim <= rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "stack: dim must be in [0, %" PRIi64 "], but got %" PRIi64 ".", rank, dim);
  if (mag_unlikely(!(rank + 1 <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "stack: result rank would exceed MAG_MAX_DIMS.");
  mag_tensor_t **tmp = (*mag_try_alloc)(NULL, count*sizeof(*tmp), 0);
  if (mag_unlikely(!tmp))
    return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "stack: failed to allocate temporary array for %zu tensors.", count);
  for (size_t i=0; i < count; ++i) {
    tmp[i] = NULL;
    mag_status_t st = mag_unsqueeze(err, &tmp[i], tensors[i], dim);
    if (mag_iserr(st)) {
      for (size_t j=0; j < i; ++j) if (tmp[j]) mag_tensor_decref(tmp[j]);
      (*mag_alloc)(tmp, 0, 0);
      return st;
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
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "hstack: tensors array must not be NULL.");
  if (mag_unlikely(!(count > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "hstack: tensor count must be > 0.");
  if (mag_unlikely(!(tensors[0] != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "hstack: first tensor must not be NULL.");
  int64_t rank = tensors[0]->coords.rank;
  return mag_cat(err, out_result, tensors, count, rank == 1 ? 0 : 1);
}

mag_status_t mag_vstack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count) {
  *out_result = NULL;
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "vstack: tensors array must not be NULL.");
  if (mag_unlikely(!(count > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "vstack: tensor count must be > 0.");
  if (mag_unlikely(!(tensors[0] != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "vstack: first tensor must not be NULL.");
  int64_t rank = tensors[0]->coords.rank;
  return rank == 1 ? mag_stack(err, out_result, tensors, count, 0) : mag_cat(err, out_result, tensors, count, 0);
}

mag_status_t mag_dstack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count) {
  *out_result = NULL;
  if (mag_unlikely(!(tensors != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "dstack: tensors array must not be NULL.");
  if (mag_unlikely(!(count > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "dstack: tensor count must be > 0.");
  if (mag_unlikely(!(tensors[0] != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "dstack: first tensor must not be NULL.");
  int64_t rank = tensors[0]->coords.rank;
  if (rank >= 3)
    return mag_cat(err, out_result, tensors, count, 2);
  mag_tensor_t **tmp = (*mag_try_alloc)(NULL, count*sizeof(*tmp), 0);
  if (mag_unlikely(!tmp))
    return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "dstack: failed to allocate temporary array for %zu tensors.", count);
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
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "chunk: input tensor must not be NULL.");
  if (mag_unlikely(!(chunks > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "chunk: chunks must be > 0.");
  int64_t rank = x->coords.rank;
  if (mag_unlikely(!(rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "chunk: input rank must be > 0.");
  if (mag_unlikely(!(dim >= 0 && dim < rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "chunk: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", rank, dim);
  int64_t n = x->coords.shape[dim];
  if (n == 0) {
    *out_chunks = NULL;
    *out_count = 0;
    return MAG_STATUS_OK;
  }
  int64_t chunk_size = (n+chunks-1)/chunks;
  int64_t actual = (n+chunk_size-1)/chunk_size;
  mag_tensor_t **res = (*mag_try_alloc)(NULL, (size_t)actual*sizeof(*res), 0);
  if (mag_unlikely(!res))
    return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "chunk: failed to allocate result array for %" PRIi64 " chunks.", actual);
  memset(res, 0, (size_t)actual*sizeof(*res));
  for (int64_t i=0; i < actual; ++i) {
    int64_t start = i * chunk_size;
    int64_t len = chunk_size;
    if (start+len > n) len = n-start;
    mag_status_t st = mag_narrow(err, &res[i], x, dim, start, len);
    if (st != MAG_STATUS_OK) {
      for (int64_t j=0; j < actual; ++j) {
        if (res[j])
          mag_tensor_decref(res[j]);
      }
      (*mag_alloc)(res, 0, 0);
      return st;
    }
  }
  *out_chunks = res;
  *out_count = (size_t)actual;
  return MAG_STATUS_OK;
}

mag_status_t mag_einsum(mag_error_t *err, mag_tensor_t **out_result, const char *equation, mag_tensor_t **args, size_t num_args) {
  return mag_einsum_eval(err, out_result, equation, (const mag_tensor_t **)args, num_args);
}

mag_status_t mag_one_hot(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *indices, int64_t num_classes) {
  *out_result = NULL;
  mag_context_t *ctx = indices->ctx;
  if (mag_unlikely(!(indices->dtype == MAG_DTYPE_INT64)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "one_hot: indices must have dtype int64, but got %s.", mag_type_trait(indices->dtype)->name);
  if (mag_unlikely(!(num_classes >= -1)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "one_hot: num_classes must be >= -1, but got %" PRIi64 ".", num_classes);
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
    int64_t max_class = mag_scalar_as_i64(max_scalar);
    mag_tensor_decref(maxv);
    num_classes = max_class >= 0 ? 1+max_class : 0;
  }
  if (mag_unlikely(!(num_classes > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "one_hot: inferred num_classes must be > 0, but got %" PRIi64 ".", num_classes);
  int64_t rank = indices->coords.rank;
  if (mag_unlikely(!(rank + 1 <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "one_hot: result rank (rank(indices)+1) exceeds the maximum rank of %d.", MAG_MAX_DIMS);
  int64_t orank = rank+1;
  int64_t oshape[MAG_MAX_DIMS];
  for (int64_t i=0; i < rank; ++i)
    oshape[i] = indices->coords.shape[i];
  oshape[rank] = num_classes;
  mag_tensor_t *result;
  mag_status_t status = mag_zeros(err, &result, ctx, MAG_DTYPE_INT64, orank, oshape, mag_tensor_device_id(indices));
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(num_classes));
  status = mag_check_dtype_and_device_compat(err, MAG_OP_ONE_HOT, &indices, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_ONE_HOT, false, &layout, &indices, 1, &result, 1);
  if (mag_iserr(status)) {
    mag_tensor_decref(result);
    return status;
  }
  *out_result = result;
  return MAG_STATUS_OK;
}

typedef enum mag_binop_flags_t {
  MAG_BINOP_NONE = 0,
  MAG_BINOP_LOGICAL = 1<<0,
  MAG_BINOP_INPLACE = 1<<1
} mag_binop_flags_t;

static mag_status_t mag_op_stub_binary(mag_error_t *err, mag_tensor_t **out_result, mag_opcode_t op, mag_tensor_t *x, mag_tensor_t *y, mag_binop_flags_t flags) {
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
            return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "binary_op: in-place true division is not allowed on integer tensors (got dtype %s).", mag_type_trait(x->dtype)->name);
      } break;
      case MAG_OP_FLOORDIV: {
        if (mag_unlikely(!(x_int && y_int)))
            return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "binary_op: in-place floor division requires integer tensors, but got dtypes %s and %s.", mag_type_trait(x->dtype)->name, mag_type_trait(y->dtype)->name);
      } break;
      default: { /* Inplace ops must keep x's dtype */
        mag_dtype_t prom;
        bool prom_ok = mag_promote_type(&prom, x->dtype, y->dtype);
        if (mag_unlikely(!(prom_ok && prom == x->dtype)))
            return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "binary_op: in-place '%s' would change the dtype of x from %s to %s.", mag_op_trait(op)->mnemonic, mag_type_trait(x->dtype)->name, mag_type_trait(prom)->name);
      } break;
    }
    prom_type = x->dtype;
    res_type = x->dtype;
  } else if (flags & MAG_BINOP_LOGICAL) { /* Inplace keeps x's dtype, but cast y to x's dtype if needed */
    bool prom_ok = mag_promote_type(&prom_type, x->dtype, y->dtype);
    if (mag_unlikely(!prom_ok))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "binary_op: logical operator '%s' does not support dtypes %s and %s.",
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
              return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
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
            return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "binary_op: operator '%s' does not support dtypes %s and %s.",
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
            return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "binary_op: operator '%s' does not support dtypes %s and %s.",
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
    status = mag_tensor_strided_view(err, &result, x);
    if (mag_iserr(status)) return status;
  } else {
    int64_t dims[MAG_MAX_DIMS];
    int64_t rank;
    if (mag_unlikely(!mag_coords_broadcast_shape(&x->coords, &y->coords, dims, &rank))) {
      char sx[MAG_FMT_DIM_BUF_SIZE];
      char sy[MAG_FMT_DIM_BUF_SIZE];
      mag_fmt_shape(&sx, &x->coords.shape, x->coords.rank);
      mag_fmt_shape(&sy, &y->coords.shape, y->coords.rank);
      return mag_set_error(err, MAG_STATUS_ERR_BROADCAST_IMPOSSIBLE,
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
  status = mag_dispatch(err, op, flags & MAG_BINOP_INPLACE, NULL, in, sizeof(in)/sizeof(*in), &result, 1);
  if (mag_iserr(status)) return status;
  if (tmp_x) mag_tensor_decref(tmp_x);
  if (tmp_y) mag_tensor_decref(tmp_y);
  *out_result = result;
  return MAG_STATUS_OK;
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
  if (mag_unlikely(!(cond->dtype == MAG_DTYPE_BOOLEAN)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "where: condition tensor must have dtype bool, but got %s.", mag_type_trait(cond->dtype)->name);
  if (mag_unlikely(!(x->dtype == y->dtype)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "where: x and y must have the same dtype, but got %s and %s.", mag_type_trait(x->dtype)->name, mag_type_trait(y->dtype)->name);
  int64_t dims[MAG_MAX_DIMS];
  int64_t rank;
  const mag_coords_t *coords[3] = {&cond->coords, &x->coords, &y->coords};
  if (mag_unlikely(!mag_coords_broadcast_multi_shape(coords, sizeof(coords)/sizeof(*coords), dims, &rank))) {
    char sc[MAG_FMT_DIM_BUF_SIZE];
    char sx[MAG_FMT_DIM_BUF_SIZE];
    char sy[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&sc, &cond->coords.shape, cond->coords.rank);
    mag_fmt_shape(&sx, &x->coords.shape, x->coords.rank);
    mag_fmt_shape(&sy, &y->coords.shape, y->coords.rank);
    return mag_set_error(err, MAG_STATUS_ERR_BROADCAST_IMPOSSIBLE,
      "where: cannot broadcast shapes %s, %s and %s.\n"
      "    Hint: ensure that cond, x and y are broadcast-compatible.",
      sc, sx, sy);
  }
  mag_tensor_t *result = NULL;
  mag_status_t status = rank ? mag_empty(err, &result, x->ctx, x->dtype, rank, dims, mag_tensor_device_id(cond)) : mag_empty_scalar(err, &result, x->ctx, x->dtype, mag_tensor_device_id(cond));
  if (mag_iserr(status)) return status;
  mag_tensor_t *in[3] = {cond, x, y};
  status = mag_check_dtype_and_device_compat(err, MAG_OP_WHERE, in, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_WHERE, false, NULL, in, sizeof(in)/sizeof(*in), &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_clamp(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *min, mag_tensor_t *max) {
  *out_result = NULL;
  if (mag_unlikely(!(x->dtype == min->dtype && min->dtype == max->dtype)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "clamp: x, min and max must have the same dtype, but got %s, %s and %s.", mag_type_trait(x->dtype)->name, mag_type_trait(min->dtype)->name, mag_type_trait(max->dtype)->name);
  int64_t dims[MAG_MAX_DIMS];
  int64_t rank;
  const mag_coords_t *coords[3] = {&x->coords, &min->coords, &max->coords};
  if (mag_unlikely(!mag_coords_broadcast_multi_shape(coords, sizeof(coords)/sizeof(*coords), dims, &rank))) {
    char sc[MAG_FMT_DIM_BUF_SIZE];
    char sx[MAG_FMT_DIM_BUF_SIZE];
    char sy[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&sc, &x->coords.shape, x->coords.rank);
    mag_fmt_shape(&sx, &min->coords.shape, min->coords.rank);
    mag_fmt_shape(&sy, &max->coords.shape, max->coords.rank);
    return mag_set_error(err, MAG_STATUS_ERR_BROADCAST_IMPOSSIBLE,
      "clamp: cannot broadcast shapes %s, %s and %s.\n"
      "    Hint: ensure that x, min and max are broadcast-compatible.",
      sc, sx, sy);
  }
  mag_tensor_t *result = NULL;
  mag_status_t status = rank ? mag_empty(err, &result, x->ctx, x->dtype, rank, dims, mag_tensor_device_id(x)) : mag_empty_scalar(err, &result, x->ctx, x->dtype, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  mag_tensor_t *in[3] = {x, min, max};
  status = mag_check_dtype_and_device_compat(err, MAG_OP_CLAMP, in, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_CLAMP, false, NULL, in, sizeof(in)/sizeof(*in), &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

static mag_status_t mag_matmul_verify_shapes(mag_error_t *err, int64_t *rb, int64_t *xb, int64_t *yb, const mag_tensor_t *x, const mag_tensor_t *y) {
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
    return mag_set_error(err, MAG_STATUS_ERR_OPERATOR_IMPOSSIBLE,
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
      return mag_set_error(err, MAG_STATUS_ERR_OPERATOR_IMPOSSIBLE,
        "matmul: batch dim %" PRIi64 " (%" PRIi64 ") of x cannot broadcast with y (%" PRIi64 ") for shapes %s and %s.",
        i, xd, yd, sx, sy
      );
    }
  }
  return MAG_STATUS_OK;
}

static mag_status_t mag_matmul_alloc_res(mag_error_t *err, mag_tensor_t **res, int64_t rb, int64_t *xb, int64_t *yb, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y) {
  mag_matmul_type_t type = mag_matmul_type_detect(x, y);
  switch (type) {
    case MAG_MATMUL_TYPE_INVALID: return mag_set_error(err, MAG_STATUS_ERR_OPERATOR_IMPOSSIBLE, "matmul: unsupported tensor shapes.");
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
    } default: return mag_set_error(err, MAG_STATUS_ERR_OPERATOR_IMPOSSIBLE, "matmul: invalid BMM matmul type '%s'.", mag_matmul_type_name(type));
  }
}

mag_status_t mag_matmul(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y) {
  *out_result = NULL;
  if (mag_unlikely(!(mag_tensor_is_floating_point_typed(x) && mag_tensor_is_floating_point_typed(y))))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "matmul: requires floating-point tensors, but got dtypes %s and %s.", mag_type_trait(x->dtype)->name, mag_type_trait(y->dtype)->name);
  if (mag_unlikely(!(x->coords.rank >= 1 && y->coords.rank >= 1)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "matmul: both tensors must have rank >= 1, but got %" PRIi64 " and %" PRIi64 ".", x->coords.rank, y->coords.rank);
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_MATMUL, (mag_tensor_t *[]){x, y}, 0);
  if (mag_iserr(status)) return status;
  mag_tensor_t *result = NULL;
  int64_t rb, xb, yb;
  status = mag_matmul_verify_shapes(err, &rb, &xb, &yb, x, y);
  if (mag_iserr(status)) return status;
  status = mag_matmul_alloc_res(err, &result, rb, &xb, &yb, &result, x, y);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_MATMUL, false, NULL, (mag_tensor_t *[2]){x, y}, 2, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_repeat_back(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_REPEAT_BACK, (mag_tensor_t *[]) {x, y}, 0);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &result, x->ctx, x->dtype, y->coords.rank, y->coords.shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  /* TODO: Check for broadcastability of x and y */
  status = mag_dispatch(err, MAG_OP_REPEAT_BACK, false, NULL, (mag_tensor_t *[2]) {x, y}, 2, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_repeat(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *repeats, int64_t repeats_len) {
  *out_result = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "repeat: input tensor must not be NULL.");
  if (mag_unlikely(!(repeats != NULL && repeats_len > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "repeat: repeats must be a non-empty sequence.");
  int64_t in_rank = x->coords.rank;
  int64_t out_rank = in_rank > repeats_len ? in_rank : repeats_len;
  if (mag_unlikely(!(out_rank <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "repeat: result rank would exceed MAG_MAX_DIMS.");
  mag_repeat_plan_t plan = {0};
  plan.in_rank = in_rank;
  plan.rank = out_rank;
  int64_t lead_x = out_rank - in_rank;
  int64_t lead_r = out_rank - repeats_len;
  for (int64_t d=0; d < out_rank; ++d) {
    int64_t is = d >= lead_x ? x->coords.shape[d - lead_x] : 1;
    int64_t rs = d >= lead_r ? repeats[d - lead_r] : 1;
    if (mag_unlikely(!(rs >= 0)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "repeat: repeat counts must be >= 0.");
    plan.in_shape[d] = is;
    plan.out_shape[d] = is*rs;
  }
  mag_tensor_t *result = NULL;
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_REPEAT, &x, 0);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &result, x->ctx, x->dtype, out_rank, plan.out_shape, mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_ptr(&plan));
  status = mag_dispatch(err, MAG_OP_REPEAT, false, &layout, &x, 1, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_repeat_interleave(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, bool flatten, int64_t dim, const int64_t *counts, int64_t count_len) {
  *out_result = NULL;
  if (mag_unlikely(!(x != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "repeat_interleave: input tensor must not be NULL.");
  if (mag_unlikely(!(counts != NULL && count_len > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "repeat_interleave: counts must be a non-empty sequence.");
  for (int64_t i=0; i < count_len; ++i)
    if (mag_unlikely(!(counts[i] >= 0)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "repeat_interleave: counts must be >= 0.");
  mag_repeat_interleave_plan_t plan = {0};
  plan.flatten = flatten;
  plan.counts = counts;
  plan.count_len = count_len;
  mag_tensor_t *xin = NULL;
  int64_t shape[MAG_MAX_DIMS];
  mag_status_t status = MAG_STATUS_OK;
  if (flatten) {
    if (mag_unlikely(!(count_len == 1 || count_len == x->numel)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "repeat_interleave: counts length (%" PRIi64 ") must match input numel (%" PRIi64 ") when dim is None.", count_len, x->numel);
    int64_t out_n = 0;
    if (count_len == 1)
      out_n = x->numel*counts[0];
    else
      for (int64_t i=0; i < count_len; ++i) out_n += counts[i];
    plan.rank = 1;
    plan.out_shape[0] = out_n;
    status = mag_contiguous(err, &xin, x);
    if (mag_iserr(status)) return status;
  } else {
    if (mag_unlikely(!(x->coords.rank > 0)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "repeat_interleave: input must have rank >= 1.");
    mag_norm_axis(&dim, x->coords.rank);
    if (mag_unlikely(!(dim >= 0 && dim < x->coords.rank)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "repeat_interleave: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", x->coords.rank, dim);
    plan.dim = dim;
    plan.rank = x->coords.rank;
    memcpy(shape, x->coords.shape, plan.rank*sizeof(*shape));
    int64_t axis_len = x->coords.shape[dim];
    if (mag_unlikely(!(count_len == 1 || count_len == axis_len)))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "repeat_interleave: counts length (%" PRIi64 ") must match size of dim (%" PRIi64 ").", count_len, axis_len);
    if (count_len == 1)
      shape[dim] *= counts[0];
    else {
      int64_t sum = 0;
      for (int64_t i=0; i < count_len; ++i) sum += counts[i];
      shape[dim] = sum;
    }
    memcpy(plan.out_shape, shape, plan.rank*sizeof(*plan.out_shape));
    status = mag_contiguous(err, &xin, x);
    if (mag_iserr(status)) return status;
  }
  mag_tensor_t *result = NULL;
  status = mag_check_dtype_and_device_compat(err, MAG_OP_REPEAT_INTERLEAVE, &xin, 0);
  if (mag_iserr(status)) return status;
  status = mag_empty(err, &result, x->ctx, x->dtype, plan.rank, flatten ? plan.out_shape : shape, mag_tensor_device_id(xin));
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_ptr(&plan));
  status = mag_dispatch(err, MAG_OP_REPEAT_INTERLEAVE, false, &layout, &xin, 1, &result, 1);
  if (mag_iserr(status)) return status;
  mag_tensor_decref(xin);
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_gather(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t dim, mag_tensor_t *idx) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  if (mag_unlikely(!(idx->dtype == MAG_DTYPE_INT64)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "gather: index tensor must have dtype int64, but got %s.", mag_type_trait(idx->dtype)->name);
  if (mag_unlikely(!(dim >= 0 && dim < tensor->coords.rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "gather: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", tensor->coords.rank, dim);
  if (mag_unlikely(!(idx->coords.rank == tensor->coords.rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "gather: index rank (%" PRIi64 ") must equal input rank (%" PRIi64 "). Use embedding() for row-select indexing.", idx->coords.rank, tensor->coords.rank);
  mag_norm_axis(&dim, tensor->coords.rank);
  if (mag_unlikely(!(dim >= 0 && dim < tensor->coords.rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "gather: normalized dim %" PRIi64 " is out of range [0, %" PRIi64 ").", dim, tensor->coords.rank);
  /* Output shape equals index shape (same as torch.gather). */
  int64_t ork = idx->coords.rank;
  int64_t ax[MAG_MAX_DIMS];
  for (int64_t i = 0; i < ork; ++i) ax[i] = idx->coords.shape[i];
  mag_status_t status = mag_empty(err, &result, tensor->ctx, tensor->dtype, ork, ax, mag_tensor_device_id(tensor));
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(dim));
  status = mag_check_dtype_and_device_compat(err, MAG_OP_GATHER, (mag_tensor_t *[2]){tensor, idx}, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_GATHER, false, &layout, (mag_tensor_t *[2]) {tensor, idx}, 2, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_embedding(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *weight, mag_tensor_t *indices) {
  *out_result = NULL;
  mag_tensor_t *result = NULL;
  if (mag_unlikely(!(indices->dtype == MAG_DTYPE_INT64)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "embedding: indices tensor must have dtype int64, but got %s.", mag_type_trait(indices->dtype)->name);
  mag_dtype_mask_t fp_mask = MAG_DTYPE_MASK_FP;
  if (mag_unlikely(!((fp_mask & mag_dtype_bit(weight->dtype)) != 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "embedding: weight tensor must have a floating-point dtype, but got %s.", mag_type_trait(weight->dtype)->name);
  if (mag_unlikely(!(weight->coords.rank >= 1)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "embedding: weight must have rank >= 1.");
  if (mag_unlikely(!(indices->coords.rank >= 1)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "embedding: indices must have rank >= 1.");
  /* Output shape: indices.shape + weight.shape[1:] */
  int64_t ork = 0;
  int64_t ax[MAG_MAX_DIMS];
  for (int64_t i = 0; i < indices->coords.rank; ++i) ax[ork++] = indices->coords.shape[i];
  for (int64_t i = 1; i < weight->coords.rank; ++i)  ax[ork++] = weight->coords.shape[i];
  if (mag_unlikely(!(ork >= 1 && ork <= MAG_MAX_DIMS)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_RANK, "embedding: output rank must be in [1, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, ork);
  mag_status_t status = mag_empty(err, &result, weight->ctx, weight->dtype, ork, ax, mag_tensor_device_id(weight));
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  status = mag_check_dtype_and_device_compat(err, MAG_OP_EMBEDDING, (mag_tensor_t *[2]){weight, indices}, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_EMBEDDING, false, &layout, (mag_tensor_t *[2]) {weight, indices}, 2, &result, 1);
  if (mag_iserr(status)) return status;
  *out_result = result;
  return MAG_STATUS_OK;
}

mag_status_t mag_index_add_(mag_error_t *err, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *source, double alpha) {
  if (mag_unlikely(!(self != NULL && index != NULL && source != NULL)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "index_add_: tensors must not be NULL.");
  if (mag_unlikely(!(index->dtype == MAG_DTYPE_INT64)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "index_add_: index must have dtype int64, but got %s.", mag_type_trait(index->dtype)->name);
  if (mag_unlikely(!(index->coords.rank == 1)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "index_add_: index must be 1-D, but got rank %" PRIi64 ".", index->coords.rank);
  if (mag_unlikely(!(self->coords.rank > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "index_add_: self must have rank >= 1.");
  if (mag_unlikely(!(source->coords.rank == self->coords.rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "index_add_: source rank (%" PRIi64 ") must match self rank (%" PRIi64 ").", source->coords.rank, self->coords.rank);
  if (mag_unlikely(!(self->dtype == source->dtype)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "index_add_: self and source must have the same dtype, but got %s and %s.", mag_type_trait(self->dtype)->name, mag_type_trait(source->dtype)->name);
  mag_norm_axis(&dim, self->coords.rank);
  if (mag_unlikely(!(dim >= 0 && dim < self->coords.rank)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_DIM, "index_add_: dim must be in [0, %" PRIi64 "), but got %" PRIi64 ".", self->coords.rank, dim);
  int64_t idx_len = index->coords.shape[0];
  for (int64_t d=0; d < self->coords.rank; ++d) {
    if (d == dim) {
      if (mag_unlikely(!(source->coords.shape[d] == idx_len)))
          return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "index_add_: source size along dim (%" PRIi64 ") must match index length (%" PRIi64 ").", source->coords.shape[d], idx_len);
    } else {
      if (mag_unlikely(!(source->coords.shape[d] == self->coords.shape[d])))
          return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "index_add_: source shape must match self on non-index dimensions (mismatch on dim %" PRIi64 ").", d);
    }
  }
  mag_status_t status = mag_check_inplace_grad_ok(err, self);
  if (mag_iserr(status)) return status;
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_int64(dim));
  mag_op_attr_registry_insert(&layout, mag_op_attr_float64(alpha));
  mag_tensor_t *inputs[3] = {self, source, index};
  status = mag_check_dtype_and_device_compat(err, MAG_OP_INDEX_ADD, inputs, 0);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_INDEX_ADD, true, &layout, inputs, 3, &self, 1);
  if (mag_iserr(status)) return status;
  return MAG_STATUS_OK;
}

mag_status_t mag_copy_(mag_error_t *err, mag_tensor_t *dst, mag_tensor_t *src) {
  if (mag_unlikely(!(dst && src)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "copy: source and destination tensors must not be NULL.");
  if (mag_unlikely(!mag_tensor_is_shape_eq(dst, src)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "copy: source and destination must have the same shape.");
  if (mag_unlikely(!(dst->dtype == src->dtype)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM,
        "copy: source and destination must have the same dtype, but got %s and %s.",
        mag_type_trait(src->dtype)->name,
        mag_type_trait(dst->dtype)->name
      );
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_CLONE, (mag_tensor_t *[2]){src, dst}, 2);
  if (mag_iserr(status)) return status;
  status = mag_dispatch(err, MAG_OP_CLONE, true, NULL, &src, 1, &dst, 1);
  if (mag_iserr(status)) return status;
  return MAG_STATUS_OK;
}

mag_status_t mag_copy_raw_(mag_error_t *err, mag_tensor_t *tensor, const void *data, size_t size_bytes) {
  if (mag_unlikely(!(data != NULL && size_bytes > 0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "copy_raw: data pointer must not be NULL and size_bytes must be > 0.");
  if (mag_unlikely(!(tensor->storage->device->id.type == MAG_BACKEND_TYPE_CPU)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "copy_raw: tensor storage must reside on CPU, but got %s.", mag_backend_type_to_str(tensor->storage->device->id.type));
  if (mag_unlikely(!(mag_tensor_numbytes(tensor) == size_bytes)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "copy_raw: buffer size (%" PRIu64 " bytes) does not match the tensor size (%" PRIu64 " bytes).", (uint64_t)size_bytes, (uint64_t)mag_tensor_numbytes(tensor));
  if (mag_unlikely(!mag_tensor_is_contiguous(tensor)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "copy_raw: tensor must be contiguous to load from a raw buffer.");
  void *dst = (void *)mag_tensor_data_ptr_mut(tensor);
  memcpy(dst, data, size_bytes);
  return MAG_STATUS_OK;
}

mag_status_t mag_zeros_(mag_error_t *err, mag_tensor_t *tensor) {
  return mag_fill_(err, tensor, mag_scalar_from_u64(0));
}

mag_status_t mag_ones_(mag_error_t *err, mag_tensor_t *tensor) {
  return mag_fill_(err, tensor, mag_scalar_from_u64(1));
}

mag_status_t mag_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t value) {
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_scalar_to_op_attr(tensor->dtype, value));
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_FILL, NULL, 0);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_FILL, false, &layout, NULL, 0, &tensor, 1);
}

mag_status_t mag_masked_fill(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value) {
  *out_result = NULL;
  mag_status_t status = mag_clone(err, out_result, tensor);
  if (mag_iserr(status)) return status;
  return mag_masked_fill_(err, *out_result, mask, value);
}

mag_status_t mag_masked_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value) {
  if (mag_unlikely(!(mask->dtype == MAG_DTYPE_BOOLEAN)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "masked_fill: mask must have dtype bool, but got %s.", mag_type_trait(mask->dtype)->name);
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_scalar_to_op_attr(tensor->dtype, value));
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_MASKED_FILL, &mask, 1);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_MASKED_FILL, false, &layout, &mask, 1, &tensor, 1);
}

mag_status_t mag_uniform_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t min, mag_scalar_t max) {
  if (mag_unlikely(!mag_scalar_same_type(min, max)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "uniform_: min and max must have the same scalar type.");
  if (mag_unlikely(!mag_tensor_is_numeric_typed(tensor)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "uniform_: requires a numeric tensor dtype, but got %s.", mag_type_trait(tensor->dtype)->name);
  if (mag_scalar_is_f64(min)) {
    if (mag_unlikely(!(mag_scalar_as_f64(min) < mag_scalar_as_f64(max))))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "uniform_: min must be less than max (got min=%f, max=%f).", mag_scalar_as_f64(min), mag_scalar_as_f64(max));
  } else if (mag_scalar_is_i64(min)) {
    if (mag_unlikely(!(mag_scalar_as_i64(min) < mag_scalar_as_i64(max))))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "uniform_: min must be less than max (got min=%" PRIi64 ", max=%" PRIi64 ").", mag_scalar_as_i64(min), mag_scalar_as_i64(max));
  } else if (mag_scalar_is_u64(min)) {
    if (mag_unlikely(!(mag_scalar_as_u64(min) < mag_scalar_as_u64(max))))
        return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "uniform_: min must be less than max (got min=%" PRIu64 ", max=%" PRIu64 ").", mag_scalar_as_u64(min), mag_scalar_as_u64(max));
  } else {
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "uniform_: unsupported scalar type for min/max.");
  }
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_scalar_to_op_attr(tensor->dtype, min));
  mag_op_attr_registry_insert(&layout, mag_scalar_to_op_attr(tensor->dtype, max));
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_RAND_UNIFORM, NULL, 0);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_RAND_UNIFORM, false, &layout, NULL, 0, &tensor, 1);
}

mag_status_t mag_normal_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t mean, mag_scalar_t stddev) {
  if (mag_unlikely(!(mag_scalar_is_f64(mean) && mag_scalar_is_f64(stddev))))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "normal_: mean and stddev must be floating-point scalars.");
  if (mag_unlikely(!mag_tensor_is_floating_point_typed(tensor)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "normal_: requires a floating-point tensor dtype, but got %s.", mag_type_trait(tensor->dtype)->name);
  double stddev_f = mag_scalar_as_f64(stddev);
  double mean_f = mag_scalar_as_f64(mean);
  if (mag_unlikely(!(stddev_f >= 0.0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "normal_: stddev must be >= 0, but got %f.", stddev_f);
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_float64(mean_f));
  mag_op_attr_registry_insert(&layout, mag_op_attr_float64(stddev_f));
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_RAND_NORMAL, NULL, 0);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_RAND_NORMAL, false, &layout, NULL, 0, &tensor, 1);
}

mag_status_t mag_bernoulli_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t p) {
  if (mag_unlikely(!mag_scalar_is_f64(p)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "bernoulli_: p must be a floating-point scalar.");
  if (mag_unlikely(!(tensor->dtype == MAG_DTYPE_BOOLEAN)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "bernoulli_: requires a bool tensor dtype, but got %s.", mag_type_trait(tensor->dtype)->name);
  double pf = mag_scalar_as_f64(p);
  if (mag_unlikely(!(pf >= 0.0 && pf <= 1.0)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "bernoulli_: probability p must be in [0.0, 1.0], but got %f.", pf);
  mag_op_attr_registry_t layout;
  mag_op_attr_registry_init(&layout);
  mag_op_attr_registry_insert(&layout, mag_op_attr_float64(pf));
  mag_status_t status = mag_check_dtype_and_device_compat(err, MAG_OP_RAND_BERNOULLI, NULL, 0);
  if (mag_iserr(status)) return status;
  return mag_dispatch(err, MAG_OP_RAND_BERNOULLI, false, &layout, NULL, 0, &tensor, 1);
}
