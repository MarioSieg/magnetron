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

#include "mag_comm.h"
#include "mag_tensor.h"

mag_status_t mag_comm_host_stage_open(
  mag_error_t *err,
  mag_comm_host_staged_t *stage,
  mag_tensor_t *tensor,
  bool load
) {
  stage->target = tensor;
  stage->host = NULL;
  stage->is_alias = false;
  if (mag_tensor_is_host(tensor) && !tensor->meta.storage_offset && mag_tensor_is_contiguous(tensor)) {
    mag_tensor_incref(tensor);
    stage->host = tensor;
    stage->is_alias = true;
    return MAG_OK;
  }
  if (load)
    return mag_tensor_is_host(tensor) ? mag_contiguous(err, &stage->host, tensor) : mag_transfer(err, &stage->host, tensor, mag_device(CPU, 0));
  return mag_empty(err, &stage->host, tensor->ctx, tensor->meta.dtype, tensor->meta.coords.rank, tensor->meta.coords.shape, mag_device(CPU, 0));
}

mag_status_t mag_comm_host_stage_commit(mag_error_t *err, mag_comm_host_staged_t *stage) {
  if (stage->is_alias) return MAG_OK;
  return mag_transfer_(err, stage->target, stage->host);
}

mag_status_t mag_comm_host_stage_alloc(
  mag_error_t *err,
  mag_context_t *ctx,
  mag_dtype_t dtype,
  int64_t numel,
  mag_tensor_t **out
) {
  numel = mag_vmax(1, numel);
  return mag_empty(err, out, ctx, dtype, 1, &numel, mag_device(CPU, 0));
}

static void mag_comm_host_nop_release(void *usr) { (void)usr; }
mag_status_t mag_comm_host_stage_borrow(
  mag_error_t *err,
  mag_context_t *ctx,
  void *base,
  mag_dtype_t dtype,
  int64_t numel,
  mag_tensor_t **out
) {
  return mag_borrow_host_buffer(
    err,
    out,
    ctx,
    base,
    numel*mag_type_trait(dtype)->size,
    dtype,
    1,
    &numel,
    true,
    &mag_comm_host_nop_release,
    NULL
  );
}

mag_status_t mag_comm_host_stage_view(
  mag_error_t *err,
  mag_tensor_t *host,
  int64_t offset,
  int64_t numel,
  mag_tensor_t **out
) {
  return mag_strided_view(err, out, host->ctx, host, 1, (int64_t[1]){numel}, (int64_t[1]){1}, offset);
}

mag_status_t mag_comm_host_stage_reduce(
  mag_error_t *err,
  mag_tensor_t *acc,
  mag_tensor_t *src,
  mag_reduce_op_t red_op
) {
  mag_tensor_t *res=NULL;
  mag_status_t stat;
  switch (red_op) {
    case MAG_REDUCE_SUM:
    case MAG_REDUCE_AVG: stat = mag_add_(err, &res, acc, src); break;
    case MAG_REDUCE_PROD: stat = mag_mul_(err, &res, acc, src); break;
    case MAG_REDUCE_AND: stat = mag_and_(err, &res, acc, src); break;
    case MAG_REDUCE_OR: stat = mag_or_(err, &res, acc, src); break;
    case MAG_REDUCE_XOR: stat = mag_xor_(err, &res, acc, src); break;
    case MAG_REDUCE_MIN:
    case MAG_REDUCE_MAX: {
      stat = red_op == MAG_REDUCE_MIN ? mag_min(err, &res, acc, src) : mag_max(err, &res, acc, src);
      if (mag_iserr(stat)) return stat;
      stat = mag_copy_(err, acc, res);
    } break;
    default: return mag_set_error(err, MAG_ERR_PARAM, "comm: unknown reduce op %d", red_op);
  }
  if (res) mag_tensor_decref(res);
  return stat;
}

mag_status_t mag_comm_host_stage_reduce_range(
  mag_error_t *err,
  mag_tensor_t *acc,
  int64_t acc_offs,
  mag_tensor_t *src,
  int64_t src_offs,
  int64_t numel,
  mag_reduce_op_t red_op
) {
  if (mag_unlikely(numel <= 0)) return MAG_OK;
  mag_tensor_t *av=NULL, *sv=NULL;
  mag_status_t stat = mag_comm_host_stage_view(err, acc, acc_offs, numel, &av);
  if (mag_iserr(stat)) return stat;
  stat = mag_comm_host_stage_view(err, src, src_offs, numel, &sv);
  if (mag_iserr(stat)) {
    mag_tensor_decref(av);
    return stat;
  }
  stat = mag_comm_host_stage_reduce(err, av, sv, red_op);
  mag_tensor_decref(av);
  mag_tensor_decref(sv);
  return stat;
}

mag_status_t mag_comm_host_stage_finalize(
  mag_error_t *err,
  mag_tensor_t *acc,
  mag_reduce_op_t red_op,
  uint32_t sz
) {
  if (red_op != MAG_REDUCE_AVG || sz <= 1) return MAG_OK; /* only avg require divison as we use sum(T)/sz for avg */
  mag_dtype_t dtype = acc->meta.dtype;
  mag_scalar_t scal_div = mag_type_category_is_floating_point(dtype) ? mag_scalar_from_float64((double)sz) : mag_scalar_from_int64((int64_t)sz);
  mag_tensor_t *divisor = NULL;
  mag_status_t stat = mag_scalar(err, &divisor, acc->ctx, dtype, scal_div, mag_device(CPU, 0));
  if (mag_iserr(stat)) return stat;
  mag_tensor_t *res = NULL;
  stat = mag_div_(err, &res, acc, divisor); /* acc /= sz */
  if (res) mag_tensor_decref(res);
  mag_tensor_decref(divisor);
  return stat;
}

void *mag_comm_stage_host_buf(mag_tensor_t *host) {
  return (void *)mag_tensor_data_ptr(host);
}

void mag_comm_host_stage_close(mag_comm_host_staged_t *stage) {
  if (stage->host) mag_tensor_decref(stage->host);
  memset(stage, 0, sizeof(*stage));
}
