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

#ifndef MAG_COMM_HOST_STAGE_H
#define MAG_COMM_HOST_STAGE_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_comm_host_staged_t {
  mag_tensor_t *target;
  mag_tensor_t *host;
  bool is_alias;
} mag_comm_host_staged_t;

extern mag_status_t mag_comm_host_stage_open(mag_error_t *err, mag_comm_host_staged_t *stage, mag_tensor_t *tensor, bool load);
extern mag_status_t mag_comm_host_stage_commit(mag_error_t *err, mag_comm_host_staged_t *stage);
extern mag_status_t mag_comm_host_stage_alloc(mag_error_t *err, mag_context_t *ctx, mag_dtype_t dtype, int64_t numel, mag_tensor_t **out);
extern mag_status_t mag_comm_host_stage_borrow(mag_error_t *err, mag_context_t *ctx, void *base, mag_dtype_t dtype, int64_t numel, mag_tensor_t **out);
extern mag_status_t mag_comm_host_stage_view(mag_error_t *err, mag_tensor_t *host, int64_t offset, int64_t numel, mag_tensor_t **out);
extern mag_status_t mag_comm_host_stage_reduce(mag_error_t *err, mag_tensor_t *acc, mag_tensor_t *src, mag_reduce_op_t red_op);
extern mag_status_t mag_comm_host_stage_reduce_range(mag_error_t *err, mag_tensor_t *acc, int64_t acc_offs, mag_tensor_t *src, int64_t src_offs, int64_t numel, mag_reduce_op_t red_op);
extern mag_status_t mag_comm_host_stage_finalize(mag_error_t *err, mag_tensor_t *acc, mag_reduce_op_t red_op, uint32_t sz);
extern void *mag_comm_stage_host_buf(mag_tensor_t *host);
extern void mag_comm_host_stage_close(mag_comm_host_staged_t *stage);


#ifdef __cplusplus
}
#endif
#endif
