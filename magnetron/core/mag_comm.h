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

#ifndef MAG_DISTRIBUTED_H
#define MAG_DISTRIBUTED_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

static MAG_AINLINE void mag_comm_wire_wu32(uint8_t **wire, uint32_t v) {
  *(*wire)++ = (uint8_t)v;
  *(*wire)++ = (uint8_t)(v>>8);
  *(*wire)++ = (uint8_t)(v>>16);
  *(*wire)++ = (uint8_t)(v>>24);
}
static MAG_AINLINE void mag_comm_wire_wu64(uint8_t **wire, uint64_t v) {
  mag_comm_wire_wu32(wire, (uint32_t)v);
  mag_comm_wire_wu32(wire, (uint32_t)(v>>32));
}
static MAG_AINLINE uint32_t mag_comm_wire_ru32(uint8_t **wire) {
  uint32_t v = 0;
  v|=(uint32_t)(*(*wire)++);
  v|=(uint32_t)(*(*wire)++)<<8;
  v|=(uint32_t)(*(*wire)++)<<16;
  v|=(uint32_t)(*(*wire)++)<<24;
  return v;
}
static MAG_AINLINE uint64_t mag_comm_wire_ru64(uint8_t **wire) {
  uint64_t v = 0;
  v|=(uint64_t)mag_comm_wire_ru32(wire);
  v|=(uint64_t)mag_comm_wire_ru32(wire)<<32;
  return v;
}

typedef struct mag_comm_ops_t mag_comm_ops_t;

struct mag_communicator_t {
  uint32_t rank;
  uint32_t size;
  const mag_comm_ops_t *com_ops;
  void *impl;
};

struct mag_comm_ops_t {
  const char *backend_name;
  mag_status_t (*init)(
    mag_error_t *err,
    mag_communicator_t *comm,
    const void *cfg
  );
  mag_status_t (*barrier)(
    mag_error_t *err,
    mag_communicator_t *comm
  );
  mag_status_t (*broadcast)(
    mag_error_t *err,
    mag_communicator_t *comm,
    mag_tensor_t *tensor,
    uint32_t root
  );
  mag_status_t (*reduce)(
    mag_error_t *err,
    mag_communicator_t *comm,
    mag_tensor_t *tensor,
    uint32_t root,
    mag_reduce_op_t red
  );
  mag_status_t (*all_reduce)(
    mag_error_t *err,
    mag_communicator_t *comm,
    mag_tensor_t *tensor,
    mag_reduce_op_t red
  );
  mag_status_t (*all_gather)(
    mag_error_t *err,
    mag_communicator_t *comm,
    mag_tensor_t *out,
    mag_tensor_t *in
  );
  mag_status_t (*reduce_scatter)(
    mag_error_t *err,
    mag_communicator_t *comm,
    mag_tensor_t *out,
    mag_tensor_t *in,
    mag_reduce_op_t red
  );
  mag_status_t (*all_to_all)(
    mag_error_t *err,
    mag_communicator_t *comm,
    mag_tensor_t *out,
    mag_tensor_t *in
  );
  mag_status_t (*all_to_all_v)(
   mag_error_t *err,
   mag_communicator_t *comm,
   mag_tensor_t *out,
   mag_tensor_t *in,
   const size_t *send_counts,
   const size_t *send_offsets,
   const size_t *recv_counts,
   const size_t *recv_offsets
  );
  mag_status_t (*send)(
    mag_error_t *err,
    mag_communicator_t *comm,
    mag_tensor_t *tensor,
    uint32_t dst
  );
  mag_status_t (*recv)(
    mag_error_t *err,
    mag_communicator_t *comm,
    mag_tensor_t *tensor,
    uint32_t src
  );
  void (*destroy)(mag_communicator_t *comm);
};

typedef struct mag_comm_host_staged_t {
  mag_tensor_t *target;
  mag_tensor_t *host;
  bool is_alias;
} mag_comm_host_staged_t;

extern mag_status_t mag_comm_host_stage_open(
  mag_error_t *err,
  mag_comm_host_staged_t *stage,
  mag_tensor_t *tensor,
  bool load
);
extern mag_status_t mag_comm_host_stage_commit(mag_error_t *err, mag_comm_host_staged_t *stage);
extern mag_status_t mag_comm_host_stage_alloc(
  mag_error_t *err,
  mag_context_t *ctx,
  mag_dtype_t dtype,
  int64_t numel,
  mag_tensor_t **out
);
extern mag_status_t mag_comm_host_stage_borrow(
  mag_error_t *err,
  mag_context_t *ctx,
  void *base,
  mag_dtype_t dtype,
  int64_t numel,
  mag_tensor_t **out
);
extern mag_status_t mag_comm_host_stage_view(
  mag_error_t *err,
  mag_tensor_t *host,
  int64_t offset,
  int64_t numel,
  mag_tensor_t **out
);
extern mag_status_t mag_comm_host_stage_reduce(
  mag_error_t *err,
  mag_tensor_t *acc,
  mag_tensor_t *src,
  mag_reduce_op_t red_op
);
extern mag_status_t mag_comm_host_stage_reduce_range(
  mag_error_t *err,
  mag_tensor_t *acc,
  int64_t acc_offs,
  mag_tensor_t *src,
  int64_t src_offs,
  int64_t numel,
  mag_reduce_op_t red_op
);
extern mag_status_t mag_comm_host_stage_finalize(
  mag_error_t *err,
  mag_tensor_t *acc,
  mag_reduce_op_t red_op,
  uint32_t sz
);
extern void *mag_comm_stage_host_buf(mag_tensor_t *host);
extern void mag_comm_host_stage_close(mag_comm_host_staged_t *stage);

#ifdef __cplusplus
}
#endif

#endif
