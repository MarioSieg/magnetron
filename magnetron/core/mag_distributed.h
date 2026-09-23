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

#ifdef __cplusplus
}
#endif

#endif
