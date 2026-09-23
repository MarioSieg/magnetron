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

#include "mag_distributed.h"
#include "mag_alloc.h"

#define mag_comm_interface_verify(err_, comm_, op_) \
  do { \
    if (mag_unlikely(!(comm_))) \
      return mag_set_error( \
        (err_), MAG_ERR_COMM, #op_ ": communicator must not be NULL"); \
    if (mag_unlikely(!(comm_)->com_ops)) \
      return mag_set_error( \
        (err_), MAG_ERR_COMM, #op_ ": communicator has no backend ops");  \
    if (mag_unlikely(!(comm_)->com_ops->op_)) \
      return mag_set_error( \
        (err_), MAG_ERR_COMM, \
        #op_ ": operation is not supported by backend '%s'", \
        (comm_)->com_ops->backend_name); \
  } while (0)

#define mag_comm_interface_verify_tensor(err_, tensor_, op_) \
  do { \
    if (mag_unlikely(!(tensor_))) \
      return mag_set_error( \
        (err_), MAG_ERR_COMM, #op_ ": tensor must not be NULL"); \
  } while (0)

#define mag_comm_interface_verify_ra(err_, comm_, rank_, name_, op_) \
  do { \
    if (mag_unlikely((rank_) >= (comm_)->size)) \
      return mag_set_error( \
        (err_), MAG_ERR_COMM, \
        #op_ ": " name_ " rank %u is out of range for communicator size %u",  \
        (rank_), \
        (comm_)->size); \
  } while (0)


mag_status_t mag_comm_init(
  mag_error_t *err,
  mag_communicator_t **out_comm,
  uint32_t rank,
  uint32_t size,
  const char *backend,
  const mag_comm_backend_desc_t *backend_desc
) {
  return MAG_OK;
}

void mag_comm_destroy(mag_communicator_t *comm) {

}

uint32_t mag_comm_rank(mag_communicator_t *comm) { return comm->rank; }

uint32_t mag_comm_size(mag_communicator_t *comm) { return comm->size; }

const char *mag_comm_backend_name(mag_communicator_t *comm) { return comm->com_ops->backend_name; }

mag_status_t mag_comm_barrier(
  mag_error_t *err,
  mag_communicator_t *comm
) {
  mag_comm_interface_verify(err, comm, barrier);
  return (*comm->com_ops->barrier)(err, comm);
}


mag_status_t mag_comm_broadcast(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  uint32_t root
) {
  mag_comm_interface_verify(err, comm, broadcast);
  mag_comm_interface_verify_tensor(err, tensor, broadcast);
  mag_comm_interface_verify_ra(err, comm, root, "root", broadcast);
  return (*comm->com_ops->broadcast)(err, comm, tensor, root);
}

mag_status_t mag_comm_reduce(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  uint32_t root,
  mag_reduce_op_t red
) {
  mag_comm_interface_verify(err, comm, reduce);
  mag_comm_interface_verify_tensor(err, tensor, reduce);
  mag_comm_interface_verify_ra(err, comm, root, "root", reduce);
  return (*comm->com_ops->reduce)(err, comm, tensor, root, red);
}

mag_status_t mag_comm_all_reduce(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  mag_reduce_op_t red
) {
  mag_comm_interface_verify(err, comm, all_reduce);
  mag_comm_interface_verify_tensor(err, tensor, all_reduce);
  return (*comm->com_ops->all_reduce)(err, comm, tensor, red);
}

mag_status_t mag_comm_all_gather(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *out,
  mag_tensor_t *in
) {
  mag_comm_interface_verify(err, comm, all_gather);
  if (mag_unlikely(!out))
    return mag_set_error(err, MAG_ERR_PARAM, "all_gather: output tensor must not be NULL");
  if (mag_unlikely(!in))
    return mag_set_error(err, MAG_ERR_PARAM, "all_gather: input tensor must not be NULL");
  return (*comm->com_ops->all_gather)(err, comm, out, in);
}


mag_status_t mag_comm_reduce_scatter(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *out,
  mag_tensor_t *in,
  mag_reduce_op_t red
) {
  mag_comm_interface_verify(err, comm, reduce_scatter);
  if (mag_unlikely(!out))
    return mag_set_error(err, MAG_ERR_PARAM, "reduce_scatter: output tensor must not be NULL");
  if (mag_unlikely(!in))
    return mag_set_error(err, MAG_ERR_PARAM, "reduce_scatter: input tensor must not be NULL");
  return (*comm->com_ops->reduce_scatter)(err, comm, out, in, red);
}

mag_status_t mag_comm_all_to_all(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *out,
  mag_tensor_t *in
) {
  mag_comm_interface_verify(err, comm, all_to_all);
  if (mag_unlikely(!out))
    return mag_set_error(err, MAG_ERR_PARAM, "all_to_all: output tensor must not be NULL");
  if (mag_unlikely(!in))
    return mag_set_error(err, MAG_ERR_PARAM, "all_to_all: input tensor must not be NULL");
  return (*comm->com_ops->all_to_all)(err, comm, out, in);
}


mag_status_t mag_comm_all_to_all_v(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *out,
  mag_tensor_t *in,
  const size_t *send_counts,
  const size_t *send_offsets,
  const size_t *recv_counts,
  const size_t *recv_offsets
) {
  mag_comm_interface_verify(err, comm, all_to_all_v);
  if (mag_unlikely(!out || !in))
    return mag_set_error(err, MAG_ERR_PARAM, "all_to_all_v: input and output tensors must not be NULL");
  if (mag_unlikely(!send_counts || !send_offsets || !recv_counts || !recv_offsets))
    return mag_set_error(err, MAG_ERR_PARAM, "all_to_all_v: counts and offsets must not be NULL");
  return (*comm->com_ops->all_to_all_v)(err, comm, out, in, send_counts, send_offsets, recv_counts, recv_offsets);
}

mag_status_t mag_comm_send(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  uint32_t dst
) {
  mag_comm_interface_verify(err, comm, send);
  mag_comm_interface_verify_tensor(err, tensor, send);
  mag_comm_interface_verify_ra(err, comm, dst, "destination", send);
  return (*comm->com_ops->send)(err, comm, tensor, dst);
}


mag_status_t mag_comm_recv(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  uint32_t src
) {
  mag_comm_interface_verify(err, comm, recv);
  mag_comm_interface_verify_tensor(err, tensor, recv);
  mag_comm_interface_verify_ra(err, comm, src, "source", recv);
  return (*comm->com_ops->recv)(err, comm, tensor, src);
}

