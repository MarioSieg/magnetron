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

#include "mag_alloc.h"
#include "mag_tcp_socket.h"

struct mag_process_group_t {
  uint32_t rank;
  uint32_t world_size;
  mag_tcp_socket_t **peers;
};

mag_status_t mag_pgroup_init_tcp(
  mag_error_t *err,
  mag_process_group_t **out,
  const char *master_addr,
  uint16_t master_port,
  uint32_t rank,
  uint32_t world_size
) {
  *out = NULL;
  mag_status_t status = MAG_STATUS_OK;

  if (mag_unlikely(out == NULL))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: out ptr is NULL.");
  if (mag_unlikely(master_addr == NULL || !*master_addr))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: master_addr is invalid.");
  if (mag_unlikely(world_size == 0))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: world_size (%u) must be > 0.", world_size);
  if (mag_unlikely(rank >= world_size))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: rank (%u) must be in [0, %u).", rank, world_size);

  mag_tcp_socket_t **peers = (*mag_try_alloc)(NULL, sizeof(*peers)*world_size, 0);
  if (mag_unlikely(peers == NULL))
    return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "process_group: failed to allocate %zu bytes.", sizeof(*peers)*world_size);
  memset(peers, 0, sizeof(*peers)*world_size);
  mag_process_group_t *pgroup = (*mag_try_alloc)(NULL, sizeof(*pgroup), 0);
  if (mag_unlikely(pgroup == NULL)) {
    (*mag_alloc)(peers, 0, 0);
    return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "process_group: failed to allocate %zu bytes.", sizeof(*pgroup));
  }
  *pgroup = (mag_process_group_t){.rank=rank, .world_size=world_size, .peers=peers};

  if (world_size == 1) { *out = pgroup; return MAG_STATUS_OK; }

  mag_tcp_socket_t *listener = NULL;
  if (rank == 0) { /* Master node */
    if (mag_unlikely(!mag_tcp_socket_listen(&listener, master_port, 128))) {
      status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "process_group: rank 0 failed to listen on port %u.", master_port);
      goto cleanup;
    }
    for (uint32_t ra = 1; ra < world_size; ++ra) {
      mag_tcp_socket_t *sock = NULL;
      if (mag_unlikely(!mag_tcp_socket_accept(&sock, listener))) {
        status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "process_group: rank 0 failed to accept peer.");
        goto cleanup;
      }
      uint32_t peer_rank = UINT32_MAX;
      if (mag_unlikely(!mag_tcp_socket_recv_all(sock, &peer_rank, sizeof(peer_rank)))) {
        mag_tcp_socket_close(sock);
        status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "process_group: rank 0 failed to receive peer rank.");
        goto cleanup;
      }
      if (mag_unlikely(!(peer_rank > 0 && peer_rank < world_size && peers[peer_rank] == NULL))) {
        mag_tcp_socket_close(sock);
        status = mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: received invalid or duplicate peer rank %u (expected a unique rank in [1, %u)).", peer_rank, world_size);
        goto cleanup;
      }
      peers[peer_rank] = sock;
    }
    mag_tcp_socket_close(listener);
    listener = NULL;
  } else {
    mag_tcp_socket_t *sock = NULL;
    if (mag_unlikely(!mag_tcp_socket_connect(&sock, master_addr, master_port))) {
      status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "process_group: rank %u failed to connect to %s:%u.", rank, master_addr, master_port);
      goto cleanup;
    }
    if (mag_unlikely(!mag_tcp_socket_send_all(sock, &rank, sizeof(rank)))) {
      mag_tcp_socket_close(sock);
      status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "process_group: rank %u failed to send rank id.", rank);
      goto cleanup;
    }
    *peers = sock;
  }
  *out = pgroup;
  pgroup = NULL;
  peers = NULL;
  return MAG_STATUS_OK;
cleanup:
  if (listener) mag_tcp_socket_close(listener);
  if (peers) {
    for (uint32_t i = 0; i < world_size; ++i)
      if (peers[i]) mag_tcp_socket_close(peers[i]);
    (*mag_alloc)(peers, 0, 0);
  }
  if (pgroup) (*mag_alloc)(pgroup, 0, 0);
  return status;
}

void mag_pgroup_destroy(mag_process_group_t *pgroup) {
  if (mag_unlikely(!pgroup)) return;
  if (pgroup->peers) {
    for (uint32_t i=0; i < pgroup->world_size; ++i)
      if (pgroup->peers[i])
        mag_tcp_socket_close(pgroup->peers[i]);
    (*mag_alloc)(pgroup->peers, 0, 0);
  }
  (*mag_alloc)(pgroup, 0, 0);
}

uint32_t mag_pgroup_rank(const mag_process_group_t *pgroup) { return pgroup->rank; }
uint32_t mag_pgroup_world_size(const mag_process_group_t *pgroup) { return pgroup->world_size; }

mag_status_t mag_pgroup_validate(mag_error_t *err, mag_process_group_t *pgroup) {
  if (mag_unlikely(pgroup == NULL))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: group ptr is NULL.");
  if (mag_unlikely(!(pgroup->rank < pgroup->world_size)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: rank (%u) must be in [0, %u).", pgroup->rank, pgroup->world_size);
  if (mag_unlikely(!(pgroup->world_size > 0)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: world_size (%u) must be > 0.", pgroup->world_size);
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_verify_tensor_is_wireable(mag_error_t *err, mag_tensor_t *tensor) {
  if (mag_unlikely(tensor == NULL))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: tensor ptr is NULL.");
  if (mag_unlikely(!mag_tensor_is_cpu(tensor)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: tensor must be on CPU to be wireable.");
  if (mag_unlikely(!mag_tensor_is_contiguous(tensor)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: tensor must be contiguous to be wireable.");
  if (mag_unlikely(!mag_tensor_is_numeric_typed(tensor)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: tensor must be numeric typed to be wireable.");
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_send_bytes(mag_error_t *err, mag_process_group_t *pgroup, uint32_t dst_rank, const void *buf, size_t nb) {
  mag_status_t status = mag_pgroup_validate(err, pgroup);
  if (mag_iserr(status)) return status;
  if (mag_unlikely(!(dst_rank < pgroup->world_size)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: dst rank (%u) must be in [0, %u).", dst_rank, pgroup->world_size);
  if (mag_unlikely(dst_rank == pgroup->rank))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: dst rank (%u) must != current process group rank (%u).", dst_rank, pgroup->rank);
  if (mag_unlikely(pgroup->peers[dst_rank] == NULL))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: dst rank (%u) socket is invalid.", dst_rank);
  if (mag_unlikely(!mag_tcp_socket_send_all(pgroup->peers[dst_rank], buf, nb)))
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "process_group: failed to send buffer of size %zuB to dst rank (%u)", nb, dst_rank);
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_recv_bytes(mag_error_t *err, mag_process_group_t *pgroup, uint32_t src_rank, void *buf, size_t nb) {
  mag_status_t status = mag_pgroup_validate(err, pgroup);
  if (mag_iserr(status)) return status;
  if (mag_unlikely(!(src_rank < pgroup->world_size)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: src rank (%u) must be in [0, %u).", src_rank, pgroup->world_size);
  if (mag_unlikely(src_rank == pgroup->rank))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: src rank (%u) must != current process group rank (%u).", src_rank, pgroup->rank);
  if (mag_unlikely(pgroup->peers[src_rank] == NULL))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "process_group: src rank (%u) socket is invalid.", src_rank);
  if (mag_unlikely(!mag_tcp_socket_recv_all(pgroup->peers[src_rank], buf, nb)))
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "process_group: failed to recv buffer of size %zuB from src rank (%u)", nb, src_rank);
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_barrier(mag_error_t *err, mag_process_group_t *pgroup) {
  mag_status_t status = mag_pgroup_validate(err, pgroup);
  if (mag_iserr(status)) return status;
  if (pgroup->world_size == 1) return MAG_STATUS_OK;
  uint8_t token=1;
  if (pgroup->rank == 0) { /* Master node */
    for (uint32_t rank=1; rank < pgroup->world_size; ++rank) {
      status = mag_pgroup_recv_bytes(err, pgroup, rank, &token, sizeof(token));
      if (mag_iserr(status)) return status;
    }
    for (uint32_t rank=1; rank < pgroup->world_size; ++rank) {
      status = mag_pgroup_send_bytes(err, pgroup, rank, &token, sizeof(token));
      if (mag_iserr(status)) return status;
    }
  } else {
    status = mag_pgroup_send_bytes(err, pgroup, 0, &token, sizeof(token));
    if (mag_iserr(status)) return status;
    status = mag_pgroup_recv_bytes(err, pgroup, 0, &token, sizeof(token));
    if (mag_iserr(status)) return status;
  }
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_broadcast_(mag_error_t *err, mag_process_group_t *pgroup, mag_tensor_t *x) {
  mag_status_t status = mag_pgroup_validate(err, pgroup);
  if (mag_iserr(status)) return status;
  status = mag_pgroup_verify_tensor_is_wireable(err, x);
  if (mag_iserr(status)) return status;
  if (pgroup->world_size == 1) return MAG_STATUS_OK;
  void *p = (void *)mag_tensor_data_ptr_mut(x);
  size_t nb = (size_t)mag_tensor_numbytes(x);
  if (pgroup->rank == 0) { /* Master node */
    for (uint32_t rank=1; rank < pgroup->world_size; ++rank) {
      status = mag_pgroup_send_bytes(err, pgroup, rank, p, nb);
      if (mag_iserr(status)) return status;
    }
  } else {
    status = mag_pgroup_recv_bytes(err, pgroup, 0, p, nb);
    if (mag_iserr(status)) return status;
  }
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_all_reduce_sum_(mag_error_t *err, mag_process_group_t *pgroup, mag_tensor_t *x) {
  mag_status_t status = mag_pgroup_validate(err, pgroup);
  if (mag_iserr(status)) return status;
  status = mag_pgroup_verify_tensor_is_wireable(err, x);
  if (mag_iserr(status)) return status;
  mag_tensor_t *tmp = NULL;
  if (pgroup->world_size == 1) return MAG_STATUS_OK;
  void *px = (void *)mag_tensor_data_ptr_mut(x);
  size_t nb = mag_tensor_numbytes(x);
  if (pgroup->rank == 0) {
    status = mag_empty_like(err, &tmp, x);
    if (mag_iserr(status))
      goto cleanup;
    void *pt = (void *)mag_tensor_data_ptr_mut(tmp);
    for (uint32_t src=1; src < pgroup->world_size; ++src) {
      status = mag_pgroup_recv_bytes(err, pgroup, src, pt, nb);
      if (mag_iserr(status))
        goto cleanup;
      mag_tensor_t *result = NULL;
      status = mag_add_(err, &result, x, tmp);
      if (mag_iserr(status))
        goto cleanup;
      mag_tensor_decref(result); /* Inplace add, we add into x too */
    }
    for (uint32_t dst=1; dst < pgroup->world_size; ++dst) {
      status = mag_pgroup_send_bytes(err, pgroup, dst, px, nb);
      if (mag_iserr(status))
        goto cleanup;
    }
  } else {
    status = mag_pgroup_send_bytes(err, pgroup, 0, px, nb);
    if (mag_iserr(status))
      goto cleanup;
    status = mag_pgroup_recv_bytes(err, pgroup, 0, px, nb);
    if (mag_iserr(status))
      goto cleanup;
  }
cleanup:
  if (tmp) mag_tensor_decref(tmp);
  return status;
}
