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
  mag_tcp_socket_t *peers;
};

mag_status_t mag_pgroup_init_tcp(
  mag_error_t *err,
  mag_process_group_t **out,
  const char *master_addr,
  uint16_t master_port,
  uint32_t rank,
  uint32_t world_size
) {
  mag_contract(err, ERR_INVALID_PARAM, {}, out != NULL, "process_group: out ptr is NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, master_addr != NULL && *master_addr, "process_group: master_addr is invalid.");
  mag_contract(err, ERR_INVALID_PARAM, {}, world_size > 0, "process_group: world_size (%u) must be > 0.", world_size);
  mag_contract(err, ERR_INVALID_PARAM, {}, rank < world_size, "process_group: rank (%u) must be in [0, %u).", rank, world_size);
  *out = NULL;
  mag_tcp_socket_t *peers = (*mag_alloc)(0, sizeof(*peers)*world_size, 0);
  for (uint32_t i=0; i < world_size; ++i) peers[i] = mag_tcp_socket_invalid();
  mag_process_group_t *pgroup = (*mag_alloc)(0, sizeof(*pgroup), 0);
  *pgroup = (mag_process_group_t) {
    .rank = rank,
    .world_size = world_size,
    .peers = peers
  };
  if (world_size == 1) {
    *out = pgroup;
    return MAG_STATUS_OK;
  }
  if (rank == 0) { /* Master node */
    mag_tcp_socket_t listener;
    mag_try_or(mag_tcp_socket_listen(&listener, master_port, 128), {
      (*mag_alloc)(peers, 0, 0);
      (*mag_alloc)(pgroup, 0, 0);
    });
    for (uint32_t ra=1; ra < world_size; ++ra) {
      mag_tcp_socket_t sock;
      mag_try_or(mag_tcp_socket_accept(&sock, listener), {
        (*mag_alloc)(peers, 0, 0);
        (*mag_alloc)(pgroup, 0, 0);
        mag_tcp_socket_close(listener);
      });
      uint32_t peer_rank = UINT32_MAX;
      mag_try_or(mag_tcp_socket_recv_all(sock, &peer_rank, sizeof(peer_rank)), {
        (*mag_alloc)(peers, 0, 0);
        (*mag_alloc)(pgroup, 0, 0);
        mag_tcp_socket_close(listener);
        mag_tcp_socket_close(sock);
      });
      mag_contract(err, ERR_INVALID_PARAM, {
        (*mag_alloc)(peers, 0, 0);
        (*mag_alloc)(pgroup, 0, 0);
        mag_tcp_socket_close(listener);
        mag_tcp_socket_close(sock);
      }, peer_rank > 0 && peer_rank < world_size && !mag_tcp_socket_is_open(peers[peer_rank]), "Error meow");
      peers[peer_rank] = sock;
    }
    mag_tcp_socket_close(listener);
  } else {
    mag_tcp_socket_t sock;
    mag_try_or(mag_tcp_socket_connect(&sock, master_addr, master_port), {
      (*mag_alloc)(peers, 0, 0);
      (*mag_alloc)(pgroup, 0, 0);
    });
    mag_try_or(mag_tcp_socket_send_all(sock, &rank, sizeof(rank)), {
      (*mag_alloc)(peers, 0, 0);
      (*mag_alloc)(pgroup, 0, 0);
      mag_tcp_socket_close(sock);
    });
    *peers = sock;
  }
  *out = pgroup;
  return MAG_STATUS_OK;
}

void mag_pgroup_destroy(mag_process_group_t *pgroup) {
  if (mag_unlikely(!pgroup)) return;
  if (pgroup->peers) {
    for (uint32_t i=0; i < pgroup->world_size; ++i)
      if (mag_tcp_socket_is_open(pgroup->peers[i]))
        mag_tcp_socket_close(pgroup->peers[i]);
    (*mag_alloc)(pgroup->peers, 0, 0);
  }
  (*mag_alloc)(pgroup, 0, 0);
}

uint32_t mag_pgroup_rank(const mag_process_group_t *pgroup) { return pgroup->rank; }
uint32_t mag_pgroup_world_size(const mag_process_group_t *pgroup) { return pgroup->world_size; }

mag_status_t mag_pgroup_validate(mag_error_t *err, mag_process_group_t *pgroup) {
  mag_contract(err, ERR_INVALID_PARAM, {}, pgroup != NULL, "progres_group: group ptr is NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, pgroup->rank < pgroup->world_size, "process_group: rank (%u) must be in [0, %u).", pgroup->rank, pgroup->world_size);
  mag_contract(err, ERR_INVALID_PARAM, {}, pgroup->world_size > 0, "process_group: world_size (%u) must be > 0.", pgroup->world_size);
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_verify_tensor_is_wireable(mag_error_t *err, mag_tensor_t *tensor) {
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor != NULL, "process_group: tensor ptr is NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, mag_tensor_is_cpu(tensor), "process_group: tensor must be on CPU to be wireable.");
  mag_contract(err, ERR_INVALID_PARAM, {}, mag_tensor_is_contiguous(tensor), "process_group: tensor must be contiguous to be wireable.");
  mag_contract(err, ERR_INVALID_PARAM, {}, mag_tensor_is_numeric_typed(tensor), "process_group: tensor must be numeric typed to be wireable.");
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_send_bytes(mag_error_t *err, mag_process_group_t *pgroup, uint32_t dst_rank, const void *buf, size_t nb) {
  mag_try(mag_pgroup_validate(err, pgroup));
  mag_contract(err, ERR_INVALID_PARAM, {}, dst_rank < pgroup->world_size, "process_group: dst rank (%u) must be in [0, %u).", dst_rank, pgroup->world_size);
  mag_contract(err, ERR_INVALID_PARAM, {}, dst_rank != pgroup->rank, "process_group: dst rank (%u) must != current process group rank (%u).", dst_rank, pgroup->rank);
  mag_contract(err, ERR_INVALID_PARAM, {}, mag_tcp_socket_is_open(pgroup->peers[dst_rank]), "process_group: dst rank (%u) socket is invalid.", dst_rank);
  mag_contract(err, ERR_INVALID_PARAM, {}, mag_tcp_socket_send_all(pgroup->peers[dst_rank], buf, nb), "progress_group: failed to send buffer of size %zuB to dst rank (%u)", nb, dst_rank);
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_recv_bytes(mag_error_t *err, mag_process_group_t *pgroup, uint32_t src_rank, void *buf, size_t nb) {
  mag_try(mag_pgroup_validate(err, pgroup));
  mag_contract(err, ERR_INVALID_PARAM, {}, src_rank < pgroup->world_size, "process_group: src rank (%u) must be in [0, %u).", src_rank, pgroup->world_size);
  mag_contract(err, ERR_INVALID_PARAM, {}, src_rank != pgroup->rank, "process_group: src rank (%u) must != current process group rank (%u).", src_rank, pgroup->rank);
  mag_contract(err, ERR_INVALID_PARAM, {}, mag_tcp_socket_is_open(pgroup->peers[src_rank]), "process_group: src rank (%u) socket is invalid.", src_rank);
  mag_contract(err, ERR_INVALID_PARAM, {}, mag_tcp_socket_recv_all(pgroup->peers[src_rank], buf, nb), "progress_group: failed to send buffer of size %zuB to src rank (%u)", nb, src_rank);
  return MAG_STATUS_OK;
}


mag_status_t mag_pgroup_barrier(mag_error_t *err, mag_process_group_t *pgroup) {
  mag_try(mag_pgroup_validate(err, pgroup));
  if (pgroup->world_size == 1) return MAG_STATUS_OK;
  uint8_t token=1;
  if (pgroup->rank == 0) { /* Master node */
    for (uint32_t rank=1; rank < pgroup->world_size; ++rank) {
      mag_try(mag_pgroup_recv_bytes(err, pgroup, rank, &token,sizeof(token)));
    }
    for (uint32_t rank=1; rank < pgroup->world_size; ++rank) {
      mag_try(mag_pgroup_send_bytes(err, pgroup, rank, &token,sizeof(token)));
    }
  } else {
    mag_try(mag_pgroup_send_bytes(err, pgroup, 0, &token,sizeof(token)));
    mag_try(mag_pgroup_recv_bytes(err, pgroup, 0, &token,sizeof(token)));
  }
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_broadcast_(mag_error_t *err, mag_process_group_t *pgroup, mag_tensor_t *x) {
  mag_try(mag_pgroup_validate(err, pgroup));
  mag_try(mag_pgroup_verify_tensor_is_wireable(err, x));
  if (pgroup->world_size == 1) return MAG_STATUS_OK;
  void *p = (void *)mag_tensor_data_ptr_mut(x);
  size_t nb = (size_t)mag_tensor_numbytes(x);
  if (pgroup->rank == 0) { /* Master node */
    for (uint32_t rank=1; rank < pgroup->world_size; ++rank) {
      mag_try(mag_pgroup_send_bytes(err, pgroup, rank, p, nb));
    }
  } else {
    mag_try(mag_pgroup_recv_bytes(err, pgroup, 0, p, nb));
  }
  return MAG_STATUS_OK;
}

mag_status_t mag_pgroup_all_reduce_sum_(mag_error_t *err, mag_process_group_t *pgroup, mag_tensor_t *x) {
  mag_try(mag_pgroup_validate(err, pgroup));
  mag_try(mag_pgroup_verify_tensor_is_wireable(err, x));
  if (pgroup->world_size == 1) return MAG_STATUS_OK;
  void *px = (void *)mag_tensor_data_ptr_mut(x);
  size_t nb = (size_t)mag_tensor_numbytes(x);
  if (pgroup->rank == 0) {
    mag_tensor_t *tmp = NULL;
    mag_try(mag_empty_like(err, &tmp, x));
    void *pt = (void *)mag_tensor_data_ptr_mut(tmp);
    for (uint32_t src=1; src < pgroup->world_size; ++src) {
      mag_try_or(mag_pgroup_recv_bytes(err, pgroup, src, pt, nb), {
        mag_tensor_decref(tmp);
      });
      mag_tensor_t *result = NULL;
      mag_try_or(mag_add_(err, &result, x, tmp), {
        mag_tensor_decref(tmp);
      });
      mag_tensor_decref(result); /* Inplace add, we add into x too */
    }
    for (uint32_t dst=1; dst < pgroup->world_size; ++dst) {
      mag_try_or(mag_pgroup_send_bytes(err, pgroup, dst, px, nb), {
       mag_tensor_decref(tmp);
     });
    }
    mag_tensor_decref(tmp);
  } else {
    mag_try(mag_pgroup_send_bytes(err, pgroup, 0, px, nb));
    mag_try(mag_pgroup_recv_bytes(err, pgroup, 0, px, nb));
  }
  return MAG_STATUS_OK;
}

