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

#include "mag_comm_bootstrap.h"
#include "mag_comm_wire_protocol.h"
#include "mag_alloc.h"
#include "mag_rc.h"
#include "mag_tcp_socket.h"

typedef struct mag_comm_mesh_t {
  MAG_RC_INJECT_HEADER;
  uint32_t world_rank;
  uint32_t world_size;
  mag_tcp_socket_t **links;
  uint32_t split_seq;
} mag_comm_mesh_t;
MAG_RC_OBJECT_IS_VALID(mag_comm_mesh_t);

static mag_status_t mag_comm_mesh_dtor(void *self) {
  mag_comm_mesh_t *mesh = self;
  for (uint32_t i=0; i < mesh->world_size; ++i)
    if (mesh->links[i]) mag_tcp_socket_close(mesh->links[i]);
  (*mag_alloc)(mesh->links, 0, 0);
  (*mag_alloc)(mesh, 0, 0);
  return MAG_OK;
}

static mag_status_t mag_comm_mesh_alloc(mag_error_t *err, mag_comm_mesh_t **out, uint32_t rank, uint32_t size) {
  mag_comm_mesh_t *mesh = (*mag_try_alloc)(NULL, sizeof(*mesh), 0);
  if (mag_unlikely(!mesh)) return mag_set_error(err, MAG_ERR_OOM, "comm: failed to allocate rank mesh.");
  mag_tcp_socket_t **links = (*mag_try_alloc)(NULL, size*sizeof(*links), 0);
  if (mag_unlikely(!links)) {
    (*mag_alloc)(mesh, 0, 0);
    return mag_set_error(err, MAG_ERR_OOM, "comm: failed to allocate rank mesh links.");
  }
  memset(links,0,size*sizeof(*links));
  *mesh = (mag_comm_mesh_t) {
    .world_rank = rank,
    .world_size = size,
    .links = links,
    .split_seq = 0
  };
  mag_rc_init_object(mesh, &mag_comm_mesh_dtor);
  *out = mesh;
  return MAG_OK;
}

struct mag_comm_bootstrap_t {
  mag_comm_mesh_t *mesh;
  uint32_t rank;
  uint32_t size;
  uint32_t id;
  uint32_t *world_rank_table;
};

mag_status_t mag_comm_bootstrap_init(
  mag_error_t *err,
  mag_comm_bootstrap_t **out,
  uint32_t world_rank,
  uint32_t world_size,
  const char *master_addr,
  uint16_t master_port,
  uint32_t timeout_ms
) {
  *out = NULL;
  mag_comm_mesh_t *mesh=NULL;
  if (mag_iserr(mag_comm_mesh_alloc(err, &mesh, world_rank, world_size)))
    return mag_set_error(err, MAG_ERR_OOM, "comm: failed to allocate rank mesh.");
  mag_status_t stat=MAG_OK;
  mag_tcp_socket_t *listener=NULL;
  mag_comm_wire_endpoint_t *tab=NULL;
  uint8_t *wire=NULL;
  if (world_size > 1) {
    tab = (*mag_try_alloc)(NULL, sizeof(*tab)*world_size, 0);
    wire = (*mag_try_alloc)(NULL, sizeof(mag_comm_wire_endpoint_t)*world_size, 0);
    if (mag_unlikely(!tab || !wire)) {
      stat = mag_set_error(err, MAG_ERR_OOM, "comm: failed to allocate endpoint table.");
      goto cleanup;
    }
    memset(tab,0, sizeof(*tab)*world_size);
    int backlog = mag_vmax(world_size, 128);
    if (world_rank == 0) { /* Coordinator */
      if (mag_unlikely(!mag_tcp_socket_listen(&listener, master_port, backlog))) {
        stat = mag_set_error(err, MAG_ERR_IO, "comm: rank 0 failed to listen on port %u; is the port already in use?", master_port);
        goto cleanup;
      }
      snprintf((char *)&tab[0].addr, sizeof(tab[0].addr), "%s", master_addr);
      tab[0].port_aux = master_port;
      for (uint32_t i=1; i < world_size; ++i) {
        mag_tcp_socket_t *sock=NULL;
        if (mag_unlikely(!mag_tcp_socket_accept(&sock, listener))) {
          stat = mag_set_error(err, MAG_ERR_IO, "comm: rank 0 failed to accept a peer connection.");
          goto cleanup;
        }
        mag_comm_wire_handshake_t handshake = {0};
        if (mag_iserr((stat = mag_comm_recv_handshake(err, sock, world_size, &handshake)))) {
          mag_tcp_socket_close(sock);
          goto cleanup;
        }
        if (mag_unlikely(handshake.rank == 0 || mesh->links[handshake.rank])) {
          mag_tcp_socket_close(sock);
          stat = mag_set_error(err, MAG_ERR_COMM, "comm: rank %u registered twice - every rank must be unique.", handshake.rank);
          goto cleanup;
        }
        if (mag_unlikely(!mag_tcp_socket_peer_addr(sock, (char(*)[MAG_TCP_ADDR_STRLEN])&tab[handshake.rank].addr))) {
          mag_tcp_socket_close(sock);
          stat = mag_set_error(err, MAG_ERR_IO, "comm: failed to resolve the address of rank %u.", handshake.rank);
          goto cleanup;
        }
        tab[handshake.rank].port_aux = handshake.listen_port;
        mesh->links[handshake.rank] = sock;
      }
      mag_comm_wire_endpoint_serialize_multi(wire, world_size, tab);
      for (uint32_t i=1; i < world_size; ++i) {
        if (mag_unlikely(!mag_tcp_socket_send_all(mesh->links[i], wire, sizeof(*tab)*world_size))) {
          stat = mag_set_error(err, MAG_ERR_IO, "comm: rank 0 failed to send the endpoint table to rank %u.", i);
          goto cleanup;
        }
      }
    } else { /* Peer */
      if (mag_unlikely(!mag_tcp_socket_listen(&listener, 0, backlog))) {
        stat = mag_set_error(err, MAG_ERR_IO, "comm: rank %u failed to open a listening socket.", world_rank);
        goto cleanup;
      }
      uint16_t listen_port = mag_tcp_socket_local_port(listener);
      mag_tcp_socket_t *master=NULL;
      if (mag_unlikely(!mag_tcp_socket_connect(&master, master_addr, master_port, timeout_ms))) {
        stat = mag_set_error(err, MAG_ERR_IO, "comm: rank %u failed to connect to rank 0 at %s:%u within %u ms.", world_rank, master_addr, master_port, timeout_ms);
        goto cleanup;
      }
      *mesh->links = master;
      if (mag_iserr(mag_comm_send_handshake(err, master, world_rank, world_size, listen_port))) {
        stat = mag_set_error(err, MAG_ERR_IO, "comm: rank %u failed to send the handshake to rank 0.", world_rank);
        goto cleanup;
      }
      if (mag_unlikely(!mag_tcp_socket_recv_all(master, wire, sizeof(*tab)*world_size))) {
        stat = mag_set_error(err, MAG_ERR_IO, "comm: rank %u failed to receive the endpoint table; rank 0 may have rejected the handshake.", world_rank);
        goto cleanup;
      }
      mag_comm_wire_endpoint_deserialize_multi(wire, world_size, tab);
      for (uint32_t i=1; i < world_rank; ++i) {
        mag_tcp_socket_t *sock=NULL;
        if (mag_unlikely(!mag_tcp_socket_accept(&sock, listener))) {
          stat = mag_set_error(err, MAG_ERR_IO, "comm: rank %u failed to accept a peer connection.", world_rank);
          goto cleanup;
        }
        mag_comm_wire_handshake_t handshake = {0};
        if (mag_iserr((stat = mag_comm_recv_handshake(err, sock, world_size, &handshake)))) {
          mag_tcp_socket_close(sock);
          goto cleanup;
        }
        if (mag_unlikely(handshake.rank == 0 || handshake.rank >= world_rank || mesh->links[handshake.rank])) {
          mag_tcp_socket_close(sock);
          stat = mag_set_error(err, MAG_ERR_COMM, "comm: rank %u received an unexpected connection from rank %u.", world_rank, handshake.rank);
          goto cleanup;
        }
        mesh->links[handshake.rank] = sock;
      }
      for (uint32_t j=1+world_rank; j < world_size; ++j) {
        mag_tcp_socket_t *sock=NULL;
        if (mag_unlikely(!mag_tcp_socket_connect(&sock, (const char *)tab[j].addr, (uint16_t)tab[j].port_aux, timeout_ms))) {
          stat = mag_set_error(err, MAG_ERR_IO, "comm: rank %u failed to connect to rank %u at %s:%" PRIu64".", world_rank, j, (const char *)tab[j].addr, tab[j].port_aux);
          goto cleanup;
        }
        mesh->links[j] = sock;
        if (mag_iserr(mag_comm_send_handshake(err, sock, world_rank, world_size, listen_port))) {
          stat = mag_set_error(err, MAG_ERR_IO, "comm: rank %u failed to send the handshake to rank %u.", world_rank, j);
          goto cleanup;
        }
      }
    }
    mag_tcp_socket_close(listener), listener=NULL;
    (*mag_alloc)(tab,0,0);
    (*mag_alloc)(wire,0,0);
    tab=NULL,wire=NULL;
  }
  return MAG_OK;
cleanup:
  if (listener) mag_tcp_socket_close(listener);
  if (tab) (*mag_alloc)(tab, 0, 0);
  if (wire) (*mag_alloc)(wire, 0, 0);
  mag_rc_decref(mesh);
  return stat;
}

uint32_t mag_comm_bootstrap_rank(const mag_comm_bootstrap_t *bs) { return bs->rank; }
uint32_t mag_comm_bootstrap_size(const mag_comm_bootstrap_t *bs) {  return bs->size; }
uint32_t mag_comm_bootstrap_world_rank(const mag_comm_bootstrap_t *bs, uint32_t rank) {  return bs->world_rank_table[rank]; }

mag_status_t mag_comm_bootstrap_send(mag_error_t *err, mag_comm_bootstrap_t *bs, uint32_t dst, uint32_t tag, const void *buf, size_t nb) {
  return MAG_OK;
}

mag_status_t mag_comm_bootstrap_recv(mag_error_t *err, mag_comm_bootstrap_t *bs, uint32_t src, uint32_t tag, void *buf, size_t nb) {
  return MAG_OK;
}

mag_status_t mag_comm_bootstrap_xchg(
  mag_error_t *err,
  mag_comm_bootstrap_t *bs,
  uint32_t tag,
  uint32_t dst,
  const void *send_buf,
  size_t send_nb,
  uint32_t src,
  void *recv_buf,
  size_t recv_nb
) {
  return MAG_OK;
}

mag_status_t mag_comm_bootstrap_broadcast(mag_error_t *err, mag_comm_bootstrap_t *bs, void *buf, size_t nb, uint32_t root) {
  return MAG_OK;
}

mag_status_t mag_comm_bootstrap_all_gather(mag_error_t *err, mag_comm_bootstrap_t *bs, const void *in, void *out, size_t nb) {
  return MAG_OK;
}

mag_status_t mag_comm_bootstrap_barrier(mag_error_t *err, mag_comm_bootstrap_t *bs) {
  return MAG_OK;
}

mag_status_t mag_comm_bootstrap_split(mag_error_t *err, mag_comm_bootstrap_t *parent, mag_comm_bootstrap_t **out_child, uint32_t color, uint32_t key) {
  return MAG_OK;
}

void  mag_comm_bootstrap_destroy(mag_comm_bootstrap_t *bs) {
  if (!bs) return;
  mag_rc_decref(bs->mesh);
  (*mag_alloc)(bs->world_rank_table, 0, 0);
  (*mag_alloc)(bs, 0, 0);
}

