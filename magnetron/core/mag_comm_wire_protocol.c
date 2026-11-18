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

#include "mag_comm_wire_protocol.h"

#include <errno.h>

void mag_comm_wire_handshake_serialize(uint8_t(*buf)[sizeof(mag_comm_wire_handshake_t)], const mag_comm_wire_handshake_t *hs) {
  uint8_t *p=*buf;
  mag_comm_wire_transfer_wu32(&p, hs->magic);
  mag_comm_wire_transfer_wu32(&p, hs->version);
  mag_comm_wire_transfer_wu32(&p, hs->rank);
  mag_comm_wire_transfer_wu32(&p, hs->size);
  mag_comm_wire_transfer_wu32(&p, hs->listen_port);
  mag_comm_wire_transfer_wu32(&p, hs->reserved);
  mag_comm_wire_transfer_wu64(&p, hs->job_id);
  mag_comm_wire_transfer_wbytes(&p, hs->addr, sizeof(hs->addr));
}

void mag_comm_wire_handshake_deserialize(const uint8_t(*buf)[sizeof(mag_comm_wire_handshake_t)], mag_comm_wire_handshake_t *hs) {
  const uint8_t *p=*buf;
  hs->magic = mag_comm_wire_transfer_ru32(&p);
  hs->version = mag_comm_wire_transfer_ru32(&p);
  hs->rank = mag_comm_wire_transfer_ru32(&p);
  hs->size = mag_comm_wire_transfer_ru32(&p);
  hs->listen_port = mag_comm_wire_transfer_ru32(&p);
  hs->reserved = mag_comm_wire_transfer_ru32(&p);
  hs->job_id = mag_comm_wire_transfer_ru64(&p);
  mag_comm_wire_transfer_rbytes(&p, hs->addr, sizeof(hs->addr));
  hs->addr[sizeof(hs->addr)-1] = '\0';
}

void mag_comm_wire_frame_hdr_serialize(uint8_t (*buf)[sizeof(mag_comm_wire_frame_hdr_t)], const mag_comm_wire_frame_hdr_t *hdr) {
  uint8_t *p=*buf;
  mag_comm_wire_transfer_wu32(&p, hdr->magic);
  mag_comm_wire_transfer_wu32(&p, hdr->tag);
  mag_comm_wire_transfer_wu64(&p, hdr->comm_id);
  mag_comm_wire_transfer_wu32(&p, hdr->src);
  mag_comm_wire_transfer_wu32(&p, hdr->reserved);
  mag_comm_wire_transfer_wu64(&p, hdr->nb);
}

void mag_comm_wire_frame_hdr_deserialize(const uint8_t (*buf)[sizeof(mag_comm_wire_frame_hdr_t)], mag_comm_wire_frame_hdr_t *hdr) {
  const uint8_t *p=*buf;
  hdr->magic = mag_comm_wire_transfer_ru32(&p);
  hdr->tag = mag_comm_wire_transfer_ru32(&p);
  hdr->comm_id = mag_comm_wire_transfer_ru64(&p);
  hdr->src = mag_comm_wire_transfer_ru32(&p);
  hdr->reserved = mag_comm_wire_transfer_ru32(&p);
  hdr->nb = mag_comm_wire_transfer_ru64(&p);
}

void mag_comm_wire_endpoint_serialize_multi(uint8_t *buf, uint32_t n, const mag_comm_wire_endpoint_t *endpoints) {
  for (uint32_t i=0; i < n; ++i) {
    mag_comm_wire_transfer_wbytes(&buf, endpoints[i].addr, sizeof(endpoints[i].addr));
    mag_comm_wire_transfer_wu64(&buf, endpoints[i].port_aux);
  }
}

void mag_comm_wire_endpoint_deserialize_multi(const uint8_t *buf, uint32_t n,  mag_comm_wire_endpoint_t *endpoints) {
  for (uint32_t i=0; i < n; ++i) {
    mag_comm_wire_transfer_rbytes(&buf, endpoints[i].addr, sizeof(endpoints[i].addr));
    endpoints[i].addr[sizeof(endpoints[i].addr)-1] = '\0';
    endpoints[i].port_aux = mag_comm_wire_transfer_ru64(&buf);
  }
}

mag_status_t mag_comm_send_handshake(mag_error_t *err, mag_tcp_socket_t *sock, const mag_comm_wire_handshake_t *hs, uint32_t timeout_ms) {
  mag_comm_wire_handshake_t h = *hs;
  h.magic = MAG_COMM_PROTO_MAGIC;
  h.version = MAG_COMM_PROTO_VER;
  h.reserved = 0;
  h.addr[sizeof(h.addr)-1] = '\0';
  uint8_t wire[sizeof(mag_comm_wire_handshake_t)] = {0};
  mag_comm_wire_handshake_serialize(&wire, &h);
  if (mag_unlikely(!mag_tcp_socket_send_all(sock, wire, sizeof(wire), timeout_ms)))
    return mag_set_error(err, MAG_ERR_IO, "comm: failed to send handshake (%s).", strerror(errno));
  return MAG_OK;
}

mag_comm_handshake_result_t mag_comm_recv_handshake(mag_error_t *err, mag_tcp_socket_t *sock, uint32_t size, uint64_t job_id, mag_comm_wire_handshake_t *hs, uint32_t timeout_ms) {
  uint8_t wire[sizeof(mag_comm_wire_handshake_t)] = {0};
  if (mag_unlikely(!mag_tcp_socket_recv_all(sock, wire, sizeof(wire), timeout_ms))) {
    mag_set_error(err, MAG_ERR_IO, "comm: failed to receive peer handshake (%s).", strerror(errno));
    return MAG_COMM_HANDSHAKE_IO;
  }
  mag_comm_wire_handshake_deserialize(&wire, hs);
  if (mag_unlikely(hs->magic != MAG_COMM_PROTO_MAGIC || hs->version != MAG_COMM_PROTO_VER)) {
    mag_set_error(err, MAG_ERR_COMM, "comm: peer handshake has wrong magic or protocol version (got v%u, expected v%u); is the peer running the same magnetron version?", hs->version, MAG_COMM_PROTO_VER);
    return MAG_COMM_HANDSHAKE_FOREIGN;
  }
  if (mag_unlikely(hs->job_id != job_id)) {
    mag_set_error(err, MAG_ERR_COMM, "comm: peer belongs to job %016" PRIx64 " but this rank is in job %016" PRIx64 "; stale rank from an earlier run?", hs->job_id, job_id);
    return MAG_COMM_HANDSHAKE_FOREIGN;
  }
  if (mag_unlikely(hs->size != size)) {
    mag_set_error(err, MAG_ERR_COMM, "comm: peer reports world size %u but this rank expects %u.", hs->size, size);
    return MAG_COMM_HANDSHAKE_INVALID;
  }
  if (mag_unlikely(hs->rank >= size)) {
    mag_set_error(err, MAG_ERR_COMM, "comm: peer rank %u is out of range for world size %u.", hs->rank, size);
    return MAG_COMM_HANDSHAKE_INVALID;
  }
  return MAG_COMM_HANDSHAKE_OK;
}
