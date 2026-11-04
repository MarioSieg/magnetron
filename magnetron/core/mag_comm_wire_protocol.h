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

#ifndef MAG_COMM_WIRE_PROTOCOL_H
#define MAG_COMM_WIRE_PROTOCOL_H

#include "mag_def.h"
#include "mag_tcp_socket.h"

/* A small custom binary protocol on TCP to bootstrap distributed comms (: */

#ifdef __cplusplus
extern "C" {
#endif

static MAG_AINLINE void mag_comm_wire_transfer_wu32(uint8_t **wire, uint32_t v) {
  *(*wire)++ = (uint8_t)v;
  *(*wire)++ = (uint8_t)(v>>8);
  *(*wire)++ = (uint8_t)(v>>16);
  *(*wire)++ = (uint8_t)(v>>24);
}

static MAG_AINLINE void mag_comm_wire_transfer_wu64(uint8_t **wire, uint64_t v) {
  mag_comm_wire_transfer_wu32(wire, (uint32_t)v);
  mag_comm_wire_transfer_wu32(wire, (uint32_t)(v>>32));
}

static MAG_AINLINE uint32_t mag_comm_wire_transfer_ru32(const uint8_t **wire) {
  uint32_t v = 0;
  v|=(uint32_t)(*(*wire)++);
  v|=(uint32_t)(*(*wire)++)<<8;
  v|=(uint32_t)(*(*wire)++)<<16;
  v|=(uint32_t)(*(*wire)++)<<24;
  return v;
}

static MAG_AINLINE uint64_t mag_comm_wire_transfer_ru64(const uint8_t **wire) {
  uint64_t v = 0;
  v|=(uint64_t)mag_comm_wire_transfer_ru32(wire);
  v|=(uint64_t)mag_comm_wire_transfer_ru32(wire)<<32;
  return v;
}

#define MAG_COMM_PROTO_MAGIC 0x4d41474du
#define MAG_COMM_PROTO_VER 1
#define MAG_COMM_PROTO_TAG_RAW 0xffff0000u
#define MAG_COMM_PROTO_TAG_BARRIER 0xffff0001u
#define MAG_COMM_PROTO_TAG_SPLIT 0xffff0002u

typedef struct mag_comm_wire_handshake_t {
  uint32_t magic;
  uint32_t version;
  uint32_t rank;
  uint32_t size;
  uint32_t listen_port;
} mag_comm_wire_handshake_t;
mag_static_assert(sizeof(mag_comm_wire_handshake_t) == 5*4);
extern void mag_comm_wire_handshake_serialize(uint8_t (*buf)[sizeof(mag_comm_wire_handshake_t)], const mag_comm_wire_handshake_t *hs);
extern void mag_comm_wire_handshake_deserialize(const uint8_t (*buf)[sizeof(mag_comm_wire_handshake_t)], mag_comm_wire_handshake_t *hs);

typedef struct mag_comm_wire_frame_hdr_t {
  uint32_t magic;
  uint32_t comm_id;
  uint32_t tag;
  uint32_t src;
  uint64_t nb;
} mag_comm_wire_frame_hdr_t;
mag_static_assert(sizeof(mag_comm_wire_frame_hdr_t) == 4*4 + 1*8);
extern void mag_comm_wire_frame_hdr_serialize(uint8_t (*buf)[sizeof(mag_comm_wire_frame_hdr_t)], const mag_comm_wire_frame_hdr_t *hdr);
extern void mag_comm_wire_frame_hdr_deserialize(const uint8_t (*buf)[sizeof(mag_comm_wire_frame_hdr_t)], mag_comm_wire_frame_hdr_t *hdr);

mag_static_assert(!(MAG_TCP_ADDR_STRLEN%sizeof(uint64_t)));
typedef struct mag_comm_wire_endpoint_t {
  uint64_t addr[MAG_TCP_ADDR_STRLEN/sizeof(uint64_t)];
  uint64_t port_aux;
} mag_comm_wire_endpoint_t;
mag_static_assert(sizeof(mag_comm_wire_endpoint_t) == 1*MAG_TCP_ADDR_STRLEN + 1*8);
extern void mag_comm_wire_endpoint_serialize_multi(uint8_t *buf, uint32_t n, const mag_comm_wire_endpoint_t *endpoints);
extern void mag_comm_wire_endpoint_deserialize_multi(const uint8_t *buf, uint32_t n,  mag_comm_wire_endpoint_t *endpoints);

extern mag_status_t mag_comm_send_handshake(mag_error_t *err, mag_tcp_socket_t *sock, uint32_t rank, uint32_t size, uint32_t listen_port);
extern mag_status_t mag_comm_recv_handshake(mag_error_t *err, mag_tcp_socket_t *sock, uint32_t size, mag_comm_wire_handshake_t *hs);

#ifdef __cplusplus
}
#endif
#endif
