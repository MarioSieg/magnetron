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

/* A small custom binary protocol on TCP to bootstrap distributed comms (: All integers are stored as LE */

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

static MAG_AINLINE void mag_comm_wire_transfer_wbytes(uint8_t **wire, const void *src, size_t nb) {
  memcpy(*wire, src, nb);
  *wire += nb;
}

static MAG_AINLINE void mag_comm_wire_transfer_rbytes(const uint8_t **wire, void *dst, size_t nb) {
  memcpy(dst, *wire, nb);
  *wire += nb;
}

#define MAG_COMM_PROTO_MAGIC 0x4d41474du
#define MAG_COMM_PROTO_VER 2
#define MAG_COMM_PROTO_TAG_RESERVED 0xffff0000u /* Tags >= this are reserved for internal use only */
#define MAG_COMM_PROTO_TAG_RAW 0xffff0000u
#define MAG_COMM_PROTO_TAG_BARRIER 0xffff0001u
#define MAG_COMM_PROTO_TAG_SPLIT 0xffff0002u
#define MAG_COMM_PROTO_TAG_BROADCAST 0xffff0003u
#define MAG_COMM_PROTO_TAG_ALL_GATHER 0xffff0004u

typedef struct mag_comm_wire_handshake_t {
  uint32_t magic;
  uint32_t version;
  uint32_t rank;
  uint32_t size;
  uint32_t listen_port;
  uint32_t reserved;
  uint64_t job_id;                    /* Rejects stray peers from other jobs or crashes */
  char addr[MAG_TCP_ADDR_STRLEN];     /* Advertised address if empty use master addr*/
} mag_comm_wire_handshake_t;
mag_static_assert(sizeof(mag_comm_wire_handshake_t) == 6*4 + 1*8 + MAG_TCP_ADDR_STRLEN);
extern void mag_comm_wire_handshake_serialize(uint8_t (*buf)[sizeof(mag_comm_wire_handshake_t)], const mag_comm_wire_handshake_t *hs);
extern void mag_comm_wire_handshake_deserialize(const uint8_t (*buf)[sizeof(mag_comm_wire_handshake_t)], mag_comm_wire_handshake_t *hs);

typedef struct mag_comm_wire_frame_hdr_t {
  uint32_t magic;
  uint32_t tag;
  uint64_t comm_id;
  uint32_t src;
  uint32_t reserved;
  uint64_t nb;
} mag_comm_wire_frame_hdr_t;
mag_static_assert(sizeof(mag_comm_wire_frame_hdr_t) == 4*4 + 2*8);
extern void mag_comm_wire_frame_hdr_serialize(uint8_t (*buf)[sizeof(mag_comm_wire_frame_hdr_t)], const mag_comm_wire_frame_hdr_t *hdr);
extern void mag_comm_wire_frame_hdr_deserialize(const uint8_t (*buf)[sizeof(mag_comm_wire_frame_hdr_t)], mag_comm_wire_frame_hdr_t *hdr);

typedef struct mag_comm_wire_endpoint_t {
  char addr[MAG_TCP_ADDR_STRLEN];
  uint64_t port_aux;
} mag_comm_wire_endpoint_t;
mag_static_assert(sizeof(mag_comm_wire_endpoint_t) == 1*MAG_TCP_ADDR_STRLEN + 1*8);
extern void mag_comm_wire_endpoint_serialize_multi(uint8_t *buf, uint32_t n, const mag_comm_wire_endpoint_t *endpoints);
extern void mag_comm_wire_endpoint_deserialize_multi(const uint8_t *buf, uint32_t n,  mag_comm_wire_endpoint_t *endpoints);

typedef enum mag_comm_handshake_result_t {
  MAG_COMM_HANDSHAKE_OK,
  MAG_COMM_HANDSHAKE_IO,       /* Connection failed or timed out */
  MAG_COMM_HANDSHAKE_FOREIGN,  /* Wrong magic/version/job: a stray connection which can be dropped */
  MAG_COMM_HANDSHAKE_INVALID   /* Same job but inconsistent configuration */
} mag_comm_handshake_result_t;

extern MAG_EXPORT mag_status_t mag_comm_send_handshake(mag_error_t *err, mag_tcp_socket_t *sock, const mag_comm_wire_handshake_t *hs, uint32_t timeout_ms);
extern MAG_EXPORT mag_comm_handshake_result_t mag_comm_recv_handshake(mag_error_t *err, mag_tcp_socket_t *sock, uint32_t size, uint64_t job_id, mag_comm_wire_handshake_t *hs, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif
#endif
