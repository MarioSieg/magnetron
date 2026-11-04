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

#ifndef MAG_COMM_BOOTSTRAP_H
#define MAG_COMM_BOOTSTRAP_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_comm_bootstrap_t mag_comm_bootstrap_t;

extern mag_status_t mag_comm_bootstrap_init(
  mag_error_t *err,
  mag_comm_bootstrap_t **out,
  uint32_t world_rank,
  uint32_t world_size,
  const char *master_addr,
  uint16_t master_port,
  uint32_t timeout_ms
);
extern uint32_t mag_comm_bootstrap_rank(const mag_comm_bootstrap_t *bs);
extern uint32_t mag_comm_bootstrap_size(const mag_comm_bootstrap_t *bs);
extern uint32_t mag_comm_bootstrap_world_rank(const mag_comm_bootstrap_t *bs, uint32_t rank);
extern mag_status_t mag_comm_bootstrap_send(mag_error_t *err, mag_comm_bootstrap_t *bs, uint32_t dst, uint32_t tag, const void *buf, size_t nb);
extern mag_status_t mag_comm_bootstrap_recv(mag_error_t *err, mag_comm_bootstrap_t *bs, uint32_t src, uint32_t tag, void *buf, size_t nb);
extern mag_status_t mag_comm_bootstrap_xchg(
  mag_error_t *err,
  mag_comm_bootstrap_t *bs,
  uint32_t tag,
  uint32_t dst,
  const void *send_buf,
  size_t send_nb,
  uint32_t src,
  void *recv_buf,
  size_t recv_nb
);
extern mag_status_t mag_comm_bootstrap_broadcast(mag_error_t *err, mag_comm_bootstrap_t *bs, void *buf, size_t nb, uint32_t root);
extern mag_status_t mag_comm_bootstrap_all_gather(mag_error_t *err, mag_comm_bootstrap_t *bs, const void *in, void *out, size_t nb);
extern mag_status_t mag_comm_bootstrap_barrier(mag_error_t *err, mag_comm_bootstrap_t *bs);
extern mag_status_t mag_comm_bootstrap_split(mag_error_t *err, mag_comm_bootstrap_t *parent, mag_comm_bootstrap_t **out_child, uint32_t color, uint32_t key);
extern void  mag_comm_bootstrap_destroy(mag_comm_bootstrap_t *bs);

#ifdef __cplusplus
}
#endif
#endif
