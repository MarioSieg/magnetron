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

#include "mag_tcp_socket.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_process_group_t mag_process_group_t;

extern MAG_EXPORT mag_status_t mag_pg_init_tcp(
    mag_error_t *err,
    mag_process_group_t **out,
    const char *master_addr,
    uint16_t master_port,
    uint32_t rank,
    uint32_t world_size
);
extern MAG_EXPORT void mag_pg_destroy(mag_process_group_t *pgroup);
extern MAG_EXPORT uint32_t mag_pg_rank(const mag_process_group_t *pgroup);
extern MAG_EXPORT uint32_t mag_pg_world_size(const mag_process_group_t *pgroup);
extern MAG_EXPORT mag_status_t mag_pg_validate(mag_error_t *err, mag_process_group_t *pgroup);
extern MAG_EXPORT mag_status_t mag_pg_verify_tensor_is_wireable(mag_error_t *err, mag_process_group_t *pgroup, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_pg_send_bytes(mag_error_t *err, mag_process_group_t *pgroup, uint32_t dst_rank, const void *buf, size_t nb);
extern MAG_EXPORT mag_status_t mag_pg_recv_bytes(mag_error_t *err, mag_process_group_t *pgroup, uint32_t src_rank, void *buf, size_t nb);
extern MAG_EXPORT mag_status_t mag_pg_barrier(mag_error_t *err, mag_process_group_t *pgroup);
extern MAG_EXPORT mag_status_t mag_pg_broadcast_(mag_error_t *err, mag_process_group_t *pgroup, mag_tensor_t *x, int root);
extern MAG_EXPORT mag_status_t mag_pg_all_reduce_sum_(mag_error_t *err, mag_process_group_t *pgroup, mag_tensor_t *x);

#ifdef __cplusplus
}
#endif

#endif
