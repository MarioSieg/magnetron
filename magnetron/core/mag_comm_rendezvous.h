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

#ifndef MAG_COMM_H
#define MAG_COMM_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_COMM_SPLIT_NOCOLOR UINT32_MAX

typedef struct mag_comm_rendezvous_ops_t mag_comm_rendezvous_ops_t;

typedef struct mag_comm_rendezvous_t {
  const mag_comm_rendezvous_ops_t *ops;
  uint32_t rank;
  uint32_t size;
  uint32_t id;
  uint32_t split_seq;
} mag_comm_rendezvous_t;

typedef struct mag_comm_rendezvous_split_plan_t {
  uint32_t color;
  uint32_t key;
  uint32_t seq;
  uint32_t child_id;
  uint32_t child_rank;
  uint32_t child_size;
  const uint32_t *ordered_members;
} mag_comm_rendezvous_split_plan_t;

typedef struct mag_comm_rendezvous_desc_t {
  const char *master_addr;
  uint16_t master_port;
  uint32_t timeout_ms;
} mag_comm_rendezvous_desc_t;

struct mag_comm_rendezvous_ops_t {
  const char *name;
  mag_status_t (*init)(mag_error_t *err, mag_comm_rendezvous_t **out, uint32_t world_rank, uint32_t world_size, const mag_comm_rendezvous_desc_t *desc);
  mag_status_t (*split)(mag_error_t *err, mag_comm_rendezvous_t *parent, mag_comm_rendezvous_t **out_child, const mag_comm_rendezvous_split_plan_t *split_plan);
  mag_status_t (*broadcast)(mag_error_t *err, mag_comm_rendezvous_t *rdv, void *buf, size_t nb, uint32_t root);
  mag_status_t (*all_gather)(mag_error_t *err, mag_comm_rendezvous_t *rdv, const void *in, void *out, size_t nb);
  mag_status_t (*barrier)(mag_error_t *err, mag_comm_rendezvous_t *rdv);
  void (*shutdown)(mag_comm_rendezvous_t *rdv);
};

extern void mag_rendezvous_base_init(mag_comm_rendezvous_t *rdv, const mag_comm_rendezvous_ops_t *ops, uint32_t rank, uint32_t size, uint32_t id);
extern mag_status_t mag_comm_rdvz_init(mag_error_t *err, const mag_comm_rendezvous_ops_t *ops, mag_comm_rendezvous_t **out, uint32_t rank, uint32_t size, const mag_comm_rendezvous_desc_t *desc);
extern mag_status_t mag_comm_rdvz_split(mag_error_t *err, mag_comm_rendezvous_t *parent, mag_comm_rendezvous_t **out_child, uint32_t color, uint32_t key);
extern mag_status_t mag_rdvz_broadcast(mag_error_t *err, mag_comm_rendezvous_t *rdv, void *buf, size_t nb, uint32_t root);
extern mag_status_t mag_rdvz_all_gather(mag_error_t *err, mag_comm_rendezvous_t *rdv, const void *in, void *out, size_t nb);
extern mag_status_t mag_rdvz_barrier(mag_error_t *err, mag_comm_rendezvous_t *rdv);
extern void mag_rdvz_destroy(mag_comm_rendezvous_t *rdv);

/* Builtin always available rendezvous backends */
extern mag_comm_rendezvous_ops_t mag_rdvz_ops_builtin_tcp;
extern mag_comm_rendezvous_ops_t mag_rdvz_ops_builtin_shm;


#ifdef __cplusplus
}
#endif
#endif
