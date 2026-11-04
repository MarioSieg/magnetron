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

#include "mag_comm_rendezvous.h"

#include "mag_alloc.h"
#include "mag_comm_wire_protocol.h"

typedef struct mag_rendezvous_split_entry_t {
  uint32_t color;
  uint32_t key;
  uint32_t rank;
} mag_rendezvous_split_entry_t;
mag_static_assert(!(3&sizeof(mag_rendezvous_split_entry_t)) && sizeof(mag_rendezvous_split_entry_t) == 3*4);

void mag_rendezvous_base_init(mag_comm_rendezvous_t *rdv, const mag_comm_rendezvous_ops_t *ops, uint32_t rank, uint32_t size, uint32_t id) {
  *rdv = (mag_comm_rendezvous_t){
    .ops=ops,
    .rank=rank,
    .size=size,
    .id=id ? id : 1,
    .split_seq=0
  };
}

mag_status_t mag_comm_rdvz_init(mag_error_t *err, const mag_comm_rendezvous_ops_t *ops, mag_comm_rendezvous_t **out, uint32_t rank, uint32_t size, const mag_comm_rendezvous_desc_t *desc) {
  *out = NULL;
  if (mag_unlikely(!ops || !ops->init))
    return mag_set_error(err, MAG_ERR_COMM, "rendezvous: no rendezvous implementation selected.");
  return (*ops->init)(err, out, rank, size, desc);
}

static int mag_rendezvous_split_entry_cmp(const void *a, const void *b) {
  const mag_rendezvous_split_entry_t *x = a, *y = b;
  if (x->key != y->key) return x->key < y->key ? -1 : 1;
  if (x->rank != y->rank) return x->rank < y->rank ? -1 : 1;
  return 0;
}

mag_status_t mag_comm_rdvz_split(mag_error_t *err, mag_comm_rendezvous_t *parent, mag_comm_rendezvous_t **out_child, uint32_t color, uint32_t key) {
  *out_child=NULL;
  uint32_t n = parent->size;
  mag_rendezvous_split_entry_t *all = (*mag_try_alloc)(NULL, n*sizeof(*all), 0);
  uint32_t *members = (*mag_try_alloc)(NULL, n*sizeof(*members), 0);
  uint8_t *wire = (*mag_try_alloc)(NULL, n*sizeof(mag_rendezvous_split_entry_t), 0);
  mag_status_t stat = MAG_OK;
  if (mag_unlikely(!all || !members || !wire)) {
    stat = mag_set_error(err, MAG_ERR_OOM, "rendezvous: failed to allocate split table.");
    goto cleanup;
  }
  uint8_t mine[sizeof(mag_rendezvous_split_entry_t)];
  uint8_t *mp=mine;
  mag_comm_wire_transfer_wu32(&mp, color);
  mag_comm_wire_transfer_wu32(&mp, key);
  mag_comm_wire_transfer_wu32(&mp, parent->rank);
  stat = mag_rdvz_all_gather(err, parent, mine, wire, sizeof(mine));
  if (mag_iserr(stat)) goto cleanup;
  const uint8_t *rp=wire;
  for (uint32_t i=0; i < n; ++i) {
    mag_rendezvous_split_entry_t *dst=all+i;
    dst->color = mag_comm_wire_transfer_ru32(&rp);
    dst->key = mag_comm_wire_transfer_ru32(&rp);
    dst->rank = mag_comm_wire_transfer_ru32(&rp);
  }
  uint32_t seq = ++parent->split_seq;
  if (color == MAG_COMM_SPLIT_NOCOLOR) goto cleanup;
  uint32_t m=0;
  for (uint32_t i=0; i < n; ++i)
    if (all[i].color == color) all[m++] = all[i];
  qsort(all, m, sizeof(*all), &mag_rendezvous_split_entry_cmp);
  uint32_t child_rank=0;
  for (uint32_t i=0; i < m; ++i) {
    members[i] = all[i].rank;
    if (all[i].rank == parent->rank)
      child_rank = i;
  }
  uint32_t id = (parent->id*0x9E3779B1u)^(seq*0x85EBCA6Bu)^(color*0xC2B2AE35u);
  mag_comm_rendezvous_split_plan_t plan = {
    .color = color,
    .key = key,
    .seq = seq,
    .child_id = id ? id : 1,
    .child_rank = child_rank,
    .child_size = m,
    .ordered_members = members,
  };
  stat = (*parent->ops->split)(err, parent, out_child, &plan);
cleanup:
  if (all) (*mag_alloc)(all, 0, 0);
  if (members) (*mag_alloc)(members, 0, 0);
  if (wire) (*mag_alloc)(wire, 0, 0);
  return stat;
}

mag_status_t mag_rdvz_broadcast(mag_error_t *err, mag_comm_rendezvous_t *rdv, void *buf, size_t nb, uint32_t root) {
  if (mag_unlikely(root >= rdv->size))
    return mag_set_error(err, MAG_ERR_COMM, "rendezvous %s: broadcast root %u is out of range for size %u.", rdv->ops->name, root, rdv->size);
  if (rdv->size == 1) return MAG_OK;
  return (*rdv->ops->broadcast)(err, rdv, buf, nb, root);
}

mag_status_t mag_rdvz_all_gather(mag_error_t *err, mag_comm_rendezvous_t *rdv, const void *in, void *out, size_t nb) {
  if (rdv->size == 1) {
    if (nb && in != out) memmove(out, in, nb);
    return MAG_OK;
  }
  return (*rdv->ops->all_gather)(err, rdv, in, out, nb);
}

mag_status_t mag_rdvz_barrier(mag_error_t *err, mag_comm_rendezvous_t *rdv) {
  if (rdv->size == 1) return MAG_OK;
  return (*rdv->ops->barrier)(err, rdv);
}

void mag_rdvz_destroy(mag_comm_rendezvous_t *rdv) {
  if (rdv && rdv->ops && rdv->ops->shutdown)
    (*rdv->ops->shutdown)(rdv);
}


