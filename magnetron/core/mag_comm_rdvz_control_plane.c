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

#include "mag_comm_rdvz_control_plane.h"

#include "mag_alloc.h"
#include "mag_comm_wire_protocol.h"
#include "mag_hash.h"
#include "mag_os.h"

#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

typedef struct mag_rendezvous_split_entry_t {
  uint32_t color;
  uint32_t key;
  uint32_t rank;
} mag_rendezvous_split_entry_t;

void mag_rendezvous_base_init(mag_comm_rendezvous_t *rdv, const mag_comm_rendezvous_ops_t *ops, uint32_t rank, uint32_t size, uint64_t id) {
  *rdv = (mag_comm_rendezvous_t){
    .ops=ops,
    .rank=rank,
    .size=size,
    .id=id ? id : 1,
    .split_seq=0
  };
}

mag_status_t mag_comm_rdvz_hdr_init(mag_error_t *err, const mag_comm_rendezvous_ops_t *ops, mag_comm_rendezvous_t **out, uint32_t rank, uint32_t size, const mag_comm_rendezvous_desc_t *desc) {
  *out = NULL;
  if (mag_unlikely(!ops || !ops->init))
    return mag_set_error(err, MAG_ERR_COMM, "rendezvous: no rendezvous implementation selected.");
  if (mag_unlikely(!size))
    return mag_set_error(err, MAG_ERR_COMM, "rendezvous %s: world size must be at least 1.", ops->name);
  if (mag_unlikely(rank >= size))
    return mag_set_error(err, MAG_ERR_COMM, "rendezvous %s: rank %u is out of range for world size %u.", ops->name, rank, size);
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
  uint8_t *wire = (*mag_try_alloc)(NULL, n*sizeof(*all), 0);
  mag_status_t stat = MAG_OK;
  if (mag_unlikely(!all || !members || !wire)) {
    stat = mag_set_error(err, MAG_ERR_OOM, "rendezvous: failed to allocate split table.");
    goto cleanup;
  }
  uint8_t mine[sizeof(*all)];
  uint8_t *mp=mine;
  mag_comm_wire_transfer_wu32(&mp, color);
  mag_comm_wire_transfer_wu32(&mp, key);
  mag_comm_wire_transfer_wu32(&mp, parent->rank);
  stat = mag_comm_rdvz_all_gather(err, parent, mine, wire, sizeof(mine));
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
  uint64_t id = mag_hmix64(parent->id^mag_hmix64(((uint64_t)seq<<32)|color));
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

mag_status_t mag_comm_rdvz_broadcast(mag_error_t *err, mag_comm_rendezvous_t *rdv, void *buf, size_t nb, uint32_t root) {
  if (mag_unlikely(root >= rdv->size))
    return mag_set_error(err, MAG_ERR_COMM, "rendezvous %s: broadcast root %u is out of range for size %u.", rdv->ops->name, root, rdv->size);
  if (rdv->size == 1) return MAG_OK;
  return (*rdv->ops->broadcast)(err, rdv, buf, nb, root);
}

mag_status_t mag_comm_rdvz_all_gather(mag_error_t *err, mag_comm_rendezvous_t *rdv, const void *in, void *out, size_t nb) {
  if (rdv->size == 1) {
    if (nb && in != out) memmove(out, in, nb);
    return MAG_OK;
  }
  return (*rdv->ops->all_gather)(err, rdv, in, out, nb);
}

mag_status_t mag_comm_rdvz_barrier(mag_error_t *err, mag_comm_rendezvous_t *rdv) {
  if (rdv->size == 1) return MAG_OK;
  return (*rdv->ops->barrier)(err, rdv);
}

void mag_comm_rdvz_abort(mag_comm_rendezvous_t *rdv) {
  if (rdv && rdv->ops && rdv->ops->abort)
    (*rdv->ops->abort)(rdv);
}

void mag_comm_rdvz_destroy(mag_comm_rendezvous_t *rdv) {
  if (rdv && rdv->ops && rdv->ops->shutdown)
    (*rdv->ops->shutdown)(rdv);
}

mag_status_t mag_comm_rdvz_topology(mag_error_t *err, mag_comm_rendezvous_t *rdv, mag_comm_rendezvous_topology_t *out) {
  *out = (mag_comm_rendezvous_topology_t){
    .local_rank=0,
    .local_size=1,
    .node_rank=0,
    .num_nodes=1
  };
  uint32_t n = rdv->size;
  uint8_t mine[8];
  uint8_t *mp = mine;
  mag_comm_wire_transfer_wu64(&mp, mag_os_boot_uid());
  uint8_t *wire = (*mag_try_alloc)(NULL, (size_t)n<<3, 0);
  uint64_t *ids = (*mag_try_alloc)(NULL, (size_t)n*sizeof(*ids), 0);
  mag_status_t stat = MAG_OK;
  if (mag_unlikely(!wire || !ids)) {
    stat = mag_set_error(err, MAG_ERR_OOM, "rendezvous: failed to allocate host table.");
    goto cleanup;
  }
  stat = mag_comm_rdvz_all_gather(err, rdv, mine, wire, sizeof(mine));
  if (mag_iserr(stat)) goto cleanup;
  const uint8_t *rp = wire;
  for (uint32_t i=0; i < n; ++i)
    ids[i] = mag_comm_wire_transfer_ru64(&rp);
  uint64_t me = ids[rdv->rank];
  uint32_t mf = rdv->rank;
  for (uint32_t i=0; i < rdv->rank; ++i) {
    if (ids[i] == me) {
      mf = i;
      break;
    }
  }
  uint32_t local_rank=0;
  uint32_t local_size=0;
  uint32_t node_rank=0;
  uint32_t num_nodes=0;
  for (uint32_t i=0; i < n; ++i) {
    bool first = true;
    for (uint32_t j=0; j < i && first; ++j)
      if (ids[j] == ids[i])
        first = false;
    if (first) {
      if (i < mf) ++node_rank;
      ++num_nodes;
    }
    if (ids[i] == me) {
      if (i < rdv->rank) ++local_rank;
      ++local_size;
    }
  }
  *out = (mag_comm_rendezvous_topology_t){
    .local_rank=local_rank,
    .local_size=local_size,
    .node_rank=node_rank,
    .num_nodes=num_nodes
  };
cleanup:
  if (wire) (*mag_alloc)(wire, 0, 0);
  if (ids) (*mag_alloc)(ids, 0, 0);
  return stat;
}

mag_status_t mag_comm_rdvz_split_local(mag_error_t *err, mag_comm_rendezvous_t *rdv, mag_comm_rendezvous_t **out_child) {
  *out_child = NULL;
  mag_comm_rendezvous_topology_t topo;
  mag_status_t stat = mag_comm_rdvz_topology(err, rdv, &topo);
  if (mag_iserr(stat)) return stat;
  return mag_comm_rdvz_split(err, rdv, out_child, topo.node_rank, rdv->rank);
}
