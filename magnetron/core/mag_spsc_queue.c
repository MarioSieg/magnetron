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

#include "mag_spsc_queue.h"
#include "mag_alloc.h"

mag_static_assert(sizeof(mag_spsc_queue_t) >= 4*MAG_DESTRUCTIVE_INTERFERENCE_SIZE);
mag_static_assert(offsetof(mag_spsc_queue_t, read_idx) - offsetof(mag_spsc_queue_t, write_idx) >= MAG_DESTRUCTIVE_INTERFERENCE_SIZE);

static MAG_AINLINE uint8_t *mag_spsc_queue_slot(const mag_spsc_queue_t *q, size_t idx) { return q->slots + idx*q->el_sz; }

bool mag_spsc_queue_init(mag_spsc_queue_t *q, size_t capacity, size_t elem_size, size_t elem_align) {
  mag_assert2(q);
  mag_assert2(elem_size);
  elem_align = mag_vmax(elem_align, 1);
  mag_assert2(!(elem_align&(elem_align-1)));
  bool aligned_size = !(elem_size&(elem_align-1));
  mag_assert2(aligned_size);
  memset(q, 0, sizeof(*q));
  capacity = mag_vmax(capacity, 1);
  size_t pad = mag_vmax(MAG_DESTRUCTIVE_INTERFERENCE_SIZE, elem_align);
  size_t max_cap = (SIZE_MAX - (pad<<1))/elem_size;
  if (capacity > max_cap-1) capacity = max_cap-1;
  size_t cap = capacity+1;
  size_t nb = cap*elem_size + (pad<<1);
  void *base = (*mag_try_alloc)(NULL, nb, pad);
  if (mag_unlikely(!base)) return false;
  q->cap = cap;
  q->el_sz = elem_size;
  q->el_al = elem_align;
  q->base = base;
  q->slots = (uint8_t *)base + pad;
  mag_atomic64_store(&q->write_idx, 0, MAG_MO_RELAXED);
  mag_atomic64_store(&q->read_idx, 0, MAG_MO_RELAXED);
  q->read_idx_cache = 0;
  q->write_idx_cache = 0;
  return true;
}

void mag_spsc_queue_destroy(mag_spsc_queue_t *q) {
  mag_assert2(q);
  if (q->base) {
    size_t pad = mag_vmax((size_t)MAG_DESTRUCTIVE_INTERFERENCE_SIZE, q->el_al);
    (*mag_alloc)(q->base, 0, pad);
  }
  memset(q, 0, sizeof(*q));
}

void *mag_spsc_queue_reserve(mag_spsc_queue_t *q) {
  size_t write_idx = mag_atomic64_load(&q->write_idx, MAG_MO_RELAXED);
  size_t next = write_idx+1;
  if (next == q->cap) next = 0;
  while (next == (size_t)q->read_idx_cache) {
    q->read_idx_cache = mag_atomic64_load(&q->read_idx, MAG_MO_ACQUIRE);
    if (next != (size_t)q->read_idx_cache) break;
    mag_cpu_pause();
  }
  return mag_spsc_queue_slot(q, write_idx);
}

void *mag_spsc_queue_try_reserve(mag_spsc_queue_t *q) {
  size_t write_idx = mag_atomic64_load(&q->write_idx, MAG_MO_RELAXED);
  size_t next = write_idx+1;
  if (next == q->cap) next = 0;
  if (next == (size_t)q->read_idx_cache) {
    q->read_idx_cache = mag_atomic64_load(&q->read_idx, MAG_MO_ACQUIRE);
    if (next == (size_t)q->read_idx_cache) return NULL;
  }
  return mag_spsc_queue_slot(q, write_idx);
}

void mag_spsc_queue_commit(mag_spsc_queue_t *q) {
  size_t write_idx = mag_atomic64_load(&q->write_idx, MAG_MO_RELAXED);
  size_t next = write_idx+1;
  if (next == q->cap) next = 0;
  mag_dassert2(next != (size_t)q->read_idx_cache);
  mag_atomic64_store(&q->write_idx, (mag_atomic64_t)next, MAG_MO_RELEASE);
}

void mag_spsc_queue_push(mag_spsc_queue_t *q, const void *el) {
  void *slot = mag_spsc_queue_reserve(q);
  memcpy(slot, el, q->el_sz);
  mag_spsc_queue_commit(q);
}

bool mag_spsc_queue_try_push(mag_spsc_queue_t *q, const void *el) {
  void *slot = mag_spsc_queue_try_reserve(q);
  if (mag_unlikely(!slot)) return false;
  memcpy(slot, el, q->el_sz);
  mag_spsc_queue_commit(q);
  return true;
}

void *mag_spsc_queue_front(mag_spsc_queue_t *q) {
  size_t read_idx = mag_atomic64_load(&q->read_idx, MAG_MO_RELAXED);
  if (read_idx == (size_t)q->write_idx_cache) {
    q->write_idx_cache = mag_atomic64_load(&q->write_idx, MAG_MO_ACQUIRE);
    if (read_idx == (size_t)q->write_idx_cache) return NULL;
  }
  return mag_spsc_queue_slot(q, read_idx);
}

void mag_spsc_queue_pop(mag_spsc_queue_t *q) {
  size_t read_idx = mag_atomic64_load(&q->read_idx, MAG_MO_RELAXED);
  mag_dassert2((size_t)mag_atomic64_load(&q->write_idx, MAG_MO_ACQUIRE) != read_idx);
  size_t next = read_idx+1;
  if (next == q->cap) next = 0;
  mag_atomic64_store(&q->read_idx, (mag_atomic64_t)next, MAG_MO_RELEASE);
}

bool mag_spsc_queue_try_pop(mag_spsc_queue_t *q, void *out) {
  void *slot = mag_spsc_queue_front(q);
  if (mag_unlikely(!slot)) return false;
  memcpy(out, slot, q->el_sz);
  mag_spsc_queue_pop(q);
  return true;
}

size_t mag_spsc_queue_size(mag_spsc_queue_t *q) {
  int64_t diff = mag_atomic64_load(&q->write_idx, MAG_MO_ACQUIRE) - mag_atomic64_load(&q->read_idx, MAG_MO_ACQUIRE);
  if (diff < 0) diff += (int64_t)q->cap;
  return diff;
}

bool mag_spsc_queue_empty(mag_spsc_queue_t *q) {
  return mag_atomic64_load(&q->write_idx, MAG_MO_ACQUIRE) == mag_atomic64_load(&q->read_idx, MAG_MO_ACQUIRE);
}

size_t mag_spsc_queue_capacity(const mag_spsc_queue_t *q) {
  return q->cap-1;
}
