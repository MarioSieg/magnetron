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

#include "mag_mpmc_queue.h"
#include "mag_alloc.h"

mag_static_assert(!(sizeof(mag_mpmc_queue_t) % MAG_DESTRUCTIVE_INTERFERENCE_SIZE));
mag_static_assert(offsetof(mag_mpmc_queue_t, tail) - offsetof(mag_mpmc_queue_t, head) == MAG_DESTRUCTIVE_INTERFERENCE_SIZE);

static MAG_AINLINE volatile mag_atomic64_t *mag_mpmc_queue_slot_turn(const mag_mpmc_queue_t *q, size_t i) {
  return (volatile mag_atomic64_t *)(q->slots + (i%q->cap)*q->slot_stride);
}
static MAG_AINLINE void *mag_mpmc_queue_slot_storage(const mag_mpmc_queue_t *q, size_t i) {
  return q->slots + (i % q->cap)*q->slot_stride + q->storage_off;
}
static MAG_AINLINE mag_atomic64_t mag_mpmc_queue_turn(const mag_mpmc_queue_t *q, size_t i) {
  return (mag_atomic64_t)(i / q->cap);
}

bool mag_mpmc_queue_init(mag_mpmc_queue_t *q, size_t capacity, size_t elem_size, size_t elem_align) {
  mag_assert2(q);
  mag_assert2(capacity);
  mag_assert2(elem_size);
  elem_align = mag_vmax(elem_align, 1);
  mag_assert2(!(elem_align&(elem_align-1)));
  bool aligned_size = !(elem_size&(elem_align-1));
  mag_assert2(aligned_size);
  memset(q, 0, sizeof(*q));
  size_t slot_align = mag_vmax(MAG_DESTRUCTIVE_INTERFERENCE_SIZE, elem_align);
  size_t storage_off = mag_align_up(sizeof(mag_atomic64_t), elem_align);
  size_t slot_stride = mag_align_up(storage_off + elem_size, slot_align);
  if (capacity > (SIZE_MAX/slot_stride)-1) return false;
  size_t bytes = (capacity+1)*slot_stride;
  void *base = (*mag_try_alloc)(NULL, bytes, slot_align);
  if (mag_unlikely(!base)) return false;
  memset(base, 0, bytes);
  q->cap = capacity;
  q->elem_size = elem_size;
  q->elem_align = elem_align;
  q->slot_stride = slot_stride;
  q->slot_align = slot_align;
  q->storage_off = storage_off;
  q->slots = base;
  mag_atomic64_store(&q->head, 0, MAG_MO_RELAXED);
  mag_atomic64_store(&q->tail, 0, MAG_MO_RELAXED);
  return true;
}

void mag_mpmc_queue_destroy(mag_mpmc_queue_t *q) {
  mag_assert2(q);
  if (q->slots) (*mag_alloc)(q->slots, 0, q->slot_align);
  memset(q, 0, sizeof(*q));
}

void mag_mpmc_queue_push(mag_mpmc_queue_t *q, const void *el) {
  size_t head = mag_atomic64_fetch_add(&q->head, 1, MAG_MO_SEQ_CST);
  volatile mag_atomic64_t *turn = mag_mpmc_queue_slot_turn(q, head);
  mag_atomic64_t want = mag_mpmc_queue_turn(q, head)*2;
  while (want != mag_atomic64_load(turn, MAG_MO_ACQUIRE))
    mag_cpu_pause();
  memcpy(mag_mpmc_queue_slot_storage(q, head), el, q->elem_size);
  mag_atomic64_store(turn, want+1, MAG_MO_RELEASE);
}

bool mag_mpmc_queue_try_push(mag_mpmc_queue_t *q, const void *el) {
  mag_atomic64_t head = mag_atomic64_load(&q->head, MAG_MO_ACQUIRE);
  for (;;) {
    volatile mag_atomic64_t *turn = mag_mpmc_queue_slot_turn(q, head);
    mag_atomic64_t want = mag_mpmc_queue_turn(q, head)<<1;
    if (want == mag_atomic64_load(turn, MAG_MO_ACQUIRE)) {
      mag_atomic64_t desired = head+1;
      if (mag_atomic64_compare_exchange_strong(&q->head, &head, &desired, MAG_MO_SEQ_CST, MAG_MO_SEQ_CST)) {
        memcpy(mag_mpmc_queue_slot_storage(q, (size_t)head), el, q->elem_size);
        mag_atomic64_store(turn, want+1, MAG_MO_RELEASE);
        return true;
      }
    } else {
      mag_atomic64_t prev = head;
      head = mag_atomic64_load(&q->head, MAG_MO_ACQUIRE);
      if (head == prev) return false;
    }
  }
}

void mag_mpmc_queue_pop(mag_mpmc_queue_t *q, void *out) {
  size_t tail = mag_atomic64_fetch_add(&q->tail, 1, MAG_MO_SEQ_CST);
  volatile mag_atomic64_t *turn = mag_mpmc_queue_slot_turn(q, tail);
  mag_atomic64_t expected = (mag_mpmc_queue_turn(q, tail)<<1)+1;
  while (expected != mag_atomic64_load(turn, MAG_MO_ACQUIRE))
    mag_cpu_pause();
  memcpy(out, mag_mpmc_queue_slot_storage(q, tail), q->elem_size);
  mag_atomic64_store(turn, expected+1, MAG_MO_RELEASE);
}

bool mag_mpmc_queue_try_pop(mag_mpmc_queue_t *q, void *out) {
  mag_atomic64_t tail = mag_atomic64_load(&q->tail, MAG_MO_ACQUIRE);
  for (;;) {
    volatile mag_atomic64_t *turn = mag_mpmc_queue_slot_turn(q, tail);
    mag_atomic64_t expected = (mag_mpmc_queue_turn(q, tail)<<1)+1;
    if (expected == mag_atomic64_load(turn, MAG_MO_ACQUIRE)) {
      mag_atomic64_t desired = tail+1;
      if (mag_atomic64_compare_exchange_strong(&q->tail, &tail, &desired, MAG_MO_SEQ_CST, MAG_MO_SEQ_CST)) {
        memcpy(out, mag_mpmc_queue_slot_storage(q, (size_t)tail), q->elem_size);
        mag_atomic64_store(turn, expected+1, MAG_MO_RELEASE);
        return true;
      }
    } else {
      mag_atomic64_t prev = tail;
      tail = mag_atomic64_load(&q->tail, MAG_MO_ACQUIRE);
      if (tail == prev) return false;
    }
  }
}

int64_t mag_mpmc_queue_size(mag_mpmc_queue_t *q) {
  return mag_atomic64_load(&q->head, MAG_MO_RELAXED) - mag_atomic64_load(&q->tail, MAG_MO_RELAXED);
}

bool mag_mpmc_queue_empty(mag_mpmc_queue_t *q) {
  return mag_mpmc_queue_size(q) <= 0;
}

size_t mag_mpmc_queue_capacity(const mag_mpmc_queue_t *q) {
  return q->cap;
}
