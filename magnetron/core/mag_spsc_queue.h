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

#ifndef MAG_SPSC_QUEUE_H
#define MAG_SPSC_QUEUE_H

#include "mag_def.h"
#include "mag_threadlib.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_spsc_queue_t {
  size_t cap;
  size_t el_sz;
  size_t el_al;
  uint8_t *slots;
  void *base;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile mag_atomic64_t write_idx;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_atomic64_t read_idx_cache;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile mag_atomic64_t read_idx;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) mag_atomic64_t write_idx_cache;
} mag_spsc_queue_t;

extern MAG_EXPORT bool mag_spsc_queue_init(mag_spsc_queue_t *q, size_t capacity, size_t elem_size, size_t elem_align);
extern MAG_EXPORT void mag_spsc_queue_destroy(mag_spsc_queue_t *q);
extern MAG_EXPORT void *mag_spsc_queue_reserve(mag_spsc_queue_t *q);
extern MAG_EXPORT void *mag_spsc_queue_try_reserve(mag_spsc_queue_t *q);
extern MAG_EXPORT void mag_spsc_queue_commit(mag_spsc_queue_t *q);
extern MAG_EXPORT void mag_spsc_queue_push(mag_spsc_queue_t *q, const void *el);
extern MAG_EXPORT bool mag_spsc_queue_try_push(mag_spsc_queue_t *q, const void *el);
extern MAG_EXPORT void *mag_spsc_queue_front(mag_spsc_queue_t *q);
extern MAG_EXPORT void mag_spsc_queue_pop(mag_spsc_queue_t *q);
extern MAG_EXPORT bool mag_spsc_queue_try_pop(mag_spsc_queue_t *q, void *out);
extern MAG_EXPORT size_t mag_spsc_queue_size(mag_spsc_queue_t *q);
extern MAG_EXPORT bool mag_spsc_queue_empty(mag_spsc_queue_t *q);
extern MAG_EXPORT size_t mag_spsc_queue_capacity(const mag_spsc_queue_t *q);

#ifdef __cplusplus
}
#endif

#endif
