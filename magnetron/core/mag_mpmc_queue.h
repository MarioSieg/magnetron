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

#ifndef MAG_MPMC_QUEUE_H
#define MAG_MPMC_QUEUE_H

#include "mag_def.h"
#include "mag_threadlib.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_mpmc_queue_t {
  size_t cap;
  size_t elem_size;
  size_t elem_align;
  size_t slot_stride;
  size_t slot_align;
  size_t storage_off;
  uint8_t *slots;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile mag_atomic64_t head;
  mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile mag_atomic64_t tail;
} mag_mpmc_queue_t;

extern MAG_EXPORT bool mag_mpmc_queue_init(mag_mpmc_queue_t *q, size_t capacity, size_t elem_size, size_t elem_align);
extern MAG_EXPORT void mag_mpmc_queue_destroy(mag_mpmc_queue_t *q);
extern MAG_EXPORT void mag_mpmc_queue_push(mag_mpmc_queue_t *q, const void *el);
extern MAG_EXPORT bool mag_mpmc_queue_try_push(mag_mpmc_queue_t *q, const void *el);
extern MAG_EXPORT void mag_mpmc_queue_pop(mag_mpmc_queue_t *q, void *out);
extern MAG_EXPORT bool mag_mpmc_queue_try_pop(mag_mpmc_queue_t *q, void *out);
extern MAG_EXPORT int64_t mag_mpmc_queue_size(mag_mpmc_queue_t *q);
extern MAG_EXPORT bool mag_mpmc_queue_empty(mag_mpmc_queue_t *q);
extern MAG_EXPORT size_t mag_mpmc_queue_capacity(const mag_mpmc_queue_t *q);

#ifdef __cplusplus
}
#endif

#endif
