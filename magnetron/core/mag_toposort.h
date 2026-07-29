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

#ifndef MAG_TOPOSORT_H
#define MAG_TOPOSORT_H

#include "mag_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_TOPOSORT_HASHSET_INIT_CAP 1024
#define MAG_TOPOSORT_STACK_INIT_CAP 512

typedef struct mag_topo_set_t {
  mag_tensor_t **buf;
  size_t len;
  size_t cap;
} mag_topo_set_t;

extern bool mag_topo_set_init(mag_topo_set_t *set, size_t cap);
extern void mag_topo_set_reset(mag_topo_set_t *set);
extern void mag_topo_set_free(mag_topo_set_t *set);

typedef struct mag_topo_stack_record_t mag_topo_stack_record_t;

typedef struct mag_topo_stack_t {
  mag_topo_stack_record_t *top;
  size_t len;
  size_t cap;
} mag_topo_stack_t;
extern bool mag_topo_stack_init(mag_topo_stack_t *stack, size_t cap);
extern void mag_topo_stack_reset(mag_topo_stack_t *stack);
extern void mag_topo_stack_free(mag_topo_stack_t *stack);

extern mag_status_t mag_topo_sort(
  mag_error_t *err,
  mag_tensor_t *root,
  mag_topo_stack_t *tmp_stack,
  mag_topo_set_t *out_sorted
);

#ifdef __cplusplus
}
#endif

#endif
