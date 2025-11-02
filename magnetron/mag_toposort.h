/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
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
    mag_tensor_t **data;
    size_t size;
    size_t capacity;
} mag_topo_set_t;

extern void mag_topo_set_init(mag_topo_set_t *ts);
extern void mag_topo_set_free(mag_topo_set_t *ts);
extern void mag_topo_sort(mag_tensor_t *root, mag_topo_set_t *out_sorted);

#ifdef __cplusplus
}
#endif

#endif
