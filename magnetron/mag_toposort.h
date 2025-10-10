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

typedef struct mag_tensor_array_t {
    mag_tensor_t **data;
    size_t size;
    size_t capacity;
} mag_tensor_array_t;

extern void mag_tensor_array_init(mag_tensor_array_t *arr);
extern void mag_tensor_array_free(mag_tensor_array_t *arr);
extern void mag_tensor_array_push(mag_tensor_array_t *arr, mag_tensor_t *t);
extern void mag_toposort(mag_tensor_t *root, mag_tensor_array_t *sorted);

#ifdef __cplusplus
}
#endif

#endif
