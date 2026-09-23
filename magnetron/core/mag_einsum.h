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

#ifndef MAG_EINSUM_H
#define MAG_EINSUM_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_EINSUM_MAX_INPUTS 64

typedef struct mag_einsum_path_t {
  size_t naive_cost;
  size_t naive_scaling;
  size_t opt_cost;
  size_t opt_scaling;
  size_t max_term;
  size_t num_steps;
  uint32_t steps[MAG_EINSUM_MAX_INPUTS][2];
} mag_einsum_path_t;

extern MAG_EXPORT mag_status_t mag_einsum_eval(mag_error_t *err, mag_tensor_t **out_result, const char *equation, const mag_tensor_t **args, size_t num_args);
extern MAG_EXPORT mag_status_t mag_einsum_path(mag_error_t *err, mag_einsum_path_t *out_path, const char *equation, const mag_tensor_t **args, size_t num_args);

#ifdef __cplusplus
}
#endif

#endif
