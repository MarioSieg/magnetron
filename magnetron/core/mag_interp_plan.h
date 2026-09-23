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

#ifndef MAG_INTERP_PLAN_H
#define MAG_INTERP_PLAN_H

#include "mag_def.h"
#include "mag_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_interp_axis_t {
  int64_t in;
  int64_t out;
  int64_t max_taps;
  int64_t *ntaps;
  int64_t *idx;
  float *w;
} mag_interp_axis_t;

typedef struct mag_interp_plan_t {
  int64_t planes;
  mag_interp_axis_t axes[3];
} mag_interp_plan_t;

typedef void *(mag_interp_alloc_fn)(void *ud, size_t nb);

extern MAG_EXPORT bool mag_interp_mode_is_nearest(mag_interp_mode_t mode);
extern MAG_EXPORT bool mag_interp_plan_build(
  mag_interp_plan_t *plan,
  const mag_op_params_t *params,
  const int64_t *small_shape,
  const int64_t *big_shape,
  int64_t rank,
  bool transposed,
  mag_interp_alloc_fn *alloc,
  void *ud
);

#ifdef __cplusplus
}
#endif

#endif
