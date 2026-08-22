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

#ifndef MAGNETRON_CPU_VECTORIZE_PLAN_H
#define MAGNETRON_CPU_VECTORIZE_PLAN_H

#include <core/mag_def.h>
#include <core/mag_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_unary_vectorization_plan_t {
  int64_t inner;
  int64_t outer_rank;
  int64_t shape[MAG_MAX_DIMS];
  int64_t rstr[MAG_MAX_DIMS];
  int64_t xstr[MAG_MAX_DIMS];
} mag_unary_vectorization_plan_t;
extern bool mag_unary_vectorization_plan_init(mag_unary_vectorization_plan_t *p, const mag_tensor_t *r, const mag_tensor_t *x);
extern void mag_unary_vectorization_plan_step(const mag_unary_vectorization_plan_t *p, int64_t o, int64_t *rb, int64_t *xb);

typedef struct mag_binary_vectorization_plan_t {
  int64_t inner;
  int64_t outer_rank;
  int64_t shape[MAG_MAX_DIMS];
  int64_t xstr[MAG_MAX_DIMS];
  int64_t ystr[MAG_MAX_DIMS];
  bool x_const;
  bool y_const;
} mag_binary_vectorization_plan_t;

extern bool mag_binary_vectorization_plan_init(mag_binary_vectorization_plan_t *p, const mag_tensor_t *r, const mag_tensor_t *x, const mag_tensor_t *y);
extern void mag_binary_vectorization_plan_step(const mag_binary_vectorization_plan_t *p, int64_t o, int64_t *xb, int64_t *yb);

#ifdef __cplusplus
}
#endif
#endif
