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

typedef enum mag_vectorize_flags_t {
  MAG_VAX_CX = 1<<0, /* const x */
  MAG_VAX_CY = 1<<1, /* const y */
  MAG_VAX_FX = 1<<2, /* flat x */
  MAG_VAX_FY = 1<<3  /* flat y */
} mag_vectorize_flags_t;

typedef struct mag_unary_vectorization_plan_t {
  int64_t inner;
  int64_t outer_rank;
  int64_t shape[MAG_MAX_DIMS];
  int64_t rstr[MAG_MAX_DIMS];
  int64_t xstr[MAG_MAX_DIMS];
  mag_vectorize_flags_t flags;
} mag_unary_vectorization_plan_t;
extern bool mag_unary_vectorization_plan_init(mag_unary_vectorization_plan_t *p, const mag_tensor_t *r, const mag_tensor_t *x);
extern void mag_unary_vectorization_plan_step(const mag_unary_vectorization_plan_t *p, int64_t o, int64_t *rb, int64_t *xb);
typedef struct mag_binary_vectorization_plan_t {
  int64_t inner;
  int64_t outer_rank;
  int64_t shape[MAG_MAX_DIMS];
  int64_t rstr[MAG_MAX_DIMS];
  int64_t xstr[MAG_MAX_DIMS];
  int64_t ystr[MAG_MAX_DIMS];
  mag_vectorize_flags_t flags;
} mag_binary_vectorization_plan_t;

extern bool mag_binary_vectorization_plan_init(mag_binary_vectorization_plan_t *p, const mag_tensor_t *r, const mag_tensor_t *x, const mag_tensor_t *y);
extern void mag_binary_vectorization_plan_step(const mag_binary_vectorization_plan_t *p, int64_t o, int64_t *rb, int64_t *xb, int64_t *yb);


typedef struct mag_tile_plan_t {
  int64_t rank;
  int64_t a;
  int64_t b;
  int64_t tile;
  int64_t ntiles;
  int64_t grid[MAG_MAX_DIMS];
  int64_t cstr[MAG_MAX_DIMS];
} mag_tile_plan_t;
extern bool mag_tile_plan_init(mag_tile_plan_t *p, const mag_tensor_t *r, const int64_t *rt, const int64_t *xt, const int64_t *yt, int64_t tile);
extern void mag_tile_plan_origin(const mag_tile_plan_t *p, int64_t i, int64_t *o);

#ifdef __cplusplus
}
#endif
#endif
