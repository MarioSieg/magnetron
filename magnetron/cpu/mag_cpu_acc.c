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

#include "mag_cpu_acc.h"

#ifdef MAG_HAS_ACCELERATE

#include <stdlib.h>
#include <core/mag_coords.h>
#include "mag_cpu_tls_arena.h"

extern MAG_THREAD_LOCAL mag_scratch_arena_t mag_tls_arena;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <Accelerate/Accelerate.h>

#define MAG_ACCEL_BF16_MIN_M_CONTIG 16
#define MAG_ACCEL_THIN_MAX_M 32
#define MAG_ACCEL_HALF_MIN_M_TRANSPOSED 3
#define MAG_ACCEL_F32_MIN_M_TRANSPOSED 4

static bool mag_accel_bnns_available(void) {
  if (__builtin_available(macOS 13.0, *)) return true;
  return false;
}

typedef struct mag_accel_gemm_t {
  int64_t batch;
  int64_t M, K, N;
  int64_t sx0, sx1;
  int64_t sy0, sy1;
  int64_t br, bx, by;
  const mag_coords_t *cr, *cx, *cy;
} mag_accel_gemm_t;

static bool mag_accel_disabled(void) {
  static int state = -1;
  if (state < 0) {
    const char *env = getenv("MAGNETRON_DISABLE_ACCELERATE");
    state = env && *env && *env != '0';
  }
  return state;
}

static bool mag_accel_operand_layout_ok(int64_t rows, int64_t cols, int64_t s0, int64_t s1) {
  if (cols == 1 && rows == 1) return true;
  if (s1 == 1) return cols == 1 || s0 >= cols || rows == 1;
  if (s0 == 1) return rows == 1 || s1 >= rows || cols == 1;
  return false;
}

static bool mag_accel_describe(mag_accel_gemm_t *g, const mag_tensor_t *x, const mag_tensor_t *y, const mag_tensor_t *r) {
  const mag_coords_t *cx = &x->meta.coords;
  const mag_coords_t *cy = &y->meta.coords;
  const mag_coords_t *cr = &r->meta.coords;
  int64_t xr = cx->rank, yr = cy->rank, rr = cr->rank;
  memset(g, 0, sizeof(*g));
  g->cr = cr; g->cx = cx; g->cy = cy;
  g->batch = 1;
  if (xr == 1 && yr == 2) {
    g->M = 1; g->K = cx->shape[0]; g->N = cy->shape[1];
    g->sx0 = g->K; g->sx1 = cx->strides[0];
    g->sy0 = cy->strides[0]; g->sy1 = cy->strides[1];
  } else if (xr == 2 && yr == 1) {
    g->M = cx->shape[0]; g->K = cx->shape[1]; g->N = 1;
    g->sx0 = cx->strides[0]; g->sx1 = cx->strides[1];
    g->sy0 = cy->strides[0]; g->sy1 = 1;
  } else if (xr >= 2 && yr >= 2) {
    g->M = cx->shape[xr-2]; g->K = cx->shape[xr-1]; g->N = cy->shape[yr-1];
    g->sx0 = cx->strides[xr-2]; g->sx1 = cx->strides[xr-1];
    g->sy0 = cy->strides[yr-2]; g->sy1 = cy->strides[yr-1];
    g->br = rr-2; g->bx = xr-2; g->by = yr-2;
    for (int64_t d=0; d < g->br; ++d) g->batch *= cr->shape[d];
  } else {
    return false;
  }
  if (g->M < 1 || g->K < 1 || g->N < 1) return false;
  if (g->M == 1) g->sx0 = g->K;
  if (g->N == 1) g->sy1 = 1;
  if (g->K == 1) { g->sx1 = 1; g->sy0 = g->N; }
  if (!mag_accel_operand_layout_ok(g->M, g->K, g->sx0, g->sx1)) return false;
  if (!mag_accel_operand_layout_ok(g->K, g->N, g->sy0, g->sy1)) return false;
  return true;
}

static void mag_accel_batch_offsets(const mag_accel_gemm_t *g, int64_t b, int64_t *ox, int64_t *oy) {
  int64_t idx[MAG_MAX_DIMS] = {0};
  for (int64_t d=g->br-1, t=b; d >= 0; --d) {
    idx[d] = t%g->cr->shape[d];
    t /= g->cr->shape[d];
  }
  int64_t mx=0, my=0;
  for (int64_t d=0; d < g->bx; ++d) mx += (g->cx->shape[d] == 1 ? 0 : idx[g->br-g->bx+d])*g->cx->strides[d];
  for (int64_t d=0; d < g->by; ++d) my += (g->cy->shape[d] == 1 ? 0 : idx[g->br-g->by+d])*g->cy->strides[d];
  *ox = mx;
  *oy = my;
}

bool mag_accel_matmul_supported(const mag_tensor_t *x, const mag_tensor_t *y, const mag_tensor_t *r) {
  if (mag_accel_disabled()) return false;
  mag_dtype_t dt = r->meta.dtype;
  if (x->meta.dtype != dt || y->meta.dtype != dt) return false;
  if (!mag_tensor_is_contiguous(r)) return false;
  mag_accel_gemm_t g;
  if (!mag_accel_describe(&g, x, y, r)) return false;
  bool y_transposed = g.sy0 == 1 && g.sy1 != 1 && g.N > 1;
  bool thin = g.M <= MAG_ACCEL_THIN_MAX_M && g.sx1 == 1 && (g.sy1 == g.K ? g.sy0 == 1 : (g.sy0 == g.N && g.sy1 == 1));
  if (thin) return false;
  switch (dt) {
    case MAG_DTYPE_FLOAT32: return !y_transposed || g.M == 1 || g.M >= MAG_ACCEL_F32_MIN_M_TRANSPOSED;
    case MAG_DTYPE_BFLOAT16:
      if (!mag_accel_bnns_available()) return false;
      if (y_transposed) return g.M >= MAG_ACCEL_HALF_MIN_M_TRANSPOSED;
      return g.M == 1 || g.M >= MAG_ACCEL_BF16_MIN_M_CONTIG;
    case MAG_DTYPE_FLOAT16:
      if (!mag_accel_bnns_available()) return false;
      if (y_transposed) return g.M >= MAG_ACCEL_HALF_MIN_M_TRANSPOSED;
      return true;
    default: return false;
  }
}

static void mag_accel_sgemm(const mag_accel_gemm_t *g, float *r, const float *x, const float *y) {
  bool tx = g->sx1 != 1;
  bool ty = g->sy1 != 1;
  int lda = (int)(tx ? g->sx1 : g->sx0);
  int ldb = (int)(ty ? g->sy1 : g->sy0);
  if (!tx && lda < g->K) lda = (int)g->K;
  if (tx && lda < g->M) lda = (int)g->M;
  if (!ty && ldb < g->N) ldb = (int)g->N;
  if (ty && ldb < g->K) ldb = (int)g->K;
  cblas_sgemm(CblasRowMajor, tx ? CblasTrans : CblasNoTrans, ty ? CblasTrans : CblasNoTrans, (int)g->M, (int)g->N, (int)g->K, 1.f, x, lda, y, ldb, 0.f, r, (int)g->N);
}

static void mag_accel_desc(BNNSNDArrayDescriptor *d, BNNSDataType dt, int64_t rows, int64_t cols, int64_t srow, int64_t scol, const void *p) {
  memset(d, 0, sizeof(*d));
  d->layout = BNNSDataLayoutRowMajorMatrix;
  d->size[0] = (size_t)cols;
  d->size[1] = (size_t)rows;
  d->stride[0] = (size_t)scol;
  d->stride[1] = (size_t)srow;
  d->data_type = dt;
  d->data = (void *)p;
}

API_AVAILABLE(macos(13.0))
static bool mag_accel_bnns_gemm(const mag_accel_gemm_t *g, BNNSDataType dt, void *r, const void *x, const void *y, void *workspace) {
  BNNSNDArrayDescriptor A, B, C;
  mag_accel_desc(&A, dt, g->M, g->K, g->sx0, g->sx1, x);
  mag_accel_desc(&B, dt, g->K, g->N, g->sy0, g->sy1, y);
  mag_accel_desc(&C, dt, g->M, g->N, g->N, 1, r);
  return BNNSMatMul(false, false, 1.f, &A, &B, &C, workspace, NULL) == 0;
}

bool mag_accel_matmul(mag_tensor_t *r, const mag_tensor_t *x, const mag_tensor_t *y) {
  mag_accel_gemm_t g;
  if (!mag_accel_describe(&g, x, y, r)) return false;
  mag_dtype_t dt = r->meta.dtype;
  int64_t el = (int64_t)mag_type_trait(dt)->size;
  uint8_t *pr = (uint8_t *)mag_tensor_data_ptr_mut(r);
  const uint8_t *px = (const uint8_t *)mag_tensor_data_ptr(x);
  const uint8_t *py = (const uint8_t *)mag_tensor_data_ptr(y);
  if (dt == MAG_DTYPE_FLOAT32) {
    for (int64_t b=0; b < g.batch; ++b) {
      int64_t ox=0, oy=0;
      if (g.batch > 1) mag_accel_batch_offsets(&g, b, &ox, &oy);
      mag_accel_sgemm(&g, (float *)(pr + b*g.M*g.N*el), (const float *)(px + ox*el), (const float *)(py + oy*el));
    }
    return true;
  }
  if (__builtin_available(macOS 13.0, *)) {
    BNNSDataType bdt = dt == MAG_DTYPE_BFLOAT16 ? BNNSDataTypeBFloat16 : BNNSDataTypeFloat16;
    BNNSNDArrayDescriptor A, B, C;
    mag_accel_desc(&A, bdt, g.M, g.K, g.sx0, g.sx1, px);
    mag_accel_desc(&B, bdt, g.K, g.N, g.sy0, g.sy1, py);
    mag_accel_desc(&C, bdt, g.M, g.N, g.N, 1, pr);
    ssize_t ws = BNNSMatMulWorkspaceSize(false, false, 1.f, &A, &B, &C, NULL);
    if (ws < 0) return false;
    size_t mark = mag_scratch_arena_mark(&mag_tls_arena);
    void *workspace = ws > 0 ? mag_scratch_arena_alloc(&mag_tls_arena, (size_t)ws) : NULL;
    bool ok = true;
    for (int64_t b=0; b < g.batch && ok; ++b) {
      int64_t ox=0, oy=0;
      if (g.batch > 1) mag_accel_batch_offsets(&g, b, &ox, &oy);
      ok = mag_accel_bnns_gemm(&g, bdt, pr + b*g.M*g.N*el, px + ox*el, py + oy*el, workspace);
    }
    mag_scratch_arena_reset(&mag_tls_arena, mark);
    return ok;
  }
  return false;
}

bool mag_accel_sgemm_available(void) {
  return !mag_accel_disabled();
}

void mag_accel_sgemm_ex(bool ta, bool tb, int64_t M, int64_t N, int64_t K, float alpha, const float *a, int64_t lda, const float *b, int64_t ldb, float beta, float *c, int64_t ldc) {
  cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, (int)M, (int)N, (int)K, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
}

#pragma clang diagnostic pop

#else

bool mag_accel_matmul_supported(const mag_tensor_t *x, const mag_tensor_t *y, const mag_tensor_t *r) {
  (void)x; (void)y; (void)r;
  return false;
}

bool mag_accel_matmul(mag_tensor_t *r, const mag_tensor_t *x, const mag_tensor_t *y) {
  (void)r; (void)x; (void)y;
  return false;
}

bool mag_accel_sgemm_available(void) {
  return false;
}

void mag_accel_sgemm_ex(bool ta, bool tb, int64_t M, int64_t N, int64_t K, float alpha, const float *a, int64_t lda, const float *b, int64_t ldb, float beta, float *c, int64_t ldc) {
  (void)ta; (void)tb; (void)M; (void)N; (void)K; (void)alpha; (void)a; (void)lda; (void)b; (void)ldb; (void)beta; (void)c; (void)ldc;
}

#endif
