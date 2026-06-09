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

typedef void (mag_gemm_kernel_contig_t)(int64_t M, int64_t N, int64_t K, void *r, const void *px, const void *py);
#define mag_gemm_kernel_contig_impl(dtype, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemm_kernel_contig_##dtype(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py) { \
    dtype *r = (dtype *)pr; \
    const dtype *x = (const dtype *)px; \
    const dtype *y = (const dtype *)py; \
    for (int64_t i=0; i < M; ++i) { \
        for (int64_t j=0; j < N; ++j) { \
          float acc = 0.f; \
          for (int64_t k=0; k < K; ++k) \
            acc += TtoF32(x[i*K + k])*TtoF32(y[k*N + j]); \
          r[i*N + j] = F32toT(acc); \
        } \
      } \
  }
mag_gemm_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemm_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemm_kernel_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemm_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemm_kernel_contig_impl

typedef void (mag_gemm_kernel_strided_t)(int64_t M, int64_t N, int64_t K, void *r, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy0, int64_t sy1);
#define mag_gemm_kernel_strided_impl(dtype, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_gemm_kernel_strided_##dtype(int64_t M, int64_t N, int64_t K, void *pr, const void *px, const void *py, int64_t sx0, int64_t sx1, int64_t sy0, int64_t sy1) { \
    dtype *r = (dtype *)pr; \
    const dtype *x = (const dtype *)px; \
    const dtype *y = (const dtype *)py; \
    for (int64_t i=0; i < M; ++i) { \
        for (int64_t j=0; j < N; ++j) { \
          float acc = 0.f; \
          for (int64_t k=0; k < K; ++k) \
            acc += TtoF32(x[i*sx0 + k*sx1])*TtoF32(y[k*sy0 + j*sy1]); \
          r[i*N + j] = F32toT(acc); \
        } \
      } \
  }
mag_gemm_kernel_strided_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_gemm_kernel_strided_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_gemm_kernel_strided_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gemm_kernel_strided_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_gemm_kernel_strided_impl

static mag_gemm_kernel_contig_t *const mag_gemm_kernel_lut_contig[4] = {
  [MAG_DTYPE_FLOAT32] = &mag_gemm_kernel_contig_float,
  [MAG_DTYPE_FLOAT16] = &mag_gemm_kernel_contig_mag_float16_t,
  [MAG_DTYPE_BFLOAT16] = &mag_gemm_kernel_contig_mag_bfloat16_t,
  [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_kernel_contig_mag_float8_e4m3fn_t
};
static mag_gemm_kernel_strided_t *const mag_gemm_kernel_lut_strided[4] = {
  [MAG_DTYPE_FLOAT32] = &mag_gemm_kernel_strided_float,
  [MAG_DTYPE_FLOAT16] = &mag_gemm_kernel_strided_mag_float16_t,
  [MAG_DTYPE_BFLOAT16] = &mag_gemm_kernel_strided_mag_bfloat16_t,
  [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_kernel_strided_mag_float8_e4m3fn_t
};

static MAG_HOTPROC void mag_matmul_gemm(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t M = x->coords.shape[0];
  int64_t K = x->coords.shape[1];
  int64_t N = y->coords.shape[1];
  int64_t sx0 = x->coords.strides[0];
  int64_t sx1 = x->coords.strides[1];
  int64_t sy0 = y->coords.strides[0];
  int64_t sy1 = y->coords.strides[1];
  int64_t ti = payload->thread_idx;
  int64_t tc = payload->thread_num;
  int64_t chunk = (M+tc-1)/tc;
  int64_t start = ti*chunk;
  int64_t end = mag_xmin(M, start+chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t Mt = end-start;
  int64_t el = (int64_t)mag_type_trait(r->dtype)->size;
  void *pr = (uint8_t *)mag_tensor_data_ptr_mut(r) + start*N*el;
  const void *px = (const uint8_t *)mag_tensor_data_ptr(x) + start*sx0*el;
  const void *py = (const void *)mag_tensor_data_ptr(y);
  if (sx0 == K && sx1 == 1 && sy0 == N && sy1 == 1)
    (*mag_gemm_kernel_lut_contig[r->dtype])(Mt, N, K, pr, px, py);
  else
    (*mag_gemm_kernel_lut_strided[r->dtype])(Mt, N, K, pr, px, py, sx0, sx1, sy0, sy1);
}

static void mag_bmm_compute_result_idx(int64_t br, int64_t batch, int64_t (*out)[MAG_MAX_DIMS], const mag_coords_t *co) {
  memset(*out, 0, sizeof(*out));
  for (int64_t dim=br-1, tb=batch; dim >= 0; --dim) {
    int64_t ax = co->shape[dim];
    (*out)[dim] = tb%ax;
    tb /= ax;
  }
}

static int64_t mag_bmm_flattened_batch_offset(int64_t br, int64_t bb, const int64_t (*idx)[MAG_MAX_DIMS], const mag_coords_t *co) {
  int64_t moff=0;
  for (int64_t dim=0; dim < bb; ++dim)
    moff += (co->shape[dim] == 1 ? 0 : (*idx)[br-bb+dim])*co->strides[dim];
  return moff;
}

static MAG_HOTPROC void mag_matmul_bmm(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t xr = x->coords.rank;
  int64_t yr = y->coords.rank;
  int64_t rr = r->coords.rank;
  int64_t M = x->coords.shape[xr-2];
  int64_t K = x->coords.shape[xr-1];
  int64_t N = y->coords.shape[yr-1];
  int64_t sx0 = x->coords.strides[xr-2];
  int64_t sx1 = x->coords.strides[xr-1];
  int64_t sy0 = y->coords.strides[yr-2];
  int64_t sy1 = y->coords.strides[yr-1];
  bool contig = sx0 == K && sx1 == 1 && sy0 == N && sy1 == 1;
  int64_t bx = xr-2;
  int64_t by = yr-2;
  int64_t br = rr-2;
  int64_t batch_tot=1;
  for (int64_t dim=0; dim < br; ++dim) batch_tot *= r->coords.shape[dim];
  int64_t rows_tot = M*batch_tot;
  int64_t ti = payload->thread_idx;
  int64_t tc = payload->thread_num;
  int64_t chunk = (rows_tot+tc-1)/tc;
  int64_t start = ti*chunk;
  int64_t end = mag_xmin(rows_tot, start+chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t el = (int64_t)mag_type_trait(r->dtype)->size;
  uint8_t *pr = (uint8_t *)mag_tensor_data_ptr_mut(r);
  const uint8_t *px = (const uint8_t *)mag_tensor_data_ptr(x);
  const uint8_t *py = (const uint8_t *)mag_tensor_data_ptr(y);
  for (int64_t i=start; i < end;) {
    int64_t batch = i/M;
    int64_t i0 = i%M;
    int64_t Mt = mag_xmin(M-i0, end-i);
    int64_t idx[MAG_MAX_DIMS];
    mag_bmm_compute_result_idx(br, batch, &idx, &r->coords);
    int64_t mox = mag_bmm_flattened_batch_offset(br, bx, &idx, &x->coords);
    int64_t moy = mag_bmm_flattened_batch_offset(br, by, &idx, &y->coords);
    void *ppr = pr + (batch*M + i0)*N*el;
    const void *ppx = px + (mox + i0*sx0)*el;
    const void *ppy = py + moy*el;
    if (contig)
      (*mag_gemm_kernel_lut_contig[r->dtype])(Mt, N, K, ppr, ppx, ppy);
    else
      (*mag_gemm_kernel_lut_strided[r->dtype])(Mt, N, K, ppr, ppx, ppy, sx0, sx1, sy0, sy1);
    i += Mt;
  }
}
