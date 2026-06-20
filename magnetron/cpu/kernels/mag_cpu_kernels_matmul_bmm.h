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

static int64_t mag_bmm_batch_total(int64_t br, const mag_coords_t *co) {
  int64_t bt=1;
  for (int64_t dim=0; dim < br; ++dim) bt*=co->shape[dim];
  return bt;
}

static MAG_HOTPROC void mag_matmul_bmm_dot(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t xr = x->coords.rank;
  int64_t yr = y->coords.rank;
  int64_t rr = r->coords.rank;
  int64_t K = x->coords.shape[xr-1];
  int64_t sx = x->coords.strides[xr-1];
  int64_t sy = y->coords.strides[yr-2];
  int64_t bx = xr-2;
  int64_t by = yr-2;
  int64_t br = rr-2;
  int64_t batch_tot = mag_bmm_batch_total(br, &r->coords);
  int64_t ti = payload->thread_idx;
  int64_t tc = payload->thread_num;
  int64_t chunk = (batch_tot+tc-1)/tc;
  int64_t start = ti*chunk;
  int64_t end = mag_xmin(batch_tot, start+chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t el = (int64_t)mag_type_trait(r->dtype)->size;
  uint8_t *pr = (uint8_t *)mag_tensor_data_ptr_mut(r);
  const uint8_t *px = (const uint8_t *)mag_tensor_data_ptr(x);
  const uint8_t *py = (const uint8_t *)mag_tensor_data_ptr(y);
  for (int64_t batch=start; batch < end; ++batch) {
    int64_t idx[MAG_MAX_DIMS];
    mag_bmm_compute_result_idx(br, batch, &idx, &r->coords);
    int64_t mox = mag_bmm_flattened_batch_offset(br, bx, &idx, &x->coords);
    int64_t moy = mag_bmm_flattened_batch_offset(br, by, &idx, &y->coords);
    void *ppr = pr + batch*el;
    const void *ppx = px + mox*el;
    const void *ppy = py + moy*el;
    mag_matmul_dot_impl(r->dtype, 0, K, ppr, ppx, sx, ppy, sy);
  }
}

static MAG_HOTPROC void mag_matmul_bmm_vec_mat(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t xr = x->coords.rank;
  int64_t yr = y->coords.rank;
  int64_t rr = r->coords.rank;
  int64_t K = x->coords.shape[xr-1];
  int64_t N = y->coords.shape[yr-1];
  int64_t sx = x->coords.strides[xr-1];
  int64_t sy0 = y->coords.strides[yr-2];
  int64_t sy1 = y->coords.strides[yr-1];
  int64_t bx = xr-2;
  int64_t by = yr-2;
  int64_t br = rr-2;
  int64_t batch_tot = mag_bmm_batch_total(br, &r->coords);
  int64_t ti = payload->thread_idx;
  int64_t tc = payload->thread_num;
  int64_t cols_tot = batch_tot*N;
  int64_t chunk = (cols_tot+tc-1)/tc;
  int64_t start = ti*chunk;
  int64_t end = mag_xmin(cols_tot, start+chunk);
  if (mag_unlikely(start >= end)) return;
  int64_t el = (int64_t)mag_type_trait(r->dtype)->size;
  uint8_t *pr = (uint8_t *)mag_tensor_data_ptr_mut(r);
  const uint8_t *px = (const uint8_t *)mag_tensor_data_ptr(x);
  const uint8_t *py = (const uint8_t *)mag_tensor_data_ptr(y);
  for (int64_t i=start; i < end;) {
    int64_t batch = i/N;
    int64_t n0 = i%N;
    int64_t Nt = mag_xmin(N-n0, end-i);
    int64_t idx[MAG_MAX_DIMS];
    mag_bmm_compute_result_idx(br, batch, &idx, &r->coords);
    int64_t mox = mag_bmm_flattened_batch_offset(br, bx, &idx, &x->coords);
    int64_t moy = mag_bmm_flattened_batch_offset(br, by, &idx, &y->coords);
    void *ppr = pr + (batch*N + n0)*el;
    const void *ppx = px + mox*el;
    const void *ppy = py + (moy + n0*sy1)*el;
    mag_matmul_gemv_vec_mat_impl(r->dtype, 0, 1, Nt, K, ppr, ppx, sx, ppy, sy0, sy1);
    i += Nt;
  }
}

static MAG_HOTPROC void mag_matmul_bmm_mat_vec(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  int64_t xr = x->coords.rank;
  int64_t yr = y->coords.rank;
  int64_t rr = r->coords.rank;
  int64_t M = x->coords.shape[xr-2];
  int64_t K = x->coords.shape[xr-1];
  int64_t sx0 = x->coords.strides[xr-2];
  int64_t sx1 = x->coords.strides[xr-1];
  int64_t sy  = y->coords.strides[yr-1];
  int64_t bx = xr-2;
  int64_t by = yr-2;
  int64_t br = rr-2;
  int64_t batch_tot = mag_bmm_batch_total(br, &r->coords);
  int64_t ti = payload->thread_idx;
  int64_t tc = payload->thread_num;
  int64_t rows_tot = batch_tot*M;
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
    int64_t m0 = i%M;
    int64_t Mt = mag_xmin(M-m0, end-i);
    int64_t idx[MAG_MAX_DIMS];
    mag_bmm_compute_result_idx(br, batch, &idx, &r->coords);
    int64_t mox = mag_bmm_flattened_batch_offset(br, bx, &idx, &x->coords);
    int64_t moy = mag_bmm_flattened_batch_offset(br, by, &idx, &y->coords);
    void *ppr = pr + (batch*M + m0)*el;
    const void *ppx = px + (mox + m0*sx0)*el;
    const void *ppy = py + moy*el;
    mag_matmul_gemv_mat_vec_impl(r->dtype, 0,1, Mt, K, ppr, ppx, sx0, sx1, ppy, sy);
    i += Mt;
  }
}

static MAG_HOTPROC void mag_matmul_bmm_gemm(const mag_kernel_payload_t *payload) {
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
    mag_matmul_gemm_impl(r->dtype, 0, 1, Mt, N, K, ppr, ppx, sx0, sx1, ppy, sy0, sy1);
    i += Mt;
  }
}

static MAG_HOTPROC void mag_matmul_bmm(const mag_kernel_payload_t *payload, mag_matmul_type_t type) {
  switch (type) {
    case MAG_MATMUL_TYPE_BMM_DOT: mag_matmul_bmm_dot(payload); return;
    case MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT: mag_matmul_bmm_vec_mat(payload); return;
    case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC: mag_matmul_bmm_mat_vec(payload); return;
    case MAG_MATMUL_TYPE_BMM_GEMM: mag_matmul_bmm_gemm(payload); return;
    default: mag_panic("matmul: invalid BMM matmul type '%s'.", mag_matmul_type_name(type)); return;
  }
}
