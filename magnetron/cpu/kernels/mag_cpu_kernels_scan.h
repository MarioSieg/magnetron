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

#define mag_scan_dim_setup(x, dim_var, dim_size_var, outer_count_var, stride_x_dim_var, outer_rank_var, shape_outer_var, mult_outer_var, outer_to_full_var) \
  int64_t dim_var = payload->cmd->params->cumu.dim; \
  if (dim_var < 0) dim_var += (x)->meta.coords.rank; \
  mag_assert2(dim_var >= 0 && dim_var < (x)->meta.coords.rank); \
  const int64_t dim_size_var = (x)->meta.coords.shape[dim_var]; \
  if (dim_size_var <= 0) return MAG_OK; \
  int64_t outer_count_var = (x)->meta.numel / dim_size_var; \
  int64_t stride_x_dim_var = (x)->meta.coords.strides[dim_var]; \
  int64_t outer_rank_var = (x)->meta.coords.rank - 1; \
  int64_t shape_outer_var[MAG_MAX_DIMS]; \
  int64_t mult_outer_var[MAG_MAX_DIMS]; \
  int64_t outer_to_full_var[MAG_MAX_DIMS]; \
  { \
    int64_t t = 0; \
    for (int64_t d=0; d < (x)->meta.coords.rank; ++d) { \
      if (d == dim_var) continue; \
      shape_outer_var[t] = (x)->meta.coords.shape[d]; \
      outer_to_full_var[t] = d; \
      ++t; \
    } \
    for (int64_t t2=0; t2 < outer_rank_var; ++t2) { \
      int64_t m = 1; \
      for (int64_t k2=t2+1; k2 < outer_rank_var; ++k2) m *= shape_outer_var[k2]; \
      mult_outer_var[t2] = m; \
    } \
  }

#define mag_scan_outer_offsets(row, x, r, dim, outer_rank, mult_outer, outer_to_full, off_x0, off_r0) \
  int64_t base_idx[MAG_MAX_DIMS]; \
  for (int64_t d=0; d < (x)->meta.coords.rank; ++d) base_idx[d] = 0; \
  int64_t rtmp = row; \
  for (int64_t t=0; t < outer_rank; ++t) { \
    const int64_t q = (mult_outer[t] == 0) ? 0 : (rtmp / mult_outer[t]); \
    if (mult_outer[t] != 0) rtmp %= mult_outer[t]; \
    base_idx[outer_to_full[t]] = q; \
  } \
  base_idx[dim] = 0; \
  int64_t off_x0 = 0; \
  int64_t off_r0 = 0; \
  for (int64_t d=0; d < (x)->meta.coords.rank; ++d) { \
    off_x0 += base_idx[d]*(x)->meta.coords.strides[d]; \
    off_r0 += base_idx[d]*(r)->meta.coords.strides[d]; \
  }

#define mag_scan_outer_offsets2(row, x, v, i, dim, outer_rank, mult_outer, outer_to_full, off_x0, off_v0, off_i0) \
  int64_t base_idx[MAG_MAX_DIMS]; \
  for (int64_t d=0; d < (x)->meta.coords.rank; ++d) base_idx[d] = 0; \
  int64_t rtmp = row; \
  for (int64_t t=0; t < outer_rank; ++t) { \
    const int64_t q = (mult_outer[t] == 0) ? 0 : (rtmp / mult_outer[t]); \
    if (mult_outer[t] != 0) rtmp %= mult_outer[t]; \
    base_idx[outer_to_full[t]] = q; \
  } \
  base_idx[dim] = 0; \
  int64_t off_x0 = 0; \
  int64_t off_v0 = 0; \
  int64_t off_i0 = 0; \
  for (int64_t d=0; d < (x)->meta.coords.rank; ++d) { \
    off_x0 += base_idx[d]*(x)->meta.coords.strides[d]; \
    off_v0 += base_idx[d]*(v)->meta.coords.strides[d]; \
    off_i0 += base_idx[d]*(i)->meta.coords.strides[d]; \
  }

#define mag_gen_stub_cusum(T, TF, CVT, ACC_T, ZERO, FROM_ACC) \
  static MAG_HOTPROC mag_status_t mag_cusum_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    mag_scan_dim_setup(x, dim, dim_size, outer_count, stride_x_dim, outer_rank, shape_outer, mult_outer, outer_to_full); \
    int64_t stride_r_dim = r->meta.coords.strides[dim]; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (outer_count + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_xmin(oa + chunk, outer_count); \
    for (int64_t row=oa; row < ob; ++row) { \
      mag_scan_outer_offsets(row, x, r, dim, outer_rank, mult_outer, outer_to_full, off_x0, off_r0); \
      ACC_T acc = ZERO; \
      for (int64_t p=0; p < dim_size; ++p) { \
        int64_t off_x = off_x0 + p * stride_x_dim; \
        int64_t off_r = off_r0 + p * stride_r_dim; \
        mag_bnd_chk(bx + off_x, x->storage->base, x->storage->size); \
        acc += CVT(bx[off_x]); \
        mag_bnd_chk(br + off_r, r->storage->base, r->storage->size); \
        br[off_r] = FROM_ACC(acc); \
      } \
    } \
    return MAG_OK; \
  }

mag_gen_stub_cusum(float, float32, mag_cvt_nop, double, 0.0, mag_cvt_nop)
mag_gen_stub_cusum(mag_float16_t, float16, mag_float16_to_float32, float, 0.0f, mag_float32_to_float16)
mag_gen_stub_cusum(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, float, 0.0f, mag_float32_to_bfloat16)
mag_gen_stub_cusum(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, float, 0.0f, mag_float32_to_float8_e4m3fn)
mag_gen_stub_cusum(uint8_t, uint8, mag_cvt_nop, uint64_t, 0, mag_cvt_nop)
mag_gen_stub_cusum(int8_t, int8, mag_cvt_nop, int64_t, 0, mag_cvt_nop)
mag_gen_stub_cusum(uint16_t, uint16, mag_cvt_nop, uint64_t, 0, mag_cvt_nop)
mag_gen_stub_cusum(int16_t, int16, mag_cvt_nop, int64_t, 0, mag_cvt_nop)
mag_gen_stub_cusum(uint32_t, uint32, mag_cvt_nop, uint64_t, 0, mag_cvt_nop)
mag_gen_stub_cusum(int32_t, int32, mag_cvt_nop, int64_t, 0, mag_cvt_nop)
mag_gen_stub_cusum(uint64_t, uint64, mag_cvt_nop, uint64_t, 0, mag_cvt_nop)
mag_gen_stub_cusum(int64_t, int64, mag_cvt_nop, int64_t, 0, mag_cvt_nop)

#undef mag_gen_stub_cusum

#define mag_gen_stub_cuprod(T, TF, CVT, ACC_T, ONE, FROM_ACC) \
  static MAG_HOTPROC mag_status_t mag_cuprod_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    mag_scan_dim_setup(x, dim, dim_size, outer_count, stride_x_dim, outer_rank, shape_outer, mult_outer, outer_to_full); \
    int64_t stride_r_dim = r->meta.coords.strides[dim]; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (outer_count + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_xmin(oa + chunk, outer_count); \
    for (int64_t row=oa; row < ob; ++row) { \
      mag_scan_outer_offsets(row, x, r, dim, outer_rank, mult_outer, outer_to_full, off_x0, off_r0); \
      ACC_T acc = ONE; \
      for (int64_t p=0; p < dim_size; ++p) { \
        int64_t off_x = off_x0 + p * stride_x_dim; \
        int64_t off_r = off_r0 + p * stride_r_dim; \
        mag_bnd_chk(bx + off_x, x->storage->base, x->storage->size); \
        acc *= CVT(bx[off_x]); \
        mag_bnd_chk(br + off_r, r->storage->base, r->storage->size); \
        br[off_r] = FROM_ACC(acc); \
      } \
    } \
    return MAG_OK; \
  }

mag_gen_stub_cuprod(float, float32, mag_cvt_nop, double, 1.0, mag_cvt_nop)
mag_gen_stub_cuprod(mag_float16_t, float16, mag_float16_to_float32, float, 1.0f, mag_float32_to_float16)
mag_gen_stub_cuprod(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, float, 1.0f, mag_float32_to_bfloat16)
mag_gen_stub_cuprod(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, float, 1.0f, mag_float32_to_float8_e4m3fn)
mag_gen_stub_cuprod(uint8_t, uint8, mag_cvt_nop, uint64_t, 1, mag_cvt_nop)
mag_gen_stub_cuprod(int8_t, int8, mag_cvt_nop, int64_t, 1, mag_cvt_nop)
mag_gen_stub_cuprod(uint16_t, uint16, mag_cvt_nop, uint64_t, 1, mag_cvt_nop)
mag_gen_stub_cuprod(int16_t, int16, mag_cvt_nop, int64_t, 1, mag_cvt_nop)
mag_gen_stub_cuprod(uint32_t, uint32, mag_cvt_nop, uint64_t, 1, mag_cvt_nop)
mag_gen_stub_cuprod(int32_t, int32, mag_cvt_nop, int64_t, 1, mag_cvt_nop)
mag_gen_stub_cuprod(uint64_t, uint64, mag_cvt_nop, uint64_t, 1, mag_cvt_nop)
mag_gen_stub_cuprod(int64_t, int64, mag_cvt_nop, int64_t, 1, mag_cvt_nop)

#undef mag_gen_stub_cuprod

#define mag_gen_stub_cuext(T, TF, CVT, NAME, IS_MAX) \
  static MAG_HOTPROC mag_status_t mag_##NAME##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    mag_tensor_t *v = payload->cmd->out[0]; \
    mag_tensor_t *idx = payload->cmd->out[1]; \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    T *bv = (T *)mag_tensor_data_ptr_mut(v); \
    int64_t *bi = (int64_t *)mag_tensor_data_ptr_mut(idx); \
    mag_scan_dim_setup(x, dim, dim_size, outer_count, stride_x_dim, outer_rank, shape_outer, mult_outer, outer_to_full); \
    int64_t stride_v_dim = v->meta.coords.strides[dim]; \
    int64_t stride_i_dim = idx->meta.coords.strides[dim]; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (outer_count + tc - 1) / tc; \
    int64_t oa = ti * chunk; \
    int64_t ob = mag_xmin(oa + chunk, outer_count); \
    for (int64_t row=oa; row < ob; ++row) { \
      mag_scan_outer_offsets2(row, x, v, idx, dim, outer_rank, mult_outer, outer_to_full, off_x0, off_v0, off_i0); \
      T best = (T){0}; \
      int64_t best_idx = 0; \
      for (int64_t p=0; p < dim_size; ++p) { \
        const int64_t off_x = off_x0 + p * stride_x_dim; \
        const int64_t off_v = off_v0 + p * stride_v_dim; \
        const int64_t off_i = off_i0 + p * stride_i_dim; \
        mag_bnd_chk(bx + off_x, x->storage->base, x->storage->size); \
        T xv = bx[off_x]; \
        double xvc = (double)CVT(xv); \
        if (p == 0) { \
          best = xv; \
          best_idx = 0; \
        } else { \
          double bestc = (double)CVT(best); \
          bool better = IS_MAX ? xvc > bestc : xvc < bestc; \
          if (better) { best = xv; best_idx = p; } \
        } \
        mag_bnd_chk(bv + off_v, v->storage->base, v->storage->size); \
        mag_bnd_chk(bi + off_i, idx->storage->base, idx->storage->size); \
        bv[off_v] = best; \
        bi[off_i] = best_idx; \
      } \
    } \
    return MAG_OK; \
  }

mag_gen_stub_cuext(float, float32, mag_cvt_nop, cumax, true)
mag_gen_stub_cuext(mag_float16_t, float16, mag_float16_to_float32, cumax, true)
mag_gen_stub_cuext(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, cumax, true)
mag_gen_stub_cuext(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, cumax, true)
mag_gen_stub_cuext(uint8_t, uint8, mag_cvt_nop, cumax, true)
mag_gen_stub_cuext(int8_t, int8, mag_cvt_nop, cumax, true)
mag_gen_stub_cuext(uint16_t, uint16, mag_cvt_nop, cumax, true)
mag_gen_stub_cuext(int16_t, int16, mag_cvt_nop, cumax, true)
mag_gen_stub_cuext(uint32_t, uint32, mag_cvt_nop, cumax, true)
mag_gen_stub_cuext(int32_t, int32, mag_cvt_nop, cumax, true)
mag_gen_stub_cuext(uint64_t, uint64, mag_cvt_nop, cumax, true)
mag_gen_stub_cuext(int64_t, int64, mag_cvt_nop, cumax, true)

mag_gen_stub_cuext(float, float32, mag_cvt_nop, cumin, false)
mag_gen_stub_cuext(mag_float16_t, float16, mag_float16_to_float32, cumin, false)
mag_gen_stub_cuext(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, cumin, false)
mag_gen_stub_cuext(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, cumin, false)
mag_gen_stub_cuext(uint8_t, uint8, mag_cvt_nop, cumin, false)
mag_gen_stub_cuext(int8_t, int8, mag_cvt_nop, cumin, false)
mag_gen_stub_cuext(uint16_t, uint16, mag_cvt_nop, cumin, false)
mag_gen_stub_cuext(int16_t, int16, mag_cvt_nop, cumin, false)
mag_gen_stub_cuext(uint32_t, uint32, mag_cvt_nop, cumin, false)
mag_gen_stub_cuext(int32_t, int32, mag_cvt_nop, cumin, false)
mag_gen_stub_cuext(uint64_t, uint64, mag_cvt_nop, cumin, false)
mag_gen_stub_cuext(int64_t, int64, mag_cvt_nop, cumin, false)

#undef mag_gen_stub_cuext
#undef mag_scan_outer_offsets2
#undef mag_scan_outer_offsets
#undef mag_scan_dim_setup
