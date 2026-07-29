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

static MAG_AINLINE float mag_bf16_dot(const mag_bfloat16_t *x, const mag_bfloat16_t *y, int64_t K) {
  mag_vf32_t vacc0 = mag_vf32_zero();
  mag_vf32_t vacc1 = mag_vf32_zero();
  int64_t k=0;
  for (; k + (MAG_VBF16_LANES<<1)-1 < K; k += (MAG_VBF16_LANES<<1)) {
    vacc0 = mag_vf32_dpbf16(vacc0, mag_vbf16_loadu(x+k), mag_vbf16_loadu(y+k));
    vacc1 = mag_vf32_dpbf16(vacc1, mag_vbf16_loadu(x+k+MAG_VBF16_LANES), mag_vbf16_loadu(y+k+MAG_VBF16_LANES));
  }
  for (; k+MAG_VBF16_LANES-1 < K; k += MAG_VBF16_LANES)
    vacc0 = mag_vf32_dpbf16(vacc0, mag_vbf16_loadu(x+k), mag_vbf16_loadu(y+k));
  float acc = mag_vf32_reduce_add(mag_vf32_add(vacc0, vacc1));
  for (; k < K; ++k) acc += mag_bfloat16_to_float32(x[k])*mag_bfloat16_to_float32(y[k]);
  return acc;
}

typedef void (mag_dot_kernel_contig_t)(int64_t numel, void *r, const void *px, const void *py);
#define mag_dot_kernel_contig_impl(T, TtoF32, F32toT, LoadTtoF32) \
  static MAG_HOTPROC void mag_dot_kernel_contig_##T(int64_t numel, void *pr, const void *px, const void *py) { \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
    mag_vf32_t vacc0 = mag_vf32_zero(); \
    mag_vf32_t vacc1 = mag_vf32_zero(); \
    mag_vf32_t vacc2 = mag_vf32_zero(); \
    mag_vf32_t vacc3 = mag_vf32_zero(); \
    int64_t i = 0; \
    for (; i+4*MAG_VF32_LANES-1 < numel; i += 4*MAG_VF32_LANES) { \
      vacc0 = mag_vf32_fmadd(LoadTtoF32(x + i + 0*MAG_VF32_LANES), LoadTtoF32(y + i + 0*MAG_VF32_LANES), vacc0); \
      vacc1 = mag_vf32_fmadd(LoadTtoF32(x + i + 1*MAG_VF32_LANES), LoadTtoF32(y + i + 1*MAG_VF32_LANES), vacc1); \
      vacc2 = mag_vf32_fmadd(LoadTtoF32(x + i + 2*MAG_VF32_LANES), LoadTtoF32(y + i + 2*MAG_VF32_LANES), vacc2); \
      vacc3 = mag_vf32_fmadd(LoadTtoF32(x + i + 3*MAG_VF32_LANES), LoadTtoF32(y + i + 3*MAG_VF32_LANES), vacc3); \
    } \
    for (; i+MAG_VF32_LANES-1 < numel; i += MAG_VF32_LANES) \
      vacc0 = mag_vf32_fmadd(LoadTtoF32(x + i), LoadTtoF32(y + i), vacc0); \
    float acc = mag_vf32_reduce_add(mag_vf32_add(mag_vf32_add(vacc0, vacc1), mag_vf32_add(vacc2, vacc3))); \
    for (; i < numel; ++i) \
      acc += TtoF32(x[i])*TtoF32(y[i]); \
    *(T *)pr = F32toT(acc); \
  }
mag_dot_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu)
mag_dot_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16)
mag_dot_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn)
#undef mag_dot_kernel_contig_impl
/* Specialized kernel */
static MAG_HOTPROC void mag_dot_kernel_contig_mag_bfloat16_t(int64_t numel, void *pr, const void *px, const void *py) {
  *(mag_bfloat16_t *)pr = mag_float32_to_bfloat16(mag_bf16_dot(px, py, numel));
}

typedef void (mag_dot_kernel_strided_t)(int64_t numel, void *r, const void *px, const void *py, int64_t sx, int64_t sy);
#define mag_dot_kernel_strided_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_dot_kernel_strided_##T(int64_t numel, void *pr, const void *px, const void *py, int64_t sx, int64_t sy) { \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
    float acc=0.f; \
    for (int64_t i=0; i < numel; ++i) \
      acc += TtoF32(x[i*sx])*TtoF32(y[i*sy]); \
    *(T *)pr = F32toT(acc); \
  }
mag_dot_kernel_strided_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_dot_kernel_strided_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_dot_kernel_strided_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_dot_kernel_strided_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_dot_kernel_strided_impl

static MAG_HOTPROC void mag_matmul_dot_impl(
  mag_dtype_t dtype,
  int64_t ti,
  int64_t N,
  void *pr,
  const void *px, int64_t sx,
  const void *py, int64_t sy
) {
  if (ti != 0) return;
  static mag_dot_kernel_contig_t *const kernel_lut_contig[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_dot_kernel_contig_float,
    [MAG_DTYPE_FLOAT16] = &mag_dot_kernel_contig_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_dot_kernel_contig_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_dot_kernel_contig_mag_float8_e4m3fn_t
  };
  static mag_dot_kernel_strided_t *const kernel_lut_strided[4] = {
    [MAG_DTYPE_FLOAT32] = &mag_dot_kernel_strided_float,
    [MAG_DTYPE_FLOAT16] = &mag_dot_kernel_strided_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = &mag_dot_kernel_strided_mag_bfloat16_t,
    [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_dot_kernel_strided_mag_float8_e4m3fn_t
  };
  if (sx == 1 && sy == 1) /* Contig fast path */
    (*kernel_lut_contig[dtype])(N, pr, px, py);
  else
    (*kernel_lut_strided[dtype])(N, pr, px, py, sx, sy);
}

static MAG_HOTPROC void mag_matmul_dot(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = payload->cmd->out[0];
  const mag_tensor_t *x = payload->cmd->in[0];
  const mag_tensor_t *y = payload->cmd->in[1];
  void *pr = (void *)mag_tensor_data_ptr_mut(r);
  const void *px = (const void *)mag_tensor_data_ptr(x);
  const void *py = (const void *)mag_tensor_data_ptr(y);
  int64_t sx = x->meta.coords.strides[0];
  int64_t sy = y->meta.coords.strides[0];
  int64_t N = x->meta.coords.shape[0];
  mag_matmul_dot_impl(r->meta.dtype, payload->thread_idx, N, pr, px, sx, py, sy);
}
