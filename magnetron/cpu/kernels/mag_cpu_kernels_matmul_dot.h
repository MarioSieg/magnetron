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

typedef void (mag_dot_kernel_contig_t)(int64_t numel, void *r, const void *px, const void *py);
#define mag_dot_kernel_contig_impl(T, TtoF32, F32toT) \
  static MAG_HOTPROC void mag_dot_kernel_contig_##T(int64_t numel, void *pr, const void *px, const void *py) { \
    const T *x = (const T *)px; \
    const T *y = (const T *)py; \
    float acc=0.f; \
    for (int64_t i=0; i < numel; ++i) \
      acc += TtoF32(x[i])*TtoF32(y[i]); \
    *(T *)pr = F32toT(acc); \
  }
mag_dot_kernel_contig_impl(float, mag_cvt_nop, mag_cvt_nop)
mag_dot_kernel_contig_impl(mag_float16_t, mag_float16_to_float32, mag_float32_to_float16)
mag_dot_kernel_contig_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_dot_kernel_contig_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_dot_kernel_contig_impl

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
  int64_t sx = x->coords.strides[0];
  int64_t sy = y->coords.strides[0];
  int64_t N = x->coords.shape[0];
  mag_matmul_dot_impl(r->dtype, payload->thread_idx, N, pr, px, sx, py, sy);
}
