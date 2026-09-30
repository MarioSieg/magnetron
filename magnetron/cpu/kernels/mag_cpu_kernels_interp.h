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

#include <core/mag_interp_plan.h>

static void *mag_interp_arena_alloc(void *ud, size_t nb) {
  (void)ud;
  return mag_scratch_arena_alloc(&mag_tls_arena, nb);
}

#define mag_interp_cvt_i2f(x) ((float)(x))
#define mag_interp_cvt_f2u8(x) ((uint8_t)(x))
#define mag_interp_cvt_f2i8(x) ((int8_t)(x))
#define mag_interp_cvt_f2u16(x) ((uint16_t)(x))
#define mag_interp_cvt_f2i16(x) ((int16_t)(x))
#define mag_interp_cvt_f2u32(x) ((uint32_t)(x))
#define mag_interp_cvt_f2i32(x) ((int32_t)(x))
#define mag_interp_cvt_f2u64(x) ((uint64_t)(x))
#define mag_interp_cvt_f2i64(x) ((int64_t)(x))

#define mag_gen_stub_interp(NAME, T, TF, CVT, RCVT, TRANSPOSED) \
  static MAG_HOTPROC mag_status_t NAME##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    if (mag_unlikely(r->meta.numel == 0)) return MAG_OK; \
    const mag_tensor_t *small = TRANSPOSED ? r : x; \
    const mag_tensor_t *big = TRANSPOSED ? x : r; \
    size_t mark = mag_scratch_arena_mark(&mag_tls_arena); \
    mag_interp_plan_t plan; \
    if (mag_unlikely(!mag_interp_plan_build(&plan, payload->cmd->params, small->meta.coords.shape, big->meta.coords.shape, small->meta.coords.rank, TRANSPOSED, &mag_interp_arena_alloc, NULL))) { \
      mag_scratch_arena_reset(&mag_tls_arena, mark); \
      return mag_set_error(err, MAG_ERR_OOM, "interpolate: failed to allocate resampling plan."); \
    } \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const mag_interp_axis_t *a0 = plan.axes; \
    const mag_interp_axis_t *a1 = plan.axes+1; \
    const mag_interp_axis_t *a2 = plan.axes+2; \
    int64_t I1 = a1->in, I2 = a2->in, I0 = a0->in; \
    int64_t O1 = a1->out, O2 = a2->out, O0 = a0->out; \
    int64_t rows = plan.planes*O0*O1; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (rows + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra + chunk, rows); \
    bool copy = !TRANSPOSED && mag_interp_mode_is_nearest((mag_interp_mode_t)payload->cmd->params->interp.mode); \
    for (int64_t row=ra; row < rb; ++row) { \
      int64_t tmp = row; \
      int64_t o1 = tmp % O1; tmp /= O1; \
      int64_t o0 = tmp % O0; tmp /= O0; \
      int64_t plane = tmp; \
      T *rrow = br + row*O2; \
      if (copy) { \
        const T *xrow = bx + ((plane*I0 + a0->idx[o0])*I1 + a1->idx[o1])*I2; \
        const int64_t *i2 = a2->idx; \
        if (I2 == O2 && a2->max_taps == 1 && i2[0] == 0 && i2[O2-1] == O2-1) memcpy(rrow, xrow, (size_t)O2*sizeof(T)); \
        else for (int64_t o2=0; o2 < O2; ++o2) rrow[o2] = xrow[i2[o2*a2->max_taps]]; \
        continue; \
      } \
      int64_t n0 = a0->ntaps[o0], n1 = a1->ntaps[o1]; \
      const int64_t *i0 = a0->idx + o0*a0->max_taps; \
      const int64_t *i1 = a1->idx + o1*a1->max_taps; \
      const float *w0 = a0->w + o0*a0->max_taps; \
      const float *w1 = a1->w + o1*a1->max_taps; \
      for (int64_t o2=0; o2 < O2; ++o2) { \
        int64_t n2 = a2->ntaps[o2]; \
        const int64_t *i2 = a2->idx + o2*a2->max_taps; \
        const float *w2 = a2->w + o2*a2->max_taps; \
        float acc = .0f; \
        for (int64_t k0=0; k0 < n0; ++k0) { \
          for (int64_t k1=0; k1 < n1; ++k1) { \
            float ww = w0[k0]*w1[k1]; \
            const T *xrow = bx + ((plane*I0 + i0[k0])*I1 + i1[k1])*I2; \
            for (int64_t k2=0; k2 < n2; ++k2) acc += ww*w2[k2]*CVT(xrow[i2[k2]]); \
          } \
        } \
        rrow[o2] = RCVT(acc); \
      } \
    } \
    mag_scratch_arena_reset(&mag_tls_arena, mark); \
    return MAG_OK; \
  }

mag_gen_stub_interp(mag_interpolate, float, float32, mag_cvt_nop, mag_cvt_nop, false)
mag_gen_stub_interp(mag_interpolate, mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16, false)
mag_gen_stub_interp(mag_interpolate, mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16, false)
mag_gen_stub_interp(mag_interpolate, mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, false)
mag_gen_stub_interp(mag_interpolate, uint8_t, uint8, mag_interp_cvt_i2f, mag_interp_cvt_f2u8, false)
mag_gen_stub_interp(mag_interpolate, int8_t, int8, mag_interp_cvt_i2f, mag_interp_cvt_f2i8, false)
mag_gen_stub_interp(mag_interpolate, uint16_t, uint16, mag_interp_cvt_i2f, mag_interp_cvt_f2u16, false)
mag_gen_stub_interp(mag_interpolate, int16_t, int16, mag_interp_cvt_i2f, mag_interp_cvt_f2i16, false)
mag_gen_stub_interp(mag_interpolate, uint32_t, uint32, mag_interp_cvt_i2f, mag_interp_cvt_f2u32, false)
mag_gen_stub_interp(mag_interpolate, int32_t, int32, mag_interp_cvt_i2f, mag_interp_cvt_f2i32, false)
mag_gen_stub_interp(mag_interpolate, uint64_t, uint64, mag_interp_cvt_i2f, mag_interp_cvt_f2u64, false)
mag_gen_stub_interp(mag_interpolate, int64_t, int64, mag_interp_cvt_i2f, mag_interp_cvt_f2i64, false)

mag_gen_stub_interp(mag_interpolate_back, float, float32, mag_cvt_nop, mag_cvt_nop, true)
mag_gen_stub_interp(mag_interpolate_back, mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16, true)
mag_gen_stub_interp(mag_interpolate_back, mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16, true)
mag_gen_stub_interp(mag_interpolate_back, mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, true)

#undef mag_gen_stub_interp
