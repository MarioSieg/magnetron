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

#include <core/mag_reduce_plan.h>

#define mag_cpu_impl_reduce_axes(T, OT, TF, FUNC, ACC_T, INIT_EXPR, UPDATE_STMT, FINAL_STMT) \
  static mag_status_t MAG_HOTPROC mag_##FUNC##_##TF(mag_error_t *err,const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    OT *br = (OT *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const mag_reduce_plan_t *plan = &payload->cmd->params->reduction.red_plan; \
    int64_t numel = r->meta.numel; \
    int64_t red_prod = plan->red_prod; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (numel + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_vmin(oa + chunk, numel); \
    bool mag_contig = plan->rank == 1 && plan->red_strides[0] == 1; \
    for (int64_t oi=oa; oi < ob; ++oi) { \
      int64_t base = mag_reduce_plan_to_offset(plan, oi); \
      ACC_T acc = INIT_EXPR; \
      if (mag_contig) { \
        for (int64_t ri=0; ri < red_prod; ++ri) { \
          int64_t roff = base + ri; \
          mag_bnd_chk(bx + roff, x->storage->base, x->storage->size); \
          { UPDATE_STMT } \
        } \
      } else { \
        for (int64_t ri=0; ri < red_prod; ++ri) { \
          int64_t tmp = ri; \
          int64_t roff = base; \
          for (int64_t k=plan->rank - 1; k >= 0; --k) { \
            int64_t sz = plan->red_sizes[k]; \
            int64_t idx = tmp % sz; \
            tmp /= sz; \
            roff += idx*plan->red_strides[k]; \
          } \
          mag_bnd_chk(bx + roff, x->storage->base, x->storage->size); \
          { UPDATE_STMT } \
        } \
      } \
      OT *o = br + oi; \
      { FINAL_STMT } \
    } \
    return MAG_OK; \
  }


#define mag_add_f32(a, b) ((a)+(b))
#define mag_cpu_impl_reduce_hfp(T, TF, FUNC, CVT, RCVT, VLOAD, VINIT, VACC, VRED, SINIT, SCOMB, FINAL_ACC) \
  static mag_status_t MAG_HOTPROC mag_##FUNC##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const mag_reduce_plan_t *plan = &payload->cmd->params->reduction.red_plan; \
    int64_t numel = r->meta.numel; \
    int64_t red_prod = plan->red_prod; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (numel + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_vmin(oa + chunk, numel); \
    bool mag_contig = plan->rank == 1 && plan->red_strides[0] == 1; \
    for (int64_t oi=oa; oi < ob; ++oi) { \
      int64_t base = mag_reduce_plan_to_offset(plan, oi); \
      float acc = (SINIT); \
      if (mag_contig) { \
        const T *p = bx + base; \
        int64_t i = 0; \
        mag_vf32_t vacc = (VINIT); \
        for (; i + MAG_VF32_LANES <= red_prod; i += MAG_VF32_LANES) { \
          mag_bnd_chk(p + i + MAG_VF32_LANES - 1, x->storage->base, x->storage->size); \
          vacc = VACC(vacc, VLOAD(p + i)); \
        } \
        acc = SCOMB(acc, VRED(vacc)); \
        for (; i < red_prod; ++i) { float xv = CVT(p[i]); acc = SCOMB(acc, xv); } \
      } else { \
        for (int64_t ri=0; ri < red_prod; ++ri) { \
          int64_t tmp = ri; \
          int64_t roff = base; \
          for (int64_t k=plan->rank - 1; k >= 0; --k) { \
            int64_t sz = plan->red_sizes[k]; \
            int64_t idx = tmp % sz; \
            tmp /= sz; \
            roff += idx*plan->red_strides[k]; \
          } \
          mag_bnd_chk(bx + roff, x->storage->base, x->storage->size); \
          float xv = CVT(bx[roff]); acc = SCOMB(acc, xv); \
        } \
      } \
      br[oi] = RCVT(FINAL_ACC(acc)); \
    } \
    return MAG_OK; \
  }

#define mag_final_id(a) (a)
#define mag_final_mean(a) ((a)/(float)red_prod)
mag_cpu_impl_reduce_hfp(float,          float32,  sum,    mag_cvt_nop,             mag_cvt_nop,               mag_vf32_loadu,      mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_id)
mag_cpu_impl_reduce_hfp(mag_float16_t,  float16,  sum,    mag_float16_to_float32,  mag_float32_to_float16,   mag_vf32_loadu_f16,  mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_id)
mag_cpu_impl_reduce_hfp(mag_bfloat16_t, bfloat16, sum,    mag_bfloat16_to_float32, mag_float32_to_bfloat16,  mag_vf32_loadu_bf16, mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_id)
mag_cpu_impl_reduce_hfp(float,          float32,  mean,   mag_cvt_nop,             mag_cvt_nop,               mag_vf32_loadu,      mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_mean)
mag_cpu_impl_reduce_hfp(mag_float16_t,  float16,  mean,   mag_float16_to_float32,  mag_float32_to_float16,   mag_vf32_loadu_f16,  mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_mean)
mag_cpu_impl_reduce_hfp(mag_bfloat16_t, bfloat16, mean,   mag_bfloat16_to_float32, mag_float32_to_bfloat16,  mag_vf32_loadu_bf16, mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_mean)
mag_cpu_impl_reduce_hfp(float,          float32,  maxima, mag_cvt_nop,             mag_cvt_nop,               mag_vf32_loadu,      mag_vf32_splat(-INFINITY),     mag_vf32_max, mag_vf32_reduce_max, -INFINITY, fmaxf,       mag_final_id)
mag_cpu_impl_reduce_hfp(mag_float16_t,  float16,  maxima, mag_float16_to_float32,  mag_float32_to_float16,   mag_vf32_loadu_f16,  mag_vf32_splat(-INFINITY),     mag_vf32_max, mag_vf32_reduce_max, -INFINITY, fmaxf,       mag_final_id)
mag_cpu_impl_reduce_hfp(mag_bfloat16_t, bfloat16, maxima, mag_bfloat16_to_float32, mag_float32_to_bfloat16,  mag_vf32_loadu_bf16, mag_vf32_splat(-INFINITY),     mag_vf32_max, mag_vf32_reduce_max, -INFINITY, fmaxf,       mag_final_id)
#undef mag_final_id
#undef mag_final_mean

mag_cpu_impl_reduce_axes(mag_float8_e4m3fn_t, mag_float8_e4m3fn_t, float8_e4m3fn, mean, float, 0.0f, acc += mag_float8_e4m3fn_to_float32(bx[roff]);, acc /= (float)red_prod; *o = mag_float32_to_float8_e4m3fn(acc); )

/* float32/float16/bfloat16 sum are SIMD-specialized above via mag_cpu_impl_reduce_hfp. */
mag_cpu_impl_reduce_axes(mag_float8_e4m3fn_t, mag_float8_e4m3fn_t, float8_e4m3fn, sum, float, 0.0f, acc += mag_float8_e4m3fn_to_float32(bx[roff]);,*o = mag_float32_to_float8_e4m3fn(acc); )
mag_cpu_impl_reduce_axes(uint8_t, uint64_t, uint8, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int64_t, int8, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint64_t, uint16, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int64_t, int16, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint64_t, uint32, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int64_t, int32, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )

mag_cpu_impl_reduce_axes(float, float, float32, prod, double, 1.0, acc *= (double)bx[roff];, *o = (float)acc; )
mag_cpu_impl_reduce_axes(mag_float16_t, mag_float16_t, float16, prod, float, 1.0f, acc *= mag_float16_to_float32(bx[roff]);, *o = mag_float32_to_float16(acc); )
mag_cpu_impl_reduce_axes(mag_bfloat16_t, mag_bfloat16_t, bfloat16, prod, float, 1.0f, acc *= mag_bfloat16_to_float32(bx[roff]);, *o = mag_float32_to_bfloat16(acc); )
mag_cpu_impl_reduce_axes(mag_float8_e4m3fn_t, mag_float8_e4m3fn_t, float8_e4m3fn, prod, float, 1.0f, acc *= mag_float8_e4m3fn_to_float32(bx[roff]);, *o = mag_float32_to_float8_e4m3fn(acc); )
mag_cpu_impl_reduce_axes(uint8_t, uint64_t, uint8, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int64_t, int8, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint64_t, uint16, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int64_t, int16, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint64_t, uint32, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int64_t, int32, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )

mag_cpu_impl_reduce_axes(float, float, float32, minima, float, INFINITY, acc = fminf(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(mag_float16_t, mag_float16_t, float16, minima, float, INFINITY, acc = fminf(acc, mag_float16_to_float32(bx[roff]));, *o = mag_float32_to_float16(acc); )
mag_cpu_impl_reduce_axes(mag_bfloat16_t, mag_bfloat16_t, bfloat16, minima, float, INFINITY, acc = fminf(acc, mag_bfloat16_to_float32(bx[roff]));, *o = mag_float32_to_bfloat16(acc); )
mag_cpu_impl_reduce_axes(mag_float8_e4m3fn_t, mag_float8_e4m3fn_t, float8_e4m3fn, minima, float, INFINITY, acc = fminf(acc, mag_float8_e4m3fn_to_float32(bx[roff]));, *o = mag_float32_to_float8_e4m3fn(acc); )
mag_cpu_impl_reduce_axes(uint8_t, uint8_t, uint8, minima, uint8_t, UINT8_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int8_t, int8, minima, int8_t, INT8_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint16_t, uint16, minima, uint16_t, UINT16_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int16_t, int16, minima, int16_t, INT16_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint32_t, uint32, minima, uint32_t, UINT32_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int32_t, int32, minima, int32_t, INT32_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, minima, uint64_t, UINT64_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, minima, int64_t, INT64_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )

/* float32/float16/bfloat16 maxima are SIMD-specialized above via mag_cpu_impl_reduce_hfp. */
mag_cpu_impl_reduce_axes(mag_float8_e4m3fn_t, mag_float8_e4m3fn_t, float8_e4m3fn, maxima, float, -INFINITY, acc = fmaxf(acc, mag_float8_e4m3fn_to_float32(bx[roff]));, *o = mag_float32_to_float8_e4m3fn(acc); )
mag_cpu_impl_reduce_axes(uint8_t, uint8_t, uint8, maxima, uint8_t, 0, acc = mag_vmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int8_t, int8, maxima, int8_t, INT8_MIN, acc = mag_vmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint16_t, uint16, maxima, uint16_t, 0, acc = mag_vmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int16_t, int16, maxima, int16_t, INT16_MIN, acc = mag_vmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint32_t, uint32, maxima, uint32_t, 0, acc = mag_vmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int32_t, int32, maxima, int32_t, INT32_MIN, acc = mag_vmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, maxima, uint64_t, 0, acc = mag_vmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, maxima, int64_t, INT64_MIN, acc = mag_vmax(acc, bx[roff]);, *o = acc; )

typedef struct mag_argmax_acc_f32_t {
  float val;
  int64_t idx;
  bool set;
} mag_argmax_acc_f32_t;

typedef struct mag_argmax_acc_i64_t {
  int64_t val;
  int64_t idx;
  bool set;
} mag_argmax_acc_i64_t;

mag_cpu_impl_reduce_axes(
  float,
  int64_t,
  float32,
  argmax,
  mag_argmax_acc_f32_t,
  {0},
  {
    float xv = bx[roff];
    if (!acc.set || xv > acc.val) {
      acc.val = xv;
      acc.idx = ri;
      acc.set = true;
    }
  },
  {
    *o = acc.idx;
  }
);

mag_cpu_impl_reduce_axes(
  float,
  int64_t,
  float32,
  argmin,
  mag_argmax_acc_f32_t,
  {0},
  {
    float xv = bx[roff];
    if (!acc.set || xv < acc.val) {
      acc.val = xv;
      acc.idx = ri;
      acc.set = true;
    }
  },
  {
    *o = acc.idx;
  }
);

mag_cpu_impl_reduce_axes(
  mag_float16_t,
  int64_t,
  float16,
  argmax,
  mag_argmax_acc_f32_t,
  {0},
  {
    float xv = mag_float16_to_float32(bx[roff]);
    if (!acc.set || xv > acc.val) {
      acc.val = xv;
      acc.idx = ri;
      acc.set = true;
    }
  },
  {
    *o = acc.idx;
  }
);

mag_cpu_impl_reduce_axes(
  mag_float16_t,
  int64_t,
  float16,
  argmin,
  mag_argmax_acc_f32_t,
  {0},
  {
    float xv = mag_float16_to_float32(bx[roff]);
    if (!acc.set || xv < acc.val) {
      acc.val = xv;
      acc.idx = ri;
      acc.set = true;
    }
  },
  {
    *o = acc.idx;
  }
);

mag_cpu_impl_reduce_axes(
  mag_bfloat16_t,
  int64_t,
  bfloat16,
  argmax,
  mag_argmax_acc_f32_t,
  {0},
  {
    float xv = mag_bfloat16_to_float32(bx[roff]);
    if (!acc.set || xv > acc.val) {
      acc.val = xv;
      acc.idx = ri;
      acc.set = true;
    }
  },
  {
    *o = acc.idx;
  }
);

mag_cpu_impl_reduce_axes(
  mag_bfloat16_t,
  int64_t,
  bfloat16,
  argmin,
  mag_argmax_acc_f32_t,
  {0},
  {
    float xv = mag_bfloat16_to_float32(bx[roff]);
    if (!acc.set || xv < acc.val) {
      acc.val = xv;
      acc.idx = ri;
      acc.set = true;
    }
  },
  {
    *o = acc.idx;
  }
);

mag_cpu_impl_reduce_axes(
  mag_float8_e4m3fn_t,
  int64_t,
  float8_e4m3fn,
  argmax,
  mag_argmax_acc_f32_t,
  {0},
  {
    float xv = mag_float8_e4m3fn_to_float32(bx[roff]);
    if (!acc.set || xv > acc.val) {
      acc.val = xv;
      acc.idx = ri;
      acc.set = true;
    }
  },
  {
    *o = acc.idx;
  }
);

mag_cpu_impl_reduce_axes(
  mag_float8_e4m3fn_t,
  int64_t,
  float8_e4m3fn,
  argmin,
  mag_argmax_acc_f32_t,
  {0},
  {
    float xv = mag_float8_e4m3fn_to_float32(bx[roff]);
    if (!acc.set || xv < acc.val) {
      acc.val = xv;
      acc.idx = ri;
      acc.set = true;
    }
  },
  {
    *o = acc.idx;
  }
);

#define mag_cpu_impl_argminmax_int(T, TF) \
  mag_cpu_impl_reduce_axes( \
    T, int64_t, TF, argmax, mag_argmax_acc_i64_t, \
    {0}, \
    { \
      int64_t xv = (int64_t)bx[roff]; \
      if (!acc.set || xv > acc.val) { \
        acc.val = xv; \
        acc.idx = ri; \
        acc.set = true; \
      } \
    }, \
    { *o = acc.idx; } \
  ); \
  mag_cpu_impl_reduce_axes( \
    T, int64_t, TF, argmin, mag_argmax_acc_i64_t, \
    {0}, \
    { \
      int64_t xv = (int64_t)bx[roff]; \
      if (!acc.set || xv < acc.val) { \
        acc.val = xv; \
        acc.idx = ri; \
        acc.set = true; \
      } \
    }, \
    { *o = acc.idx; } \
  )

mag_cpu_impl_argminmax_int(uint8_t,  uint8);
mag_cpu_impl_argminmax_int(int8_t,   int8);
mag_cpu_impl_argminmax_int(uint16_t, uint16);
mag_cpu_impl_argminmax_int(int16_t,  int16);
mag_cpu_impl_argminmax_int(uint32_t, uint32);
mag_cpu_impl_argminmax_int(int32_t,  int32);
mag_cpu_impl_argminmax_int(uint64_t, uint64);
mag_cpu_impl_argminmax_int(int64_t,  int64);

#undef mag_cpu_impl_argminmax_int

#undef mag_cpu_impl_reduce_axes

#define mag_cpu_impl_reduce_axes_logical(T, TF, FUNC, IDENTITY, UPDATE_STMT, BREAK_COND) \
  static mag_status_t MAG_HOTPROC mag_##FUNC##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    uint8_t *br = (uint8_t *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const mag_reduce_plan_t *plan = &payload->cmd->params->reduction.red_plan; \
    int64_t numel = r->meta.numel; \
    int64_t red_prod = plan->red_prod; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (numel + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_vmin(oa + chunk, numel); \
    for (int64_t oi=oa; oi < ob; ++oi) { \
      uint8_t acc = (IDENTITY); \
      if (red_prod == 0) { \
        br[oi] = acc; \
        continue; \
      } \
      int64_t base = mag_reduce_plan_to_offset(plan, oi); \
      for (int64_t ri=0; ri < red_prod; ++ri) { \
        int64_t tmp = ri; \
        int64_t roff = base; \
        for (int64_t k=plan->rank-1; k >= 0; --k) { \
          int64_t sz = plan->red_sizes[k]; \
          int64_t idx = tmp % sz; \
          tmp /= sz; \
          roff += idx*plan->red_strides[k]; \
        } \
        mag_bnd_chk(bx + roff, x->storage->base, x->storage->size); \
        { UPDATE_STMT } \
        if (BREAK_COND) break; \
      } \
      br[oi] = acc; \
    } \
    return MAG_OK; \
  }


#define mag_impl_logical_reduce_pair(T, TF, unpack) \
  mag_cpu_impl_reduce_axes_logical( \
    T, TF, any, \
    0, \
    { if (unpack(bx[roff]) != 0) acc = 1; }, \
    acc == 1 \
  ); \
  mag_cpu_impl_reduce_axes_logical( \
    T, TF, all, \
    1, \
    { if (unpack(bx[roff]) == 0) acc = 0; }, \
    acc == 0 \
  )

#define mag_unpack_nop(x) (x)
#define mag_unpack_packed(x) ((x).bits)

mag_impl_logical_reduce_pair(float, float32, mag_unpack_nop);
mag_impl_logical_reduce_pair(mag_float16_t, float16, mag_unpack_packed);
mag_impl_logical_reduce_pair(mag_bfloat16_t, bfloat16, mag_unpack_packed);
mag_impl_logical_reduce_pair(mag_float8_e4m3fn_t, float8_e4m3fn, mag_unpack_packed);
mag_impl_logical_reduce_pair(uint8_t, uint8, mag_unpack_nop);
mag_impl_logical_reduce_pair(int8_t, int8, mag_unpack_nop);
mag_impl_logical_reduce_pair(uint16_t, uint16, mag_unpack_nop);
mag_impl_logical_reduce_pair(int16_t, int16, mag_unpack_nop);
mag_impl_logical_reduce_pair(uint32_t, uint32, mag_unpack_nop);
mag_impl_logical_reduce_pair(int32_t, int32, mag_unpack_nop);
mag_impl_logical_reduce_pair(uint64_t, uint64, mag_unpack_nop);
mag_impl_logical_reduce_pair(int64_t, int64, mag_unpack_nop);

#undef mag_unpack_nop
#undef mag_unpack_packed

#undef mag_impl_logical_reduce_pair
