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

#define MAG_REDUCE_ROW_BLOCK 256
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
    int64_t red_rank = plan->red_rank; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t od = red_rank > 0 && plan->red_strides[red_rank-1] == 1 ? red_rank-1 : red_rank; \
    int64_t inner = od < red_rank ? plan->red_sizes[red_rank-1] : 1; \
    int64_t outer = red_prod/inner; \
    int64_t row_len = plan->nk > 0 && plan->in_strides[plan->keep_axes[plan->nk-1]] == 1 ? plan->in_shape[plan->keep_axes[plan->nk-1]] : 0; \
    int64_t chunk = (numel + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_vmin(oa + chunk, numel); \
    int64_t oi = oa; \
    while (oi < ob) { \
      ACC_T acc = INIT_EXPR; \
      if (row_len >= 2 && inner == 1) { \
        int64_t row_start = oi - oi%row_len; \
        int64_t seg_end = mag_vmin(ob, row_start + row_len); \
        int64_t base = mag_reduce_plan_to_offset(plan, row_start); \
        for (int64_t j0=oi-row_start; j0 < seg_end-row_start; j0 += MAG_REDUCE_ROW_BLOCK) { \
          int64_t nb = mag_vmin((int64_t)MAG_REDUCE_ROW_BLOCK, seg_end-row_start-j0); \
          ACC_T accs[MAG_REDUCE_ROW_BLOCK]; \
          for (int64_t j=0; j < nb; ++j) { ACC_T init = INIT_EXPR; accs[j] = init; } \
          for (int64_t ri=0; ri < red_prod; ++ri) { \
            int64_t tmp = ri; \
            int64_t roff0 = base + j0; \
            for (int64_t k=red_rank - 1; k >= 0; --k) { \
              int64_t sz = plan->red_sizes[k]; \
              int64_t idx = tmp % sz; \
              tmp /= sz; \
              roff0 += idx*plan->red_strides[k]; \
            } \
            mag_bnd_chk(bx + roff0 + nb - 1, x->storage->base, x->storage->size); \
            for (int64_t j=0; j < nb; ++j) { \
              int64_t roff = roff0 + j; \
              acc = accs[j]; \
              { UPDATE_STMT } \
              accs[j] = acc; \
            } \
          } \
          for (int64_t j=0; j < nb; ++j) { \
            OT *o = br + row_start + j0 + j; \
            acc = accs[j]; \
            { FINAL_STMT } \
          } \
        } \
        oi = seg_end; \
        continue; \
      } \
      int64_t base = mag_reduce_plan_to_offset(plan, oi); \
      for (int64_t ro=0; ro < outer; ++ro) { \
        int64_t tmp = ro; \
        int64_t roff0 = base; \
        for (int64_t k=od - 1; k >= 0; --k) { \
          int64_t sz = plan->red_sizes[k]; \
          int64_t idx = tmp % sz; \
          tmp /= sz; \
          roff0 += idx*plan->red_strides[k]; \
        } \
        mag_bnd_chk(bx + roff0 + inner - 1, x->storage->base, x->storage->size); \
        for (int64_t j=0; j < inner; ++j) { \
          int64_t roff = roff0 + j; \
          int64_t ri = ro*inner + j; \
          (void)ri; \
          { UPDATE_STMT } \
        } \
      } \
      OT *o = br+oi; \
      { FINAL_STMT } \
      ++oi; \
    } \
    return MAG_OK; \
  }


#define mag_add_f32(a, b) ((a)+(b))
#define mag_final_id(a) (a)
#define mag_final_mean(a) ((a)/(float)red_prod)
static MAG_AINLINE float mag_vf32_reduce_min_lanes(mag_vf32_t v) {
  float lanes[MAG_VF32_LANES];
  mag_vf32_storeu(lanes, v);
  float m = lanes[0];
  for (int i=1; i < MAG_VF32_LANES; ++i) m = fminf(m, lanes[i]);
  return m;
}
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
    int64_t red_rank = plan->red_rank; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t od = red_rank > 0 && plan->red_strides[red_rank-1] == 1 ? red_rank-1 : red_rank; \
    int64_t inner = od < red_rank ? plan->red_sizes[red_rank-1] : 1; \
    int64_t outer = red_prod/inner; \
    int64_t row_len = plan->nk > 0 && plan->in_strides[plan->keep_axes[plan->nk-1]] == 1 ? plan->in_shape[plan->keep_axes[plan->nk-1]] : 0; \
    int64_t chunk = (numel+tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_vmin(oa+chunk, numel); \
    int64_t oi = oa; \
    while (oi < ob) { \
      if (row_len >= MAG_VF32_LANES && inner == 1) { \
        int64_t row_start = oi - oi%row_len; \
        int64_t seg_end = mag_vmin(ob, row_start + row_len); \
        int64_t base = mag_reduce_plan_to_offset(plan, row_start); \
        for (int64_t j0=oi-row_start; j0 < seg_end-row_start; j0 += MAG_REDUCE_ROW_BLOCK) { \
          int64_t nb = mag_vmin((int64_t)MAG_REDUCE_ROW_BLOCK, seg_end-row_start-j0); \
          float accs[MAG_REDUCE_ROW_BLOCK]; \
          for (int64_t j=0; j < nb; ++j) accs[j] = (SINIT); \
          for (int64_t ri=0; ri < red_prod; ++ri) { \
            int64_t tmp = ri; \
            int64_t roff = base+j0; \
            for (int64_t k=red_rank - 1; k >= 0; --k) { \
              int64_t sz = plan->red_sizes[k]; \
              int64_t idx = tmp % sz; \
              tmp /= sz; \
              roff += idx*plan->red_strides[k]; \
            } \
            const T *p = bx+roff; \
            mag_bnd_chk(p+nb - 1, x->storage->base, x->storage->size); \
            int64_t j = 0; \
            for (; j+MAG_VF32_LANES <= nb; j += MAG_VF32_LANES) mag_vf32_storeu(accs+j, VACC(mag_vf32_loadu(accs+j), VLOAD(p+j))); \
            for (; j < nb; ++j) { float xv = CVT(p[j]); accs[j] = SCOMB(accs[j], xv); } \
          } \
          for (int64_t j=0; j < nb; ++j) br[row_start+j0+j] = RCVT(FINAL_ACC(accs[j])); \
        } \
        oi = seg_end; \
        continue; \
      } \
      int64_t base = mag_reduce_plan_to_offset(plan, oi); \
      float acc = (SINIT); \
      mag_vf32_t vacc = (VINIT); \
      mag_vf32_t vacc1 = (VINIT); \
      mag_vf32_t vacc2 = (VINIT); \
      mag_vf32_t vacc3 = (VINIT); \
      for (int64_t ro=0; ro < outer; ++ro) { \
        int64_t tmp = ro; \
        int64_t roff = base; \
        for (int64_t k=od - 1; k >= 0; --k) { \
          int64_t sz = plan->red_sizes[k]; \
          int64_t idx = tmp % sz; \
          tmp /= sz; \
          roff += idx*plan->red_strides[k]; \
        } \
        const T *p = bx+roff; \
        mag_bnd_chk(p+inner - 1, x->storage->base, x->storage->size); \
        int64_t i = 0; \
        for (; i+4*MAG_VF32_LANES <= inner; i += 4*MAG_VF32_LANES) { \
          vacc = VACC(vacc, VLOAD(p+i)); \
          vacc1 = VACC(vacc1, VLOAD(p+i+MAG_VF32_LANES)); \
          vacc2 = VACC(vacc2, VLOAD(p+i+2*MAG_VF32_LANES)); \
          vacc3 = VACC(vacc3, VLOAD(p+i+3*MAG_VF32_LANES)); \
        } \
        for (; i+MAG_VF32_LANES <= inner; i += MAG_VF32_LANES) vacc = VACC(vacc, VLOAD(p+i)); \
        for (; i < inner; ++i) { float xv = CVT(p[i]); acc = SCOMB(acc, xv); } \
      } \
      vacc = VACC(vacc, vacc1); \
      vacc2 = VACC(vacc2, vacc3); \
      vacc = VACC(vacc, vacc2); \
      acc = SCOMB(acc, VRED(vacc)); \
      br[oi] = RCVT(FINAL_ACC(acc)); \
      ++oi; \
    } \
    return MAG_OK; \
  }

mag_cpu_impl_reduce_hfp(float,          float32,  sum,    mag_cvt_nop,             mag_cvt_nop,               mag_vf32_loadu,      mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_id)
mag_cpu_impl_reduce_hfp(mag_float16_t,  float16,  sum,    mag_float16_to_float32,  mag_float32_to_float16,   mag_vf32_loadu_f16,  mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_id)
mag_cpu_impl_reduce_hfp(mag_bfloat16_t, bfloat16, sum,    mag_bfloat16_to_float32, mag_float32_to_bfloat16,  mag_vf32_loadu_bf16, mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_id)
mag_cpu_impl_reduce_hfp(float,          float32,  mean,   mag_cvt_nop,             mag_cvt_nop,               mag_vf32_loadu,      mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_mean)
mag_cpu_impl_reduce_hfp(mag_float16_t,  float16,  mean,   mag_float16_to_float32,  mag_float32_to_float16,   mag_vf32_loadu_f16,  mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_mean)
mag_cpu_impl_reduce_hfp(mag_bfloat16_t, bfloat16, mean,   mag_bfloat16_to_float32, mag_float32_to_bfloat16,  mag_vf32_loadu_bf16, mag_vf32_zero(),               mag_vf32_add, mag_vf32_reduce_add, 0.0f,      mag_add_f32, mag_final_mean)
#define MAG_REDUCE_PROD_UNROLL 8
#define mag_cpu_impl_reduce_prod_hfp(T, TF, CVT, RCVT) \
  static mag_status_t MAG_HOTPROC mag_prod_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
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
    bool mag_contig = plan->red_rank == 1 && plan->red_strides[0] == 1; \
    int64_t row_len = plan->nk > 0 && plan->in_strides[plan->keep_axes[plan->nk-1]] == 1 ? plan->in_shape[plan->keep_axes[plan->nk-1]] : 0; \
    int64_t chunk = (numel + tc - 1)/tc; \
    if (row_len > 0 && !mag_contig) chunk = (chunk + row_len - 1)/row_len*row_len; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_vmin(oa + chunk, numel); \
    for (int64_t oi=oa; oi < ob; ++oi) { \
      int64_t base = mag_reduce_plan_to_offset(plan, oi); \
      double acc = 1.0; \
      if (mag_contig) { \
        const T *p = bx + base; \
        double lanes[MAG_REDUCE_PROD_UNROLL]; \
        for (int u=0; u < MAG_REDUCE_PROD_UNROLL; ++u) lanes[u] = 1.0; \
        int64_t i = 0; \
        for (; i + MAG_REDUCE_PROD_UNROLL <= red_prod; i += MAG_REDUCE_PROD_UNROLL) { \
          mag_bnd_chk(p + i + MAG_REDUCE_PROD_UNROLL - 1, x->storage->base, x->storage->size); \
          for (int u=0; u < MAG_REDUCE_PROD_UNROLL; ++u) lanes[u] *= (double)CVT(p[i + u]); \
        } \
        for (; i < red_prod; ++i) { \
          mag_bnd_chk(p + i, x->storage->base, x->storage->size); \
          acc *= (double)CVT(p[i]); \
        } \
        for (int u=0; u < MAG_REDUCE_PROD_UNROLL; ++u) acc *= lanes[u]; \
      } else if (row_len >= 2 && oi % row_len == 0 && oi + row_len <= ob) { \
        for (int64_t j0=0; j0 < row_len; j0 += MAG_REDUCE_ROW_BLOCK) { \
          int64_t nb = mag_vmin((int64_t)MAG_REDUCE_ROW_BLOCK, row_len-j0); \
          double accs[MAG_REDUCE_ROW_BLOCK]; \
          for (int64_t j=0; j < nb; ++j) accs[j] = 1.0; \
          for (int64_t ri=0; ri < red_prod; ++ri) { \
            int64_t tmp = ri; \
            int64_t roff0 = base + j0; \
            for (int64_t k=plan->red_rank - 1; k >= 0; --k) { \
              int64_t sz = plan->red_sizes[k]; \
              int64_t idx = tmp % sz; \
              tmp /= sz; \
              roff0 += idx*plan->red_strides[k]; \
            } \
            mag_bnd_chk(bx + roff0 + nb - 1, x->storage->base, x->storage->size); \
            const T *p = bx + roff0; \
            for (int64_t j=0; j < nb; ++j) accs[j] *= (double)CVT(p[j]); \
          } \
          for (int64_t j=0; j < nb; ++j) br[oi + j0 + j] = RCVT((float)accs[j]); \
        } \
        oi += row_len - 1; \
        continue; \
      } else { \
        for (int64_t ri=0; ri < red_prod; ++ri) { \
          int64_t tmp = ri; \
          int64_t roff = base; \
          for (int64_t k=plan->red_rank - 1; k >= 0; --k) { \
            int64_t sz = plan->red_sizes[k]; \
            int64_t idx = tmp % sz; \
            tmp /= sz; \
            roff += idx*plan->red_strides[k]; \
          } \
          mag_bnd_chk(bx + roff, x->storage->base, x->storage->size); \
          acc *= (double)CVT(bx[roff]); \
        } \
      } \
      br[oi] = RCVT((float)acc); \
    } \
    return MAG_OK; \
  }
mag_cpu_impl_reduce_prod_hfp(float, float32, mag_cvt_nop, mag_cvt_nop)
mag_cpu_impl_reduce_prod_hfp(mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16)
mag_cpu_impl_reduce_prod_hfp(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_cpu_impl_reduce_hfp(float, float32, minima, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu, mag_vf32_splat(INFINITY), mag_vf32_min, mag_vf32_reduce_min_lanes, INFINITY, fminf, mag_final_id)
mag_cpu_impl_reduce_hfp(mag_float16_t, float16, minima, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16, mag_vf32_splat(INFINITY), mag_vf32_min, mag_vf32_reduce_min_lanes, INFINITY, fminf, mag_final_id)
mag_cpu_impl_reduce_hfp(mag_bfloat16_t, bfloat16, minima, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16, mag_vf32_splat(INFINITY), mag_vf32_min, mag_vf32_reduce_min_lanes, INFINITY, fminf, mag_final_id)
mag_cpu_impl_reduce_hfp(float,          float32,  maxima, mag_cvt_nop,             mag_cvt_nop,               mag_vf32_loadu,      mag_vf32_splat(-INFINITY),     mag_vf32_max, mag_vf32_reduce_max, -INFINITY, fmaxf,       mag_final_id)
mag_cpu_impl_reduce_hfp(mag_float16_t,  float16,  maxima, mag_float16_to_float32,  mag_float32_to_float16,   mag_vf32_loadu_f16,  mag_vf32_splat(-INFINITY),     mag_vf32_max, mag_vf32_reduce_max, -INFINITY, fmaxf,       mag_final_id)
mag_cpu_impl_reduce_hfp(mag_bfloat16_t, bfloat16, maxima, mag_bfloat16_to_float32, mag_float32_to_bfloat16,  mag_vf32_loadu_bf16, mag_vf32_splat(-INFINITY),     mag_vf32_max, mag_vf32_reduce_max, -INFINITY, fmaxf,       mag_final_id)
mag_cpu_impl_reduce_hfp(mag_float8_e4m3fn_t, float8_e4m3fn, minima, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn, mag_vf32_splat(INFINITY), mag_vf32_min, mag_vf32_reduce_min_lanes, INFINITY, fminf, mag_final_id)
mag_cpu_impl_reduce_hfp(mag_float8_e4m3fn_t, float8_e4m3fn, maxima, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn, mag_vf32_splat(-INFINITY), mag_vf32_max, mag_vf32_reduce_max, -INFINITY, fmaxf, mag_final_id)
mag_cpu_impl_reduce_prod_hfp(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)
#undef mag_final_id
#undef mag_final_mean


/* float32/float16/bfloat16 sum are SIMD-specialized above via mag_cpu_impl_reduce_hfp. */
mag_cpu_impl_reduce_axes(uint8_t, uint64_t, uint8, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int64_t, int8, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint64_t, uint16, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int64_t, int16, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint64_t, uint32, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int64_t, int32, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )

mag_cpu_impl_reduce_axes(uint8_t, uint64_t, uint8, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int64_t, int8, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint64_t, uint16, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int64_t, int16, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint64_t, uint32, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int64_t, int32, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )

mag_cpu_impl_reduce_axes(uint8_t, uint8_t, uint8, minima, uint8_t, UINT8_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int8_t, int8, minima, int8_t, INT8_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint16_t, uint16, minima, uint16_t, UINT16_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int16_t, int16, minima, int16_t, INT16_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint32_t, uint32, minima, uint32_t, UINT32_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int32_t, int32, minima, int32_t, INT32_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, minima, uint64_t, UINT64_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, minima, int64_t, INT64_MAX, acc = mag_vmin(acc, bx[roff]);, *o = acc; )

/* float32/float16/bfloat16 maxima are SIMD-specialized above via mag_cpu_impl_reduce_hfp. */
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

#define mag_cpu_impl_argext_hfp(T, TF, FUNC, CVT, VLOAD, VCMP, SCMP) \
  static mag_status_t MAG_HOTPROC mag_##FUNC##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    int64_t *br = (int64_t *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const mag_reduce_plan_t *plan = &payload->cmd->params->reduction.red_plan; \
    int64_t numel = r->meta.numel; \
    int64_t red_prod = plan->red_prod; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (numel+tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_vmin(oa+chunk, numel); \
    bool mag_contig = plan->red_rank == 1 && plan->red_strides[0] == 1; \
    int64_t row_len = plan->nk > 0 && plan->in_strides[plan->keep_axes[plan->nk-1]] == 1 ? plan->in_shape[plan->keep_axes[plan->nk-1]] : 0; \
    if (row_len > 0 && !mag_contig) { chunk = (chunk + row_len - 1)/row_len*row_len; oa = ti*chunk; ob = mag_vmin(oa + chunk, numel); } \
    for (int64_t oi=oa; oi < ob; ++oi) { \
      int64_t base = mag_reduce_plan_to_offset(plan, oi); \
      float best = 0.f; \
      int64_t best_idx = 0; \
      bool set = false; \
      int64_t ri = 0; \
      if (mag_contig && red_prod >= MAG_VF32_LANES && red_prod < (1ll<<24)) { \
        const T *p = bx+base; \
        float lane_ids[MAG_VF32_LANES]; \
        for (int64_t l=0; l < MAG_VF32_LANES; ++l) lane_ids[l] = (float)l; \
        mag_vf32_t vlane = mag_vf32_loadu(lane_ids); \
        mag_vf32_t vbest = VLOAD(p); \
        mag_vf32_t vidx = vlane; \
        for (ri=MAG_VF32_LANES; ri+MAG_VF32_LANES <= red_prod; ri += MAG_VF32_LANES) { \
          mag_bnd_chk(p+ri+MAG_VF32_LANES - 1, x->storage->base, x->storage->size); \
          mag_vf32_t v = VLOAD(p+ri); \
          mag_vmask32_t m = VCMP(v, vbest); \
          vbest = mag_vf32_blend(m, v, vbest); \
          vidx = mag_vf32_blend(m, mag_vf32_add(vlane, mag_vf32_splat((float)ri)), vidx); \
        } \
        float bests[MAG_VF32_LANES]; \
        float idxs[MAG_VF32_LANES]; \
        mag_vf32_storeu(bests, vbest); \
        mag_vf32_storeu(idxs, vidx); \
        for (int64_t l=0; l < MAG_VF32_LANES; ++l) { \
          int64_t li = (int64_t)idxs[l]; \
          if (!set || SCMP(bests[l], best) || (bests[l] == best && li < best_idx)) { best = bests[l]; best_idx = li; set = true; } \
        } \
        for (; ri < red_prod; ++ri) { \
          float xv = CVT(p[ri]); \
          if (SCMP(xv, best)) { best = xv; best_idx = ri; } \
        } \
      } else if (row_len >= 2 && oi % row_len == 0 && oi + row_len <= ob) { \
        for (int64_t j0=0; j0 < row_len; j0 += MAG_REDUCE_ROW_BLOCK) { \
          int64_t nb = mag_vmin((int64_t)MAG_REDUCE_ROW_BLOCK, row_len-j0); \
          float bests[MAG_REDUCE_ROW_BLOCK]; \
          int64_t idxs[MAG_REDUCE_ROW_BLOCK]; \
          for (ri=0; ri < red_prod; ++ri) { \
            int64_t tmp = ri; \
            int64_t roff0 = base + j0; \
            for (int64_t k=plan->red_rank - 1; k >= 0; --k) { \
              int64_t sz = plan->red_sizes[k]; \
              int64_t idx = tmp % sz; \
              tmp /= sz; \
              roff0 += idx*plan->red_strides[k]; \
            } \
            const T *p = bx + roff0; \
            mag_bnd_chk(p + nb - 1, x->storage->base, x->storage->size); \
            for (int64_t j=0; j < nb; ++j) { \
              float xv = CVT(p[j]); \
              if (ri == 0 || SCMP(xv, bests[j])) { bests[j] = xv; idxs[j] = ri; } \
            } \
          } \
          for (int64_t j=0; j < nb; ++j) br[oi + j0 + j] = idxs[j]; \
        } \
        oi += row_len - 1; \
        continue; \
      } else { \
        for (ri=0; ri < red_prod; ++ri) { \
          int64_t tmp = ri; \
          int64_t roff = base; \
          if (mag_contig) roff += ri; \
          else for (int64_t k=plan->red_rank - 1; k >= 0; --k) { \
            int64_t sz = plan->red_sizes[k]; \
            int64_t idx = tmp % sz; \
            tmp /= sz; \
            roff += idx*plan->red_strides[k]; \
          } \
          mag_bnd_chk(bx+roff, x->storage->base, x->storage->size); \
          float xv = CVT(bx[roff]); \
          if (!set || SCMP(xv, best)) { best = xv; best_idx = ri; set = true; } \
        } \
      } \
      br[oi] = best_idx; \
    } \
    return MAG_OK; \
  }
#define mag_scmp_gt(a, b) ((a) > (b))
#define mag_scmp_lt(a, b) ((a) < (b))
mag_cpu_impl_argext_hfp(float, float32, argmax, mag_cvt_nop, mag_vf32_loadu, mag_vf32_cmpgt, mag_scmp_gt)
mag_cpu_impl_argext_hfp(float, float32, argmin, mag_cvt_nop, mag_vf32_loadu, mag_vf32_cmplt, mag_scmp_lt)
mag_cpu_impl_argext_hfp(mag_float16_t, float16, argmax, mag_float16_to_float32, mag_vf32_loadu_f16, mag_vf32_cmpgt, mag_scmp_gt)
mag_cpu_impl_argext_hfp(mag_float16_t, float16, argmin, mag_float16_to_float32, mag_vf32_loadu_f16, mag_vf32_cmplt, mag_scmp_lt)
mag_cpu_impl_argext_hfp(mag_bfloat16_t, bfloat16, argmax, mag_bfloat16_to_float32, mag_vf32_loadu_bf16, mag_vf32_cmpgt, mag_scmp_gt)
mag_cpu_impl_argext_hfp(mag_bfloat16_t, bfloat16, argmin, mag_bfloat16_to_float32, mag_vf32_loadu_bf16, mag_vf32_cmplt, mag_scmp_lt)
mag_cpu_impl_argext_hfp(mag_float8_e4m3fn_t, float8_e4m3fn, argmax, mag_float8_e4m3fn_to_float32, mag_vf32_loadu_float8_e4m3fn, mag_vf32_cmpgt, mag_scmp_gt)
mag_cpu_impl_argext_hfp(mag_float8_e4m3fn_t, float8_e4m3fn, argmin, mag_float8_e4m3fn_to_float32, mag_vf32_loadu_float8_e4m3fn, mag_vf32_cmplt, mag_scmp_lt)








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

mag_cpu_impl_reduce_axes(mag_float8_e4m3fn_t, mag_float8_e4m3fn_t, float8_e4m3fn, sum, float, 0.0f, acc += mag_float8_e4m3fn_to_float32(bx[roff]);, *o = mag_float32_to_float8_e4m3fn(acc); )
mag_cpu_impl_reduce_axes(mag_float8_e4m3fn_t, mag_float8_e4m3fn_t, float8_e4m3fn, mean, float, 0.0f, acc += mag_float8_e4m3fn_to_float32(bx[roff]);, acc /= (float)red_prod; *o = mag_float32_to_float8_e4m3fn(acc); )
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
    int64_t chunk = (numel+tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_vmin(oa+chunk, numel); \
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
        for (int64_t k=plan->red_rank-1; k >= 0; --k) { \
          int64_t sz = plan->red_sizes[k]; \
          int64_t idx = tmp % sz; \
          tmp /= sz; \
          roff += idx*plan->red_strides[k]; \
        } \
        mag_bnd_chk(bx+roff, x->storage->base, x->storage->size); \
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
