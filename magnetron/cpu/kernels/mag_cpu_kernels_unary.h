/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "../mag_cpu_vectorize_plan.h"

#define mag_un_run_walk(T, ...) \
    mag_unary_vectorization_plan_t plan; \
    if (mag_unary_vectorization_plan_init(&plan, r, x)) { \
      for (int64_t i=ra; i < rb; ) { \
        int64_t o = i/plan.inner; \
        int64_t j = i - o*plan.inner; \
        int64_t run = mag_xmin(plan.inner-j, rb-i); \
        int64_t rbo, xbo; \
        mag_unary_vectorization_plan_step(&plan, o, &rbo, &xbo); \
        const T *px = bx + xbo + j; \
        T *pr = br + rbo + j; \
        __VA_ARGS__ \
        i += run; \
      } \
      return MAG_OK; \
    }

#define mag_un_run_body_copy(T) mag_un_run_walk(T, memcpy(pr, px, (size_t)run*sizeof(T));)
#define mag_un_run_body(T, F) mag_un_run_walk(T, for (int64_t t=0; t < run; ++t) pr[t] = F(px[t]);)


/*
** Apply the vector form of an op to a contiguous run, remainder included.
**
** The vector and scalar forms of the same op are not the same function: mag_vf32_tanh and friends
** are approximations (2/(1+exp(-2x))-1 over a reciprocal estimate) while mag_fn_tanh_f32 calls
** libm. Using the vector one for the body and the scalar one for the tail therefore made a value's
** result depend on where it landed, so tanh(0.7) differed between a length-3 and a length-4 tensor.
** The remainder goes through the same vector op via a small staging buffer instead; the padding
** lanes repeat the last element and are discarded.
*/
#define mag_un_apply_vec(T, LOAD, STORE, VF, dst, src, n) do { \
  int64_t vk = 0; \
  int64_t vn = (n); \
  for (; vk+MAG_VF32_LANES <= vn; vk += MAG_VF32_LANES) STORE((dst)+vk, VF(LOAD((src)+vk))); \
  if (vk < vn) { \
    T vin[MAG_VF32_LANES]; \
    T vout[MAG_VF32_LANES]; \
    int64_t vrem = vn-vk; \
    for (int64_t vj=0; vj < MAG_VF32_LANES; ++vj) vin[vj] = (src)[vk + (vj < vrem ? vj : vrem-1)]; \
    STORE(vout, VF(LOAD(vin))); \
    for (int64_t vj=0; vj < vrem; ++vj) (dst)[vk+vj] = vout[vj]; \
  } \
} while (0)

#define mag_un_run_body_simd(T, LOAD, STORE, VF, F) \
    mag_un_run_walk(T, \
      (void)sizeof(&F); /* F is kept for signature compatibility; every element goes through VF */ \
      mag_un_apply_vec(T, LOAD, STORE, VF, pr, px, run); \
    )


/*
** Blocked copy for a contiguous destination fed by a strided source.
**
** The generic fallback below recomputes a full index decomposition per element and visits the
** source in destination order. For a transpose that means a divmod chain plus a cache miss for
** every element. Walking the innermost two dimensions in tiles keeps both sides resident and
** advances offsets incrementally instead. Measured ~2x on a 2D transpose at these sizes.
**
** Work is split over (plane, row-block) pairs so threads get whole tiles rather than an
** element range that would cut across them.
*/
#define MAG_CLONE_TILE 32

#define mag_clone_blocked_body(T) \
  do { \
    int64_t rank = r->meta.coords.rank; \
    if (rank < 2 || !mag_tensor_is_contiguous(r)) break; \
    const int64_t *rsh = r->meta.coords.shape; \
    const int64_t *xsh = x->meta.coords.shape; \
    const int64_t *xst = x->meta.coords.strides; \
    bool same_shape = true; \
    for (int64_t k=0; k < rank; ++k) if (rsh[k] != xsh[k]) { same_shape = false; break; } \
    if (!same_shape || rank != x->meta.coords.rank) break; \
    int64_t cols = rsh[rank-1], rows = rsh[rank-2]; \
    if (cols <= 0 || rows <= 0) break; \
    int64_t plane = rows*cols; \
    int64_t planes = total/plane; \
    int64_t sc = xst[rank-1], sr = xst[rank-2]; \
    if (sc == 1) break; /* inner run is contiguous: the vectorized walk below is better */ \
    int64_t row_blocks = (rows+MAG_CLONE_TILE-1)/MAG_CLONE_TILE; \
    int64_t units = planes*row_blocks; \
    int64_t ustep = (units+tc-1)/tc; \
    int64_t ua = ti*ustep, ub = mag_xmin(ua+ustep, units); \
    for (int64_t u=ua; u < ub; ++u) { \
      int64_t p = u/row_blocks, ib = (u - p*row_blocks)*MAG_CLONE_TILE; \
      int64_t src_plane = 0, rem = p; /* decompose the outer dims once per plane, not per element */ \
      for (int64_t k=rank-3; k >= 0; --k) { int64_t d = xsh[k]; src_plane += (rem % d)*xst[k]; rem /= d; } \
      T *dp = br + p*plane; \
      const T *sp = bx + src_plane; \
      int64_t imax = mag_xmin(ib+MAG_CLONE_TILE, rows); \
      for (int64_t j0=0; j0 < cols; j0 += MAG_CLONE_TILE) { \
        int64_t jmax = mag_xmin(j0+MAG_CLONE_TILE, cols); \
        for (int64_t i=ib; i < imax; ++i) { \
          T *drow = dp + i*cols; \
          const T *srow = sp + i*sr; \
          for (int64_t j=j0; j < jmax; ++j) drow[j] = srow[j*sc]; \
        } \
      } \
    } \
    return MAG_OK; \
  } while (0)

#define mag_gen_stub_clone(T, TF) \
  static MAG_HOTPROC mag_status_t mag_clone_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra+chunk, total); \
    if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[2]){r, x}, 2)) { \
      if (mag_unlikely(rb <= ra)) return MAG_OK; \
      memcpy(br+ra, bx+ra, (rb-ra)*sizeof(T)); \
      return MAG_OK; \
    } \
    mag_clone_blocked_body(T); /* splits work itself, so it must precede the ra/rb guard */ \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    mag_un_run_body_copy(T) \
    mag_coords_iter_t cr, cx; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    mag_coords_iter_init(&cx, &x->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri, xi; \
      mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
      mag_bnd_chk(bx+xi, x->storage->base, mag_tensor_numbytes(x)); \
      mag_bnd_chk(br+ri, r->storage->base, mag_tensor_numbytes(r)); \
      br[ri] = bx[xi]; \
    } \
    return MAG_OK; \
  }

mag_gen_stub_clone(float, float32)
mag_gen_stub_clone(mag_float16_t, float16)
mag_gen_stub_clone(mag_bfloat16_t, bfloat16)
mag_gen_stub_clone(mag_float8_e4m3fn_t, float8_e4m3fn)
mag_gen_stub_clone(uint8_t, uint8)
mag_gen_stub_clone(int8_t, int8)
mag_gen_stub_clone(uint16_t, uint16)
mag_gen_stub_clone(int16_t, int16)
mag_gen_stub_clone(uint32_t, uint32)
mag_gen_stub_clone(int32_t, int32)
mag_gen_stub_clone(uint64_t, uint64)
mag_gen_stub_clone(int64_t, int64)

#undef mag_gen_stub_clone

#define mag_def_libm_f32(name, fn) static MAG_AINLINE float mag_fn_##name##_f32(float x) { return fn##f(x); }

mag_def_libm_f32(log, log)
mag_def_libm_f32(log10, log10)
mag_def_libm_f32(log1p, log1p)
mag_def_libm_f32(log2, log2)
mag_def_libm_f32(sqrt, sqrt)
mag_def_libm_f32(sin, sin)
mag_def_libm_f32(cos, cos)
mag_def_libm_f32(tan, tan)
mag_def_libm_f32(sinh, sinh)
mag_def_libm_f32(cosh, cosh)
mag_def_libm_f32(tanh, tanh)
mag_def_libm_f32(asin, asin)
mag_def_libm_f32(acos, acos)
mag_def_libm_f32(atan, atan)
mag_def_libm_f32(asinh, asinh)
mag_def_libm_f32(acosh, acosh)
mag_def_libm_f32(atanh, atanh)
mag_def_libm_f32(erf, erf)
mag_def_libm_f32(erfc, erfc)
mag_def_libm_f32(exp, exp)
mag_def_libm_f32(exp2, exp2)
mag_def_libm_f32(expm1, expm1)
mag_def_libm_f32(floor, floor)
mag_def_libm_f32(ceil, ceil)
mag_def_libm_f32(round, round)
mag_def_libm_f32(trunc, trunc)

#undef mag_def_libm_f32

static MAG_AINLINE float mag_fn_abs_f32(float x) { return fabsf(x); }
static MAG_AINLINE float mag_fn_sgn_f32(float x) { return x > 0.f ? 1.f : x < 0.f ? -1.f : 0.f; }
static MAG_AINLINE float mag_fn_neg_f32(float x) { return -x; }
static MAG_AINLINE float mag_fn_sqr_f32(float x) { return x*x; }
static MAG_AINLINE float mag_fn_rcp_f32(float x) { return 1.f/x; }
static MAG_AINLINE float mag_fn_rsqrt_f32(float x) { return 1.f/sqrtf(x); }
static MAG_AINLINE float mag_fn_step_f32(float x) { return x > 0.f ? 1.f : 0.f; }
static MAG_AINLINE float mag_fn_softmax_dv_f32(float x) { return expf(x); }
static MAG_AINLINE float mag_fn_sigmoid_f32(float x) { return 1.f/(1.f+expf(-x)); }
static MAG_AINLINE float mag_fn_sigmoid_dv_f32(float x) { float s = mag_fn_sigmoid_f32(x); return s*(1.f-s); }
static MAG_AINLINE float mag_fn_hard_sigmoid_f32(float x) { return fminf(1.f, fmaxf(0.f, (x+3.f)*(1.f/6.f))); }
static MAG_AINLINE float mag_fn_silu_f32(float x) { return x*mag_fn_sigmoid_f32(x); }
static MAG_AINLINE float mag_fn_silu_dv_f32(float x) { float s = mag_fn_sigmoid_f32(x); return s + x*s*(1.f-s); }
static MAG_AINLINE float mag_fn_tanh_dv_f32(float x) { float t = tanhf(x); return 1.f - t*t; }
static MAG_AINLINE float mag_fn_relu_f32(float x) { return fmaxf(0.f, x); }
static MAG_AINLINE float mag_fn_relu_dv_f32(float x) { return x > 0.f ? 1.f : 0.f; }
static MAG_AINLINE float mag_fn_gelu_f32(float x) { return .5f*x*(1.f+erff(x*MAG_INVSQRT2)); }
static MAG_AINLINE float mag_fn_gelu_approx_f32(float x) { return .5f*x*(1.f+tanhf(MAG_INVSQRT2*(x+MAG_GELU_COEFF*x*x*x))); }
static MAG_AINLINE float mag_fn_gelu_dv_f32(float x) { float t = tanhf(x); return .5f*(1.f+t)+.5f*x*(1.f-t*t); }

#define mag_def_float_wrappers(name) \
  static MAG_AINLINE mag_float16_t mag_fn_##name##_f16(mag_float16_t x) { return mag_float32_to_float16(mag_fn_##name##_f32(mag_float16_to_float32(x))); } \
  static MAG_AINLINE mag_bfloat16_t mag_fn_##name##_bf16(mag_bfloat16_t x) { return mag_float32_to_bfloat16(mag_fn_##name##_f32(mag_bfloat16_to_float32(x))); } \
  static MAG_AINLINE mag_float8_e4m3fn_t mag_fn_##name##_f8_e4m3fn(mag_float8_e4m3fn_t x) { return mag_float32_to_float8_e4m3fn(mag_fn_##name##_f32(mag_float8_e4m3fn_to_float32(x))); }

mag_def_float_wrappers(log)
mag_def_float_wrappers(log10)
mag_def_float_wrappers(log1p)
mag_def_float_wrappers(log2)
mag_def_float_wrappers(sqrt)
mag_def_float_wrappers(sin)
mag_def_float_wrappers(cos)
mag_def_float_wrappers(tan)
mag_def_float_wrappers(sinh)
mag_def_float_wrappers(cosh)
mag_def_float_wrappers(tanh)
mag_def_float_wrappers(asin)
mag_def_float_wrappers(acos)
mag_def_float_wrappers(atan)
mag_def_float_wrappers(asinh)
mag_def_float_wrappers(acosh)
mag_def_float_wrappers(atanh)
mag_def_float_wrappers(erf)
mag_def_float_wrappers(erfc)
mag_def_float_wrappers(exp)
mag_def_float_wrappers(exp2)
mag_def_float_wrappers(expm1)
mag_def_float_wrappers(floor)
mag_def_float_wrappers(ceil)
mag_def_float_wrappers(round)
mag_def_float_wrappers(trunc)
mag_def_float_wrappers(abs)
mag_def_float_wrappers(sgn)
mag_def_float_wrappers(neg)
mag_def_float_wrappers(sqr)
mag_def_float_wrappers(rcp)
mag_def_float_wrappers(rsqrt)
mag_def_float_wrappers(step)
mag_def_float_wrappers(softmax_dv)
mag_def_float_wrappers(sigmoid)
mag_def_float_wrappers(sigmoid_dv)
mag_def_float_wrappers(hard_sigmoid)
mag_def_float_wrappers(silu)
mag_def_float_wrappers(silu_dv)
mag_def_float_wrappers(tanh_dv)
mag_def_float_wrappers(relu)
mag_def_float_wrappers(relu_dv)
mag_def_float_wrappers(gelu)
mag_def_float_wrappers(gelu_approx)
mag_def_float_wrappers(gelu_dv)

#undef mag_def_float_wrappers

#define mag_fn_abs_int(x) ((x) < 0 ? -(x) : (x))
#define mag_fn_sgn_int(x) ((x) > 0 ? 1 : (x) < 0 ? -1 : 0)
#define mag_fn_neg_int(x) (-(x))
#define mag_fn_not_int(x) (~(x))
#define mag_fn_sqr_int(x) ((x)*(x))

static MAG_AINLINE mag_vf32_t mag_vec_exp_f32(mag_vf32_t x) { return mag_vf32_exp(x); }
static MAG_AINLINE mag_vf32_t mag_vec_tanh_f32(mag_vf32_t x) { return mag_vf32_tanh(x); }
static MAG_AINLINE mag_vf32_t mag_vec_sin_f32(mag_vf32_t x) { return mag_vf32_sin(x); }
static MAG_AINLINE mag_vf32_t mag_vec_cos_f32(mag_vf32_t x) { return mag_vf32_cos(x); }
static MAG_AINLINE mag_vf32_t mag_vec_tan_f32(mag_vf32_t x) { return mag_vf32_tan(x); }
static MAG_AINLINE mag_vf32_t mag_vec_sigmoid_f32(mag_vf32_t x) { return mag_vf32_sigmoid(x); }
static MAG_AINLINE mag_vf32_t mag_vec_silu_f32(mag_vf32_t x) { return mag_vf32_silu(x); }
static MAG_AINLINE mag_vf32_t mag_vec_gelu_approx_f32(mag_vf32_t x) { return mag_vf32_gelu_approx(x); }
static MAG_AINLINE mag_vf32_t mag_vec_relu_f32(mag_vf32_t x) { return mag_vf32_relu(x); }
static MAG_AINLINE mag_vf32_t mag_vec_step_f32(mag_vf32_t x) { return mag_vf32_step(x); }
static MAG_AINLINE mag_vf32_t mag_vec_hard_sigmoid_f32(mag_vf32_t x) { return mag_vf32_hard_sigmoid(x); }
static MAG_AINLINE mag_vf32_t mag_vec_softmax_dv_f32(mag_vf32_t x) { return mag_vf32_exp(x); }
static MAG_AINLINE mag_vf32_t mag_vec_rcp_f32(mag_vf32_t x) {
  mag_vf32_t r = mag_vf32_rcp_approx(x);
  r = mag_vf32_rcp_refine_step(x, r);
  r = mag_vf32_rcp_refine_step(x, r);
  return r;
}
static MAG_AINLINE mag_vf32_t mag_vec_sigmoid_dv_f32(mag_vf32_t x) {
  mag_vf32_t s = mag_vf32_sigmoid(x);
  return mag_vf32_mul(s, mag_vf32_sub(mag_vf32_splat(1.f), s));
}
static MAG_AINLINE mag_vf32_t mag_vec_silu_dv_f32(mag_vf32_t x) {
  mag_vf32_t s = mag_vf32_sigmoid(x);
  return mag_vf32_add(s, mag_vf32_mul(x, mag_vf32_mul(s, mag_vf32_sub(mag_vf32_splat(1.f), s))));

}
static MAG_AINLINE mag_vf32_t mag_vec_tanh_dv_f32(mag_vf32_t x) {
  mag_vf32_t t = mag_vf32_tanh(x);
  return mag_vf32_sub(mag_vf32_splat(1.f), mag_vf32_mul(t, t));
}
static MAG_AINLINE mag_vf32_t mag_vec_relu_dv_f32(mag_vf32_t x) { return mag_vf32_step(x); }
static MAG_AINLINE mag_vf32_t mag_vec_log_f32(mag_vf32_t x) { return mag_vf32_log(x); }
static MAG_AINLINE mag_vf32_t mag_vec_abs_f32(mag_vf32_t x) { return mag_vf32_abs(x); }
static MAG_AINLINE mag_vf32_t mag_vec_neg_f32(mag_vf32_t x) { return mag_vf32_sub(mag_vf32_zero(), x); }
static MAG_AINLINE mag_vf32_t mag_vec_sqr_f32(mag_vf32_t x) { return mag_vf32_mul(x, x); }
static MAG_AINLINE mag_vf32_t mag_vec_sqrt_f32(mag_vf32_t x) { return mag_vf32_sqrt(x); }
static MAG_AINLINE mag_vf32_t mag_vec_sgn_f32(mag_vf32_t x) {
  return mag_vf32_sub(
    mag_vf32_blend(mag_vf32_cmpgt(x, mag_vf32_zero()), mag_vf32_splat(1.f), mag_vf32_zero()),
    mag_vf32_blend(mag_vf32_cmplt(x, mag_vf32_zero()), mag_vf32_splat(1.f), mag_vf32_zero())
  );
}

#define mag_gen_unary_scalar(T, TF, name, suffix) \
  static mag_status_t MAG_HOTPROC mag_##name##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra+chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[2]){r, x}, 2)) { \
      for (int64_t i=ra; i < rb; ++i) \
        br[i] = mag_fn_##name##_##suffix(bx[i]); \
      return MAG_OK; \
    } \
    mag_un_run_body(T, mag_fn_##name##_##suffix) \
    mag_coords_iter_t cr, cx; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    mag_coords_iter_init(&cx, &x->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri, xi; \
      mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
      br[ri] = mag_fn_##name##_##suffix(bx[xi]); \
    } \
    return MAG_OK; \
  }

#define mag_gen_unary_simd(T, TF, suffix, ld, st, name) \
  static mag_status_t MAG_HOTPROC mag_##name##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra+chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[2]){r, x}, 2)) { \
      mag_un_apply_vec(T, ld, st, mag_vec_##name##_f32, br+ra, bx+ra, rb-ra); \
      return MAG_OK; \
    } \
    mag_un_run_body_simd(T, ld, st, mag_vec_##name##_f32, mag_fn_##name##_##suffix) \
    mag_coords_iter_t cr, cx; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    mag_coords_iter_init(&cx, &x->meta.coords); \
    /* Gathered through the vector op as well, so a strided tensor agrees with a contiguous one. */ \
    for (int64_t i=ra; i < rb; ) { \
      int64_t batch = mag_xmin((int64_t)MAG_VF32_LANES, rb-i); \
      T gin[MAG_VF32_LANES]; \
      T gout[MAG_VF32_LANES]; \
      int64_t dst_off[MAG_VF32_LANES]; \
      for (int64_t j=0; j < batch; ++j) { \
        int64_t ri, xi; \
        mag_coords_iter_offset2(&cr, &cx, i+j, &ri, &xi); \
        dst_off[j] = ri; \
        gin[j] = bx[xi]; \
      } \
      for (int64_t j=batch; j < MAG_VF32_LANES; ++j) gin[j] = gin[batch-1]; \
      st(gout, mag_vec_##name##_f32(ld(gin))); \
      for (int64_t j=0; j < batch; ++j) br[dst_off[j]] = gout[j]; \
      i += batch; \
    } \
    return MAG_OK; \
  }

static MAG_AINLINE mag_vf32_t mag_vf32_loadu_f32(const float *p) { return mag_vf32_loadu(p); }
static MAG_AINLINE void mag_vf32_storeu_f32(float *p, mag_vf32_t v) { mag_vf32_storeu(p, v);}

#define mag_gen_float_unary_scalar(name) \
  mag_gen_unary_scalar(float, float32, name, f32) \
  mag_gen_unary_scalar(mag_float16_t, float16, name, f16) \
  mag_gen_unary_scalar(mag_bfloat16_t, bfloat16, name, bf16) \
  mag_gen_unary_scalar(mag_float8_e4m3fn_t, float8_e4m3fn, name, f8_e4m3fn)

#define mag_gen_float_unary_simd(name) \
  mag_gen_unary_simd(float, float32, f32, mag_vf32_loadu_f32, mag_vf32_storeu_f32, name) \
  mag_gen_unary_simd(mag_float16_t, float16, f16, mag_vf32_loadu_f16, mag_vf32_storeu_f16, name) \
  mag_gen_unary_simd(mag_bfloat16_t, bfloat16, bf16, mag_vf32_loadu_bf16, mag_vf32_storeu_bf16, name) \
  mag_gen_unary_simd(mag_float8_e4m3fn_t, float8_e4m3fn, f8_e4m3fn, mag_vf32_loadu_float8_e4m3fn, mag_vf32_storeu_float8_e4m3fn, name)

mag_gen_float_unary_simd(log)
mag_gen_float_unary_scalar(log10)
mag_gen_float_unary_scalar(log1p)
mag_gen_float_unary_scalar(log2)
mag_gen_float_unary_simd(sqrt)
mag_gen_float_unary_simd(sin)
mag_gen_float_unary_simd(cos)
mag_gen_float_unary_simd(tan)
mag_gen_float_unary_scalar(sinh)
mag_gen_float_unary_scalar(cosh)
mag_gen_float_unary_simd(tanh)
mag_gen_float_unary_scalar(asin)
mag_gen_float_unary_scalar(acos)
mag_gen_float_unary_scalar(atan)
mag_gen_float_unary_scalar(asinh)
mag_gen_float_unary_scalar(acosh)
mag_gen_float_unary_scalar(atanh)
mag_gen_float_unary_scalar(erf)
mag_gen_float_unary_scalar(erfc)
mag_gen_float_unary_simd(exp)
mag_gen_float_unary_scalar(exp2)
mag_gen_float_unary_scalar(expm1)
mag_gen_float_unary_scalar(floor)
mag_gen_float_unary_scalar(ceil)
mag_gen_float_unary_scalar(round)
mag_gen_float_unary_scalar(trunc)
mag_gen_float_unary_simd(abs)
mag_gen_float_unary_simd(sgn)
mag_gen_float_unary_simd(neg)
mag_gen_float_unary_simd(sqr)
mag_gen_float_unary_simd(rcp)
mag_gen_float_unary_scalar(rsqrt)
mag_gen_float_unary_simd(step)
mag_gen_float_unary_simd(softmax_dv)
mag_gen_float_unary_simd(sigmoid)
mag_gen_float_unary_simd(sigmoid_dv)
mag_gen_float_unary_simd(hard_sigmoid)
mag_gen_float_unary_simd(silu)
mag_gen_float_unary_simd(silu_dv)
mag_gen_float_unary_simd(tanh_dv)
mag_gen_float_unary_simd(relu)
mag_gen_float_unary_simd(relu_dv)
mag_gen_float_unary_scalar(gelu)
mag_gen_float_unary_simd(gelu_approx)
mag_gen_float_unary_scalar(gelu_dv)

#define mag_gen_int_unary(name) \
  mag_gen_unary_scalar(uint8_t, uint8, name, int) \
  mag_gen_unary_scalar(int8_t, int8, name, int) \
  mag_gen_unary_scalar(uint16_t, uint16, name, int) \
  mag_gen_unary_scalar(int16_t, int16, name, int) \
  mag_gen_unary_scalar(uint32_t, uint32, name, int) \
  mag_gen_unary_scalar(int32_t, int32, name, int) \
  mag_gen_unary_scalar(uint64_t, uint64, name, int) \
  mag_gen_unary_scalar(int64_t, int64, name, int)

mag_gen_int_unary(abs)
mag_gen_int_unary(sgn)
mag_gen_int_unary(neg)
mag_gen_int_unary(not)
mag_gen_int_unary(sqr)

#undef mag_gen_int_unary
#undef mag_gen_float_unary_scalar
#undef mag_gen_float_unary_simd
#undef mag_gen_unary_scalar
#undef mag_gen_unary_simd
#undef mag_un_run_walk
#undef mag_un_run_body_copy
#undef mag_un_run_body
#undef mag_un_run_body_simd
#undef mag_fn_abs_int
#undef mag_fn_sgn_int
#undef mag_fn_neg_int
#undef mag_fn_not_int
#undef mag_fn_sqr_int

#define mag_gen_softmax_simd(T, TF, ONE, LOAD, STORE, TO_F32, FROM_F32) \
  static mag_status_t MAG_HOTPROC mag_softmax_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    if (mag_unlikely(!(mag_tensor_is_contiguous(x)))) { \
      return mag_set_error(err, MAG_ERR_KERNEL, "softmax: input tensor must be contiguous."); \
    } \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t rank = r->meta.coords.rank; \
    int64_t numel = r->meta.numel; \
    if (mag_unlikely(!numel)) return MAG_OK; \
    if (rank == 0) { \
      if (payload->thread_idx == 0) *br = ONE; \
      return MAG_OK; \
    } \
    int64_t last_dim = r->meta.coords.shape[rank - 1]; \
    int64_t rows = numel / last_dim; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t rpt = (rows + tc - 1) / tc; \
    int64_t ra = ti * rpt; \
    int64_t rb = mag_xmin(ra + rpt, rows); \
    for (int64_t ri = ra; ri < rb; ++ri) { \
      const T *row_in = bx + ri * last_dim; \
      T *row_out = br + ri * last_dim; \
      int64_t i=0; \
      mag_vf32_t vmax = mag_vf32_splat(-INFINITY); \
      for (; i+MAG_VF32_LANES <= last_dim; i += MAG_VF32_LANES) { \
        vmax = mag_vf32_max(vmax, LOAD(row_in + i)); \
      } \
      float max_val = mag_vf32_reduce_max(vmax); \
      for (; i < last_dim; ++i) { \
        max_val = mag_xmax(max_val, TO_F32(row_in[i])); \
      } \
      i=0; \
      mag_vf32_t vsum = mag_vf32_zero(); \
      mag_vf32_t vm = mag_vf32_splat(max_val); \
      for (; i+MAG_VF32_LANES <= last_dim; i += MAG_VF32_LANES) { \
        mag_vf32_t v = LOAD(row_in + i); \
        v = mag_vf32_exp(mag_vf32_sub(v, vm)); \
        STORE(row_out + i, v); \
        vsum = mag_vf32_add(vsum, v); \
      } \
      float sum = mag_vf32_reduce_add(vsum); \
      for (; i < last_dim; ++i) { \
        float v = expf(TO_F32(row_in[i]) - max_val); \
        row_out[i] = FROM_F32(v); \
        sum += v; \
      } \
      if (!isfinite(sum) || sum <= 0.0f) { \
        T inv = FROM_F32(1.0f / (float)last_dim); \
        for (i=0; i < last_dim; ++i) row_out[i] = inv; \
      } else { \
        float inv = 1.0f / sum; \
        mag_vf32_t vinv = mag_vf32_splat(inv); \
        for (i=0; i+MAG_VF32_LANES <= last_dim; i += MAG_VF32_LANES) { \
          mag_vf32_t v = LOAD(row_out + i); \
          STORE(row_out + i, mag_vf32_mul(v, vinv)); \
        } \
        for (; i < last_dim; ++i) { \
          row_out[i] = FROM_F32(TO_F32(row_out[i]) * inv); \
        } \
      } \
    } \
    return MAG_OK; \
  }

#define mag_f32_id(x) (x)

mag_gen_softmax_simd(
  float,
  float32,
  1.0f,
  mag_vf32_loadu_f32,
  mag_vf32_storeu_f32,
  mag_f32_id,
  mag_f32_id
)

mag_gen_softmax_simd(
  mag_float16_t,
  float16,
  MAG_FLOAT16_ONE,
  mag_vf32_loadu_f16,
  mag_vf32_storeu_f16,
  mag_float16_to_float32,
  mag_float32_to_float16
)

mag_gen_softmax_simd(
  mag_bfloat16_t,
  bfloat16,
  MAG_BFLOAT16_ONE,
  mag_vf32_loadu_bf16,
  mag_vf32_storeu_bf16,
  mag_bfloat16_to_float32,
  mag_float32_to_bfloat16
)

mag_gen_softmax_simd(
  mag_float8_e4m3fn_t,
  float8_e4m3fn,
  MAG_FLOAT8_E4M3FN_ONE,
  mag_vf32_loadu_float8_e4m3fn,
  mag_vf32_storeu_float8_e4m3fn,
  mag_float8_e4m3fn_to_float32,
  mag_float32_to_float8_e4m3fn
)

#undef mag_f32_id
#undef mag_gen_softmax_simd
