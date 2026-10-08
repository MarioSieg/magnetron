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

typedef struct mag_conv_geom_t {
  int64_t N;
  int64_t cin;
  int64_t cout;
  int64_t groups;
  int64_t big[3];
  int64_t small[3];
  int64_t k[3];
  int64_t s[3];
  int64_t p[3];
  int64_t d[3];
} mag_conv_geom_t;

static void mag_conv_geom_init(mag_conv_geom_t *g, const mag_op_params_t *params, const int64_t *big, const int64_t *small, const int64_t *k) {
  int64_t spatial = params->conv.spatial;
  int64_t off = 3 - spatial;
  g->groups = params->conv.groups;
  for (int64_t i=0; i < 3; ++i) {
    g->big[i] = 1;
    g->small[i] = 1;
    g->k[i] = 1;
    g->s[i] = 1;
    g->p[i] = 0;
    g->d[i] = 1;
  }
  for (int64_t i=0; i < spatial; ++i) {
    g->big[off+i] = big[i];
    g->small[off+i] = small[i];
    g->k[off+i] = k[i];
    g->s[off+i] = params->conv.stride[i];
    g->p[off+i] = params->conv.padding[i];
    g->d[off+i] = params->conv.dilation[i];
  }
}

static MAG_AINLINE void mag_conv_small_range(int64_t big, int64_t small, int64_t s, int64_t p, int64_t kd, int64_t *lo, int64_t *hi) {
  int64_t a = p - kd;
  int64_t b = big + p - kd;
  *lo = a <= 0 ? 0 : (a + s - 1)/s;
  *hi = b <= 0 ? 0 : mag_vmin(small, (b + s - 1)/s);
}

#define mag_gen_stub_conv(T, TF, CVT, RCVT, LDV) \
  static mag_status_t MAG_HOTPROC mag_conv_direct_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *w = payload->cmd->in[1]; \
    const mag_tensor_t *b = payload->cmd->num_in > 2 ? payload->cmd->in[2] : NULL; \
    if (mag_unlikely(r->meta.numel == 0)) return MAG_OK; \
    mag_conv_geom_t g; \
    mag_conv_geom_init(&g, payload->cmd->params, x->meta.coords.shape+2, r->meta.coords.shape+2, w->meta.coords.shape+2); \
    g.N = x->meta.coords.shape[0]; \
    g.cin = x->meta.coords.shape[1]; \
    g.cout = r->meta.coords.shape[1]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *bw = (const T *)mag_tensor_data_ptr(w); \
    const T *bb = b ? (const T *)mag_tensor_data_ptr(b) : NULL; \
    int64_t cinG = g.cin/g.groups; \
    int64_t coutG = g.cout/g.groups; \
    int64_t Id = g.big[0], Ih = g.big[1], Iw = g.big[2]; \
    int64_t Od = g.small[0], Oh = g.small[1], Ow = g.small[2]; \
    int64_t Kd = g.k[0], Kh = g.k[1], Kw = g.k[2]; \
    int64_t rows = g.N*g.cout*Od*Oh; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (rows + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra + chunk, rows); \
    if (ra >= rb) return MAG_OK; \
    const int64_t LANES = MAG_VF32_LANES; \
    size_t mark = mag_scratch_arena_mark(&mag_tls_arena); \
    float *acc = mag_scratch_arena_alloc(&mag_tls_arena, (size_t)Ow*sizeof(float)); \
    if (mag_unlikely(!acc)) return mag_set_error(err, MAG_ERR_OOM, "conv: failed to allocate %" PRIi64 " f32 accumulators.", Ow); \
    for (int64_t row=ra; row < rb; ++row) { \
      int64_t tmp = row; \
      int64_t oh = tmp % Oh; tmp /= Oh; \
      int64_t od = tmp % Od; tmp /= Od; \
      int64_t co = tmp % g.cout; tmp /= g.cout; \
      int64_t n = tmp; \
      int64_t grp = co/coutG; \
      float bias = bb ? CVT(bb[co]) : .0f; \
      for (int64_t ow=0; ow < Ow; ++ow) acc[ow] = bias; \
      for (int64_t cig=0; cig < cinG; ++cig) { \
        int64_t ci = grp*cinG + cig; \
        const T *xplane = bx + (n*g.cin + ci)*Id*Ih*Iw; \
        const T *wplane = bw + (co*cinG + cig)*Kd*Kh*Kw; \
        for (int64_t kd=0; kd < Kd; ++kd) { \
          int64_t id = od*g.s[0] - g.p[0] + kd*g.d[0]; \
          if (id < 0 || id >= Id) continue; \
          for (int64_t kh=0; kh < Kh; ++kh) { \
            int64_t ih = oh*g.s[1] - g.p[1] + kh*g.d[1]; \
            if (ih < 0 || ih >= Ih) continue; \
            const T *xrow = xplane + (id*Ih + ih)*Iw; \
            const T *wrow = wplane + (kd*Kh + kh)*Kw; \
            for (int64_t kw=0; kw < Kw; ++kw) { \
              float wv = CVT(wrow[kw]); \
              int64_t lo, hi; \
              mag_conv_small_range(Iw, Ow, g.s[2], g.p[2], kw*g.d[2], &lo, &hi); \
              if (lo >= hi) continue; \
              if (g.s[2] == 1) { \
                const T *xp = xrow + lo - g.p[2] + kw*g.d[2]; \
                mag_vf32_t vw = mag_vf32_splat(wv); \
                int64_t ow = lo; \
                for (; ow + LANES <= hi; ow += LANES) mag_vf32_storeu(acc+ow, mag_vf32_fmadd(vw, LDV(xp + ow - lo), mag_vf32_loadu(acc+ow))); \
                for (; ow < hi; ++ow) acc[ow] += wv*CVT(xp[ow-lo]); \
              } else { \
                for (int64_t ow=lo; ow < hi; ++ow) acc[ow] += wv*CVT(xrow[ow*g.s[2] - g.p[2] + kw*g.d[2]]); \
              } \
            } \
          } \
        } \
      } \
      T *rrow = br + row*Ow; \
      for (int64_t ow=0; ow < Ow; ++ow) rrow[ow] = RCVT(acc[ow]); \
    } \
    mag_scratch_arena_reset(&mag_tls_arena, mark); \
    return MAG_OK; \
  }

mag_gen_stub_conv(float, float32, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu)
mag_gen_stub_conv(mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16)
mag_gen_stub_conv(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16)
mag_gen_stub_conv(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn)

#undef mag_gen_stub_conv

#define mag_gen_stub_conv_transpose(T, TF, CVT, RCVT, LDV) \
  static mag_status_t MAG_HOTPROC mag_conv_transpose_direct_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *w = payload->cmd->in[1]; \
    const mag_tensor_t *b = payload->cmd->num_in > 2 ? payload->cmd->in[2] : NULL; \
    if (mag_unlikely(r->meta.numel == 0)) return MAG_OK; \
    mag_conv_geom_t g; \
    mag_conv_geom_init(&g, payload->cmd->params, r->meta.coords.shape+2, x->meta.coords.shape+2, w->meta.coords.shape+2); \
    g.N = x->meta.coords.shape[0]; \
    g.cin = x->meta.coords.shape[1]; \
    g.cout = r->meta.coords.shape[1]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *bw = (const T *)mag_tensor_data_ptr(w); \
    const T *bb = b ? (const T *)mag_tensor_data_ptr(b) : NULL; \
    int64_t cinG = g.cin/g.groups; \
    int64_t coutG = g.cout/g.groups; \
    int64_t Id = g.small[0], Ih = g.small[1], Iw = g.small[2]; \
    int64_t Od = g.big[0], Oh = g.big[1], Ow = g.big[2]; \
    int64_t Kd = g.k[0], Kh = g.k[1], Kw = g.k[2]; \
    int64_t planes = g.N*g.cout; \
    int64_t plane_numel = Od*Oh*Ow; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (planes + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra + chunk, planes); \
    if (ra >= rb) return MAG_OK; \
    const int64_t LANES = MAG_VF32_LANES; \
    size_t mark = mag_scratch_arena_mark(&mag_tls_arena); \
    float *acc = mag_scratch_arena_alloc(&mag_tls_arena, (size_t)plane_numel*sizeof(float)); \
    if (mag_unlikely(!acc)) return mag_set_error(err, MAG_ERR_OOM, "conv_transpose: failed to allocate %" PRIi64 " f32 accumulators.", plane_numel); \
    for (int64_t plane=ra; plane < rb; ++plane) { \
      int64_t co = plane % g.cout; \
      int64_t n = plane / g.cout; \
      int64_t grp = co/coutG; \
      int64_t cog = co - grp*coutG; \
      float bias = bb ? CVT(bb[co]) : .0f; \
      for (int64_t i=0; i < plane_numel; ++i) acc[i] = bias; \
      for (int64_t cig=0; cig < cinG; ++cig) { \
        int64_t ci = grp*cinG + cig; \
        const T *xplane = bx + (n*g.cin + ci)*Id*Ih*Iw; \
        const T *wplane = bw + (ci*coutG + cog)*Kd*Kh*Kw; \
        for (int64_t id=0; id < Id; ++id) { \
          for (int64_t kd=0; kd < Kd; ++kd) { \
            int64_t od = id*g.s[0] - g.p[0] + kd*g.d[0]; \
            if (od < 0 || od >= Od) continue; \
            for (int64_t ih=0; ih < Ih; ++ih) { \
              const T *xrow = xplane + (id*Ih + ih)*Iw; \
              for (int64_t kh=0; kh < Kh; ++kh) { \
                int64_t oh = ih*g.s[1] - g.p[1] + kh*g.d[1]; \
                if (oh < 0 || oh >= Oh) continue; \
                float *arow = acc + (od*Oh + oh)*Ow; \
                const T *wrow = wplane + (kd*Kh + kh)*Kw; \
                for (int64_t kw=0; kw < Kw; ++kw) { \
                  float wv = CVT(wrow[kw]); \
                  int64_t lo, hi; \
                  mag_conv_small_range(Ow, Iw, g.s[2], g.p[2], kw*g.d[2], &lo, &hi); \
                  if (lo >= hi) continue; \
                  if (g.s[2] == 1) { \
                    float *ap = arow + lo - g.p[2] + kw*g.d[2]; \
                    mag_vf32_t vw = mag_vf32_splat(wv); \
                    int64_t iw = lo; \
                    for (; iw + LANES <= hi; iw += LANES) mag_vf32_storeu(ap + iw - lo, mag_vf32_fmadd(vw, LDV(xrow+iw), mag_vf32_loadu(ap + iw - lo))); \
                    for (; iw < hi; ++iw) ap[iw-lo] += wv*CVT(xrow[iw]); \
                  } else { \
                    for (int64_t iw=lo; iw < hi; ++iw) arow[iw*g.s[2] - g.p[2] + kw*g.d[2]] += wv*CVT(xrow[iw]); \
                  } \
                } \
              } \
            } \
          } \
        } \
      } \
      T *rplane = br + plane*plane_numel; \
      for (int64_t i=0; i < plane_numel; ++i) rplane[i] = RCVT(acc[i]); \
    } \
    mag_scratch_arena_reset(&mag_tls_arena, mark); \
    return MAG_OK; \
  }

mag_gen_stub_conv_transpose(float, float32, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu)
mag_gen_stub_conv_transpose(mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16)
mag_gen_stub_conv_transpose(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16)
mag_gen_stub_conv_transpose(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn)

#undef mag_gen_stub_conv_transpose

#define mag_gen_stub_conv_wgrad(T, TF, CVT, RCVT, LDV) \
  static mag_status_t MAG_HOTPROC mag_conv_wgrad_direct_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *a = payload->cmd->in[0]; \
    const mag_tensor_t *b = payload->cmd->in[1]; \
    if (mag_unlikely(r->meta.numel == 0)) return MAG_OK; \
    mag_conv_geom_t g; \
    mag_conv_geom_init(&g, payload->cmd->params, a->meta.coords.shape+2, b->meta.coords.shape+2, r->meta.coords.shape+2); \
    g.N = a->meta.coords.shape[0]; \
    g.cin = a->meta.coords.shape[1]; \
    g.cout = b->meta.coords.shape[1]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *ba = (const T *)mag_tensor_data_ptr(a); \
    const T *bbp = (const T *)mag_tensor_data_ptr(b); \
    int64_t caG = g.cin/g.groups; \
    int64_t cbG = g.cout/g.groups; \
    int64_t Id = g.big[0], Ih = g.big[1], Iw = g.big[2]; \
    int64_t Od = g.small[0], Oh = g.small[1], Ow = g.small[2]; \
    int64_t Kd = g.k[0], Kh = g.k[1], Kw = g.k[2]; \
    int64_t items = g.cout*caG*Kd; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (items + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra + chunk, items); \
    if (ra >= rb) return MAG_OK; \
    const int64_t LANES = MAG_VF32_LANES; \
    size_t mark = mag_scratch_arena_mark(&mag_tls_arena); \
    float *acc = mag_scratch_arena_alloc(&mag_tls_arena, (size_t)Kh*Kw*sizeof(float)); \
    if (mag_unlikely(!acc)) return mag_set_error(err, MAG_ERR_OOM, "conv_wgrad: failed to allocate %" PRIi64 " f32 accumulators.", Kh*Kw); \
    for (int64_t item=ra; item < rb; ++item) { \
      int64_t tmp = item; \
      int64_t kd = tmp % Kd; tmp /= Kd; \
      int64_t cag = tmp % caG; tmp /= caG; \
      int64_t cb = tmp; \
      int64_t grp = cb/cbG; \
      int64_t ca = grp*caG + cag; \
      for (int64_t i=0; i < Kh*Kw; ++i) acc[i] = .0f; \
      for (int64_t n=0; n < g.N; ++n) { \
        const T *aplane = ba + (n*g.cin + ca)*Id*Ih*Iw; \
        const T *bplane = bbp + (n*g.cout + cb)*Od*Oh*Ow; \
        for (int64_t od=0; od < Od; ++od) { \
          int64_t id = od*g.s[0] - g.p[0] + kd*g.d[0]; \
          if (id < 0 || id >= Id) continue; \
          for (int64_t oh=0; oh < Oh; ++oh) { \
            const T *brow = bplane + (od*Oh + oh)*Ow; \
            for (int64_t kh=0; kh < Kh; ++kh) { \
              int64_t ih = oh*g.s[1] - g.p[1] + kh*g.d[1]; \
              if (ih < 0 || ih >= Ih) continue; \
              const T *arow = aplane + (id*Ih + ih)*Iw; \
              for (int64_t kw=0; kw < Kw; ++kw) { \
                int64_t lo, hi; \
                mag_conv_small_range(Iw, Ow, g.s[2], g.p[2], kw*g.d[2], &lo, &hi); \
                if (lo >= hi) continue; \
                float sum = .0f; \
                if (g.s[2] == 1) { \
                  const T *ap = arow + lo - g.p[2] + kw*g.d[2]; \
                  mag_vf32_t vacc = mag_vf32_zero(); \
                  int64_t ow = lo; \
                  for (; ow + LANES <= hi; ow += LANES) vacc = mag_vf32_fmadd(LDV(brow+ow), LDV(ap + ow - lo), vacc); \
                  sum = mag_vf32_reduce_add(vacc); \
                  for (; ow < hi; ++ow) sum += CVT(brow[ow])*CVT(ap[ow-lo]); \
                } else { \
                  for (int64_t ow=lo; ow < hi; ++ow) sum += CVT(brow[ow])*CVT(arow[ow*g.s[2] - g.p[2] + kw*g.d[2]]); \
                } \
                acc[kh*Kw + kw] += sum; \
              } \
            } \
          } \
        } \
      } \
      T *rrow = br + ((cb*caG + cag)*Kd + kd)*Kh*Kw; \
      for (int64_t i=0; i < Kh*Kw; ++i) rrow[i] = RCVT(acc[i]); \
    } \
    mag_scratch_arena_reset(&mag_tls_arena, mark); \
    return MAG_OK; \
  }

mag_gen_stub_conv_wgrad(float, float32, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu)
mag_gen_stub_conv_wgrad(mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16)
mag_gen_stub_conv_wgrad(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16)
mag_gen_stub_conv_wgrad(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn)

#undef mag_gen_stub_conv_wgrad

#ifndef MAG_CONV_GEMM_MIN_M
  #define MAG_CONV_GEMM_MIN_M 4
#endif
#ifndef MAG_CONV_GEMM_MIN_K
  #define MAG_CONV_GEMM_MIN_K 4
#endif

static MAG_AINLINE void mag_conv_kcol_decode(int64_t kk, int64_t Kd, int64_t Kh, int64_t Kw, int64_t *cig, int64_t *kd, int64_t *kh, int64_t *kw) {
  *kw = kk % Kw; kk /= Kw;
  *kh = kk % Kh; kk /= Kh;
  *kd = kk % Kd; kk /= Kd;
  *cig = kk;
}

static MAG_AINLINE void mag_conv_kcol_next(int64_t Kd, int64_t Kh, int64_t Kw, int64_t *cig, int64_t *kd, int64_t *kh, int64_t *kw) {
  if (++*kw < Kw) return;
  *kw = 0;
  if (++*kh < Kh) return;
  *kh = 0;
  if (++*kd < Kd) return;
  *kd = 0;
  ++*cig;
}

static MAG_AINLINE void mag_conv_clip_range(int64_t i0, int64_t s, int64_t len, int64_t lim, int64_t *lo, int64_t *hi) {
  int64_t l = i0 < 0 ? (-i0 + s-1)/s : 0;
  int64_t h = lim > i0 ? (lim - i0 + s-1)/s : 0;
  if (l > len) l = len;
  if (h > len) h = len;
  if (h < l) h = l;
  *lo = l;
  *hi = h;
}

static void mag_conv_col_table(const mag_conv_geom_t *g, const int64_t *sp, int64_t n0, int64_t nct, int32_t *o0, int32_t *o1, int32_t *o2) {
  int64_t W = sp[2], H = sp[1];
  for (int64_t c=0; c < nct; ++c) {
    int64_t n = n0 + c;
    int64_t w = n % W; n /= W;
    int64_t h = n % H; n /= H;
    o0[c] = (int32_t)(n*g->s[0] - g->p[0]);
    o1[c] = (int32_t)(h*g->s[1] - g->p[1]);
    o2[c] = (int32_t)(w*g->s[2] - g->p[2]);
  }
}

static MAG_AINLINE void mag_conv_ukernel_sweep(int64_t mbt, int64_t nct, int64_t kct, const float *ap, const float *bp, float *cc, int64_t ldc) {
  for (int64_t jr=0; jr < nct; jr += MAG_GEMM_NR) {
    const float *bpp = bp + jr/MAG_GEMM_NR*kct*MAG_GEMM_NR;
    for (int64_t ir=0; ir < mbt; ir += MAG_GEMM_MR)
      mag_gemm_ukernel(kct, ap + ir/MAG_GEMM_MR*kct*MAG_GEMM_MR, bpp, cc + ir*ldc + jr, ldc);
  }
}

typedef float (mag_conv_ld_t)(const void *p, int64_t i);
#define mag_conv_ld_impl(T, CVT) \
  static float mag_conv_ld_##T(const void *p, int64_t i) { return CVT(((const T *)p)[i]); }
mag_conv_ld_impl(float, mag_cvt_nop)
mag_conv_ld_impl(mag_float16_t, mag_float16_to_float32)
mag_conv_ld_impl(mag_bfloat16_t, mag_bfloat16_to_float32)
mag_conv_ld_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32)
#undef mag_conv_ld_impl

typedef void (mag_conv_im2col_b_t)(float *restrict bp, const void *pxg, const mag_conv_geom_t *g, int64_t pc, int64_t kct, int64_t nct, const int32_t *od0, const int32_t *oh0, const int32_t *ow0);
#define mag_conv_im2col_b_impl(T, CVT, LDV) \
  static MAG_HOTPROC void mag_conv_im2col_b_##T(float *restrict bp, const void *pxg, const mag_conv_geom_t *g, int64_t pc, int64_t kct, int64_t nct, const int32_t *od0, const int32_t *oh0, const int32_t *ow0) { \
    const T *xg = (const T *)pxg; \
    int64_t Id = g->big[0], Ih = g->big[1], Iw = g->big[2]; \
    int64_t Kd = g->k[0], Kh = g->k[1], Kw = g->k[2]; \
    int64_t Isp = Id*Ih*Iw; \
    int64_t s2 = g->s[2]; \
    for (int64_t p=0; p < nct; p += MAG_GEMM_NR) { \
      int64_t nn = mag_vmin((int64_t)MAG_GEMM_NR, nct-p); \
      const int32_t *cod = od0+p, *coh = oh0+p, *cow = ow0+p; \
      bool same_row = cod[0] == cod[nn-1] && coh[0] == coh[nn-1]; \
      int64_t cig, kd, kh, kw; \
      mag_conv_kcol_decode(pc, Kd, Kh, Kw, &cig, &kd, &kh, &kw); \
      for (int64_t k=0; k < kct; ++k) { \
        const T *xplane = xg + cig*Isp; \
        int64_t kdd = kd*g->d[0], khd = kh*g->d[1], kwd = kw*g->d[2]; \
        float *dst = bp + k*MAG_GEMM_NR; \
        if (same_row) { \
          int64_t id = cod[0]+kdd, ih = coh[0]+khd; \
          if ((uint64_t)id >= (uint64_t)Id || (uint64_t)ih >= (uint64_t)Ih) { \
            for (int64_t c=0; c < nn; ++c) dst[c] = 0.f; \
          } else { \
            const T *xrow = xplane + (id*Ih + ih)*Iw; \
            int64_t iw0 = cow[0]+kwd; \
            if (s2 == 1) { \
              if (iw0 >= 0 && iw0+nn <= Iw) { \
                const T *src = xrow + iw0; \
                if (nn == MAG_GEMM_NR) for (int64_t c=0; c < MAG_GEMM_NR; c += MAG_VF32_LANES) mag_vf32_storeu(dst+c, LDV(src+c)); \
                else for (int64_t c=0; c < nn; ++c) dst[c] = CVT(src[c]); \
              } else { \
                for (int64_t c=0; c < nn; ++c) { int64_t iw = iw0+c; dst[c] = (uint64_t)iw < (uint64_t)Iw ? CVT(xrow[iw]) : 0.f; } \
              } \
            } else { \
              int64_t lo, hi; \
              mag_conv_clip_range(iw0, s2, nn, Iw, &lo, &hi); \
              for (int64_t c=0; c < lo; ++c) dst[c] = 0.f; \
              for (int64_t c=lo; c < hi; ++c) dst[c] = CVT(xrow[iw0+c*s2]); \
              for (int64_t c=hi; c < nn; ++c) dst[c] = 0.f; \
            } \
          } \
        } else { \
          for (int64_t c=0; c < nn; ++c) { \
            int64_t id = cod[c]+kdd, ih = coh[c]+khd, iw = cow[c]+kwd; \
            bool in = (uint64_t)id < (uint64_t)Id && (uint64_t)ih < (uint64_t)Ih && (uint64_t)iw < (uint64_t)Iw; \
            dst[c] = in ? CVT(xplane[(id*Ih + ih)*Iw + iw]) : 0.f; \
          } \
        } \
        for (int64_t c=nn; c < MAG_GEMM_NR; ++c) dst[c] = 0.f; \
        mag_conv_kcol_next(Kd, Kh, Kw, &cig, &kd, &kh, &kw); \
      } \
      bp += kct*MAG_GEMM_NR; \
    } \
  }
mag_conv_im2col_b_impl(float, mag_cvt_nop, mag_vf32_loadu)
mag_conv_im2col_b_impl(mag_float16_t, mag_float16_to_float32, mag_vf32_loadu_f16)
mag_conv_im2col_b_impl(mag_bfloat16_t, mag_bfloat16_to_float32, mag_vf32_loadu_bf16)
mag_conv_im2col_b_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32, mag_vf32_loadu_float8_e4m3fn)
#undef mag_conv_im2col_b_impl

typedef struct mag_conv_runs_t {
  int64_t n;
  int32_t *k0;
  int32_t *len;
  int32_t *img;
  int32_t *sp;
  int32_t *od0;
  int32_t *oh0;
  int32_t *ow0;
} mag_conv_runs_t;

static void mag_conv_runs_build(mag_conv_runs_t *r, const mag_conv_geom_t *g, int64_t Osp, int64_t pc, int64_t kct) {
  int64_t Ow = g->small[2], Oh = g->small[1];
  int64_t nr = 0;
  int64_t k = 0;
  while (k < kct) {
    int64_t kk = pc + k;
    int64_t n = kk/Osp;
    int64_t sp = kk%Osp;
    int64_t ow = sp%Ow;
    int64_t t = sp/Ow;
    int64_t oh = t%Oh;
    int64_t od = t/Oh;
    int64_t len = mag_vmin(Ow-ow, kct-k);
    r->k0[nr] = (int32_t)k;
    r->len[nr] = (int32_t)len;
    r->img[nr] = (int32_t)n;
    r->sp[nr] = (int32_t)sp;
    r->od0[nr] = (int32_t)(od*g->s[0] - g->p[0]);
    r->oh0[nr] = (int32_t)(oh*g->s[1] - g->p[1]);
    r->ow0[nr] = (int32_t)(ow*g->s[2] - g->p[2]);
    ++nr;
    k += len;
  }
  r->n = nr;
}

typedef void (mag_conv_wgrad_pack_a_t)(float *restrict ap, const void *pdy, int64_t mbt, int64_t kct, int64_t cout, int64_t Osp, int64_t co0, const mag_conv_runs_t *runs);
#define mag_conv_wgrad_pack_a_impl(T, CVT) \
  static MAG_HOTPROC void mag_conv_wgrad_pack_a_##T(float *restrict ap, const void *pdy, int64_t mbt, int64_t kct, int64_t cout, int64_t Osp, int64_t co0, const mag_conv_runs_t *runs) { \
    const T *dy = (const T *)pdy; \
    for (int64_t q=0; q < mbt; q += MAG_GEMM_MR) { \
      int64_t mm = mag_vmin((int64_t)MAG_GEMM_MR, mbt-q); \
      float *dst = ap + q*kct; \
      int64_t i=0; \
      for (; i < mm; ++i) { \
        int64_t co = co0 + q + i; \
        for (int64_t r=0; r < runs->n; ++r) { \
          const T *src = dy + ((int64_t)runs->img[r]*cout + co)*Osp + runs->sp[r]; \
          float *d = dst + (int64_t)runs->k0[r]*MAG_GEMM_MR + i; \
          int64_t len = runs->len[r]; \
          for (int64_t j=0; j < len; ++j) d[j*MAG_GEMM_MR] = CVT(src[j]); \
        } \
      } \
      for (; i < MAG_GEMM_MR; ++i) for (int64_t k=0; k < kct; ++k) dst[k*MAG_GEMM_MR + i] = 0.f; \
    } \
  }
mag_conv_wgrad_pack_a_impl(float, mag_cvt_nop)
mag_conv_wgrad_pack_a_impl(mag_float16_t, mag_float16_to_float32)
mag_conv_wgrad_pack_a_impl(mag_bfloat16_t, mag_bfloat16_to_float32)
mag_conv_wgrad_pack_a_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32)
#undef mag_conv_wgrad_pack_a_impl

typedef void (mag_conv_wgrad_pack_b_t)(float *restrict bp, const void *pxg, const mag_conv_geom_t *g, int64_t kct, int64_t nct, int64_t cin, const mag_conv_runs_t *runs, const int32_t *ccig, const int32_t *ckdd, const int32_t *ckhd, const int32_t *ckwd);
#define mag_conv_wgrad_pack_b_impl(T, CVT) \
  static MAG_HOTPROC void mag_conv_wgrad_pack_b_##T(float *restrict bp, const void *pxg, const mag_conv_geom_t *g, int64_t kct, int64_t nct, int64_t cin, const mag_conv_runs_t *runs, const int32_t *ccig, const int32_t *ckdd, const int32_t *ckhd, const int32_t *ckwd) { \
    const T *xg = (const T *)pxg; \
    int64_t Id = g->big[0], Ih = g->big[1], Iw = g->big[2]; \
    int64_t Isp = Id*Ih*Iw; \
    int64_t s2 = g->s[2]; \
    for (int64_t p=0; p < nct; p += MAG_GEMM_NR) { \
      int64_t nn = mag_vmin((int64_t)MAG_GEMM_NR, nct-p); \
      int64_t c=0; \
      for (; c < nn; ++c) { \
        int64_t cig = ccig[p+c], kdd = ckdd[p+c], khd = ckhd[p+c], kwd = ckwd[p+c]; \
        const T *xplane = xg + cig*Isp; \
        for (int64_t r=0; r < runs->n; ++r) { \
          float *d = bp + (int64_t)runs->k0[r]*MAG_GEMM_NR + c; \
          int64_t len = runs->len[r]; \
          int64_t id = runs->od0[r]+kdd, ih = runs->oh0[r]+khd; \
          if ((uint64_t)id >= (uint64_t)Id || (uint64_t)ih >= (uint64_t)Ih) { \
            for (int64_t j=0; j < len; ++j) d[j*MAG_GEMM_NR] = 0.f; \
            continue; \
          } \
          const T *xrow = xplane + (int64_t)runs->img[r]*cin*Isp + (id*Ih + ih)*Iw; \
          int64_t iw0 = runs->ow0[r]+kwd; \
          int64_t lo, hi; \
          mag_conv_clip_range(iw0, s2, len, Iw, &lo, &hi); \
          for (int64_t j=0; j < lo; ++j) d[j*MAG_GEMM_NR] = 0.f; \
          if (s2 == 1) for (int64_t j=lo; j < hi; ++j) d[j*MAG_GEMM_NR] = CVT(xrow[iw0+j]); \
          else for (int64_t j=lo; j < hi; ++j) d[j*MAG_GEMM_NR] = CVT(xrow[iw0+j*s2]); \
          for (int64_t j=hi; j < len; ++j) d[j*MAG_GEMM_NR] = 0.f; \
        } \
      } \
      for (; c < MAG_GEMM_NR; ++c) for (int64_t k=0; k < kct; ++k) bp[k*MAG_GEMM_NR + c] = 0.f; \
      bp += kct*MAG_GEMM_NR; \
    } \
  }
mag_conv_wgrad_pack_b_impl(float, mag_cvt_nop)
mag_conv_wgrad_pack_b_impl(mag_float16_t, mag_float16_to_float32)
mag_conv_wgrad_pack_b_impl(mag_bfloat16_t, mag_bfloat16_to_float32)
mag_conv_wgrad_pack_b_impl(mag_float8_e4m3fn_t, mag_float8_e4m3fn_to_float32)
#undef mag_conv_wgrad_pack_b_impl

static mag_gemm_pack_a_t *const mag_conv_lut_pack_a[4] = {
  [MAG_DTYPE_FLOAT32] = &mag_gemm_pack_a_float,
  [MAG_DTYPE_FLOAT16] = &mag_gemm_pack_a_mag_float16_t,
  [MAG_DTYPE_BFLOAT16] = &mag_gemm_pack_a_mag_bfloat16_t,
  [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_pack_a_mag_float8_e4m3fn_t
};
static mag_gemm_pack_b_t *const mag_conv_lut_pack_b[4] = {
  [MAG_DTYPE_FLOAT32] = &mag_gemm_pack_b_float,
  [MAG_DTYPE_FLOAT16] = &mag_gemm_pack_b_mag_float16_t,
  [MAG_DTYPE_BFLOAT16] = &mag_gemm_pack_b_mag_bfloat16_t,
  [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_pack_b_mag_float8_e4m3fn_t
};
static mag_gemm_store_c_t *const mag_conv_lut_store_c[4] = {
  [MAG_DTYPE_FLOAT32] = &mag_gemm_store_c_float,
  [MAG_DTYPE_FLOAT16] = &mag_gemm_store_c_mag_float16_t,
  [MAG_DTYPE_BFLOAT16] = &mag_gemm_store_c_mag_bfloat16_t,
  [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_gemm_store_c_mag_float8_e4m3fn_t
};
static mag_conv_ld_t *const mag_conv_lut_ld[4] = {
  [MAG_DTYPE_FLOAT32] = &mag_conv_ld_float,
  [MAG_DTYPE_FLOAT16] = &mag_conv_ld_mag_float16_t,
  [MAG_DTYPE_BFLOAT16] = &mag_conv_ld_mag_bfloat16_t,
  [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_conv_ld_mag_float8_e4m3fn_t
};
static mag_conv_im2col_b_t *const mag_conv_lut_im2col_b[4] = {
  [MAG_DTYPE_FLOAT32] = &mag_conv_im2col_b_float,
  [MAG_DTYPE_FLOAT16] = &mag_conv_im2col_b_mag_float16_t,
  [MAG_DTYPE_BFLOAT16] = &mag_conv_im2col_b_mag_bfloat16_t,
  [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_conv_im2col_b_mag_float8_e4m3fn_t
};
static mag_conv_wgrad_pack_a_t *const mag_conv_lut_wgrad_pack_a[4] = {
  [MAG_DTYPE_FLOAT32] = &mag_conv_wgrad_pack_a_float,
  [MAG_DTYPE_FLOAT16] = &mag_conv_wgrad_pack_a_mag_float16_t,
  [MAG_DTYPE_BFLOAT16] = &mag_conv_wgrad_pack_a_mag_bfloat16_t,
  [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_conv_wgrad_pack_a_mag_float8_e4m3fn_t
};
static mag_conv_wgrad_pack_b_t *const mag_conv_lut_wgrad_pack_b[4] = {
  [MAG_DTYPE_FLOAT32] = &mag_conv_wgrad_pack_b_float,
  [MAG_DTYPE_FLOAT16] = &mag_conv_wgrad_pack_b_mag_float16_t,
  [MAG_DTYPE_BFLOAT16] = &mag_conv_wgrad_pack_b_mag_bfloat16_t,
  [MAG_DTYPE_FLOAT8_E4M3FN] = &mag_conv_wgrad_pack_b_mag_float8_e4m3fn_t
};

static MAG_HOTPROC mag_status_t mag_conv_gemm_forward(mag_error_t *err, const mag_kernel_payload_t *payload, mag_dtype_t dtype, const mag_conv_geom_t *g, void *pr, const void *px, const void *pw, const void *pb) {
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  int64_t cinG = g->cin/g->groups, coutG = g->cout/g->groups;
  int64_t Isp = g->big[0]*g->big[1]*g->big[2];
  int64_t Osp = g->small[0]*g->small[1]*g->small[2];
  int64_t Kcol = cinG*g->k[0]*g->k[1]*g->k[2];
  int64_t M = coutG, N = Osp, K = Kcol;
  int64_t batch = g->N*g->groups;
  int64_t tc = payload->thread_num;
  int64_t mt, nt;
  mag_gemm_pick_tiles(tc, batch, M, N, K, el, true, &mt, &nt);
  int64_t tiles_m = (M + mt-1)/mt;
  int64_t tiles_n = (N + nt-1)/nt;
  int64_t tiles_per_batch = tiles_m*tiles_n;
  int64_t total = batch*tiles_per_batch;
  int64_t KC = mag_vmin((int64_t)MAG_GEMM_KC, K);
  int64_t apad = mag_gemm_round_up(mt, MAG_GEMM_MR);
  int64_t ldc_max = mag_gemm_round_up(nt, MAG_GEMM_NR);
  size_t ap_nb = mag_gemm_align_up((size_t)(apad*KC)*sizeof(float));
  size_t bp_nb = mag_gemm_align_up((size_t)(ldc_max*KC)*sizeof(float));
  size_t cc_nb = mag_gemm_align_up((size_t)(apad*ldc_max)*sizeof(float));
  size_t tab_nb = mag_gemm_align_up((size_t)ldc_max*sizeof(int32_t));
  size_t mark = mag_scratch_arena_mark(&mag_tls_arena);
  uint8_t *blk = mag_scratch_arena_alloc(&mag_tls_arena, ap_nb + bp_nb + cc_nb + 3*tab_nb);
  if (mag_unlikely(!blk)) return mag_set_error(err, MAG_ERR_OOM, "conv: failed to allocate gemm scratch.");
  float *ap = (float *)blk;
  float *bp = (float *)(blk + ap_nb);
  float *cc = (float *)(blk + ap_nb + bp_nb);
  int32_t *od0 = (int32_t *)(blk + ap_nb + bp_nb + cc_nb);
  int32_t *oh0 = (int32_t *)((uint8_t *)od0 + tab_nb);
  int32_t *ow0 = (int32_t *)((uint8_t *)oh0 + tab_nb);
  mag_gemm_pack_a_t *pack_a = mag_conv_lut_pack_a[dtype];
  mag_conv_im2col_b_t *im2col = mag_conv_lut_im2col_b[dtype];
  mag_gemm_store_c_t *store_c = mag_conv_lut_store_c[dtype];
  mag_conv_ld_t *ld = mag_conv_lut_ld[dtype];
  const uint8_t *xb = px;
  const uint8_t *wb = pw;
  uint8_t *rb = pr;
  for (int64_t t=payload->thread_idx; t < total; t = tc + mag_tile_sched_acquire_next(payload->tile_sched)) {
    int64_t b = t/tiles_per_batch;
    int64_t rem = t%tiles_per_batch;
    int64_t jc = rem/tiles_m;
    int64_t ic = rem%tiles_m;
    int64_t n = b/g->groups;
    int64_t grp = b%g->groups;
    int64_t m0 = ic*mt, m1 = mag_vmin(M, m0+mt);
    int64_t n0 = jc*nt, n1 = mag_vmin(N, n0+nt);
    int64_t mbt = m1-m0, nct = n1-n0;
    int64_t ldc = mag_gemm_round_up(nct, MAG_GEMM_NR);
    int64_t mpad = mag_gemm_round_up(mbt, MAG_GEMM_MR);
    mag_conv_col_table(g, g->small, n0, nct, od0, oh0, ow0);
    for (int64_t i=0; i < mbt; ++i) {
      float bias = pb ? (*ld)(pb, grp*coutG + m0 + i) : 0.f;
      float *row = cc + i*ldc;
      for (int64_t j=0; j < ldc; ++j) row[j] = bias;
    }
    if (mpad > mbt) memset(cc + mbt*ldc, 0, (size_t)((mpad-mbt)*ldc)*sizeof(float));
    const uint8_t *wg = wb + (grp*coutG + m0)*Kcol*el;
    const uint8_t *xg = xb + (n*g->cin + grp*cinG)*Isp*el;
    for (int64_t pc=0; pc < K; pc += KC) {
      int64_t kct = mag_vmin(KC, K-pc);
      (*pack_a)(ap, wg + pc*el, mbt, kct, Kcol, 1);
      (*im2col)(bp, xg, g, pc, kct, nct, od0, oh0, ow0);
      mag_conv_ukernel_sweep(mbt, nct, kct, ap, bp, cc, ldc);
    }
    (*store_c)(rb + ((n*g->cout + grp*coutG + m0)*Osp + n0)*el, Osp, cc, ldc, mbt, nct);
  }
  mag_scratch_arena_reset(&mag_tls_arena, mark);
  return MAG_OK;
}

static MAG_HOTPROC mag_status_t mag_conv_gemm_wgrad(mag_error_t *err, const mag_kernel_payload_t *payload, mag_dtype_t dtype, const mag_conv_geom_t *g, void *pr, const void *px, const void *pdy) {
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  int64_t cinG = g->cin/g->groups, coutG = g->cout/g->groups;
  int64_t Kd = g->k[0], Kh = g->k[1], Kw = g->k[2];
  int64_t Kvol = Kd*Kh*Kw;
  int64_t Isp = g->big[0]*g->big[1]*g->big[2];
  int64_t Osp = g->small[0]*g->small[1]*g->small[2];
  int64_t Kcol = cinG*Kvol;
  int64_t M = coutG, N = Kcol, K = g->N*Osp;
  int64_t batch = g->groups;
  int64_t tc = payload->thread_num;
  int64_t mt, nt;
  mag_gemm_pick_tiles(tc, batch, M, N, K, el, true, &mt, &nt);
  int64_t tiles_m = (M + mt-1)/mt;
  int64_t tiles_n = (N + nt-1)/nt;
  int64_t tiles_per_batch = tiles_m*tiles_n;
  int64_t total = batch*tiles_per_batch;
  int64_t KC = mag_vmin((int64_t)MAG_GEMM_KC, K);
  int64_t apad = mag_gemm_round_up(mt, MAG_GEMM_MR);
  int64_t ldc_max = mag_gemm_round_up(nt, MAG_GEMM_NR);
  size_t ap_nb = mag_gemm_align_up((size_t)(apad*KC)*sizeof(float));
  size_t bp_nb = mag_gemm_align_up((size_t)(ldc_max*KC)*sizeof(float));
  size_t cc_nb = mag_gemm_align_up((size_t)(apad*ldc_max)*sizeof(float));
  size_t ktab_nb = mag_gemm_align_up((size_t)KC*sizeof(int32_t));
  size_t ctab_nb = mag_gemm_align_up((size_t)ldc_max*sizeof(int32_t));
  size_t mark = mag_scratch_arena_mark(&mag_tls_arena);
  uint8_t *blk = mag_scratch_arena_alloc(&mag_tls_arena, ap_nb + bp_nb + cc_nb + 7*ktab_nb + 4*ctab_nb);
  if (mag_unlikely(!blk)) return mag_set_error(err, MAG_ERR_OOM, "conv_wgrad: failed to allocate gemm scratch.");
  float *ap = (float *)blk;
  float *bp = (float *)(blk + ap_nb);
  float *cc = (float *)(blk + ap_nb + bp_nb);
  uint8_t *tp = blk + ap_nb + bp_nb + cc_nb;
  mag_conv_runs_t runs;
  runs.k0 = (int32_t *)tp; tp += ktab_nb;
  runs.len = (int32_t *)tp; tp += ktab_nb;
  runs.img = (int32_t *)tp; tp += ktab_nb;
  runs.sp = (int32_t *)tp; tp += ktab_nb;
  runs.od0 = (int32_t *)tp; tp += ktab_nb;
  runs.oh0 = (int32_t *)tp; tp += ktab_nb;
  runs.ow0 = (int32_t *)tp; tp += ktab_nb;
  int32_t *ccig = (int32_t *)tp; tp += ctab_nb;
  int32_t *ckdd = (int32_t *)tp; tp += ctab_nb;
  int32_t *ckhd = (int32_t *)tp; tp += ctab_nb;
  int32_t *ckwd = (int32_t *)tp;
  mag_conv_wgrad_pack_a_t *pack_a = mag_conv_lut_wgrad_pack_a[dtype];
  mag_conv_wgrad_pack_b_t *pack_b = mag_conv_lut_wgrad_pack_b[dtype];
  mag_gemm_store_c_t *store_c = mag_conv_lut_store_c[dtype];
  const uint8_t *xb = px;
  const uint8_t *dyb = pdy;
  uint8_t *rb = pr;
  for (int64_t t=payload->thread_idx; t < total; t = tc + mag_tile_sched_acquire_next(payload->tile_sched)) {
    int64_t grp = t/tiles_per_batch;
    int64_t rem = t%tiles_per_batch;
    int64_t jc = rem/tiles_m;
    int64_t ic = rem%tiles_m;
    int64_t m0 = ic*mt, m1 = mag_vmin(M, m0+mt);
    int64_t n0 = jc*nt, n1 = mag_vmin(N, n0+nt);
    int64_t mbt = m1-m0, nct = n1-n0;
    int64_t ldc = mag_gemm_round_up(nct, MAG_GEMM_NR);
    int64_t mpad = mag_gemm_round_up(mbt, MAG_GEMM_MR);
    for (int64_t c=0; c < nct; ++c) {
      int64_t cig, kd, kh, kw;
      mag_conv_kcol_decode(n0+c, Kd, Kh, Kw, &cig, &kd, &kh, &kw);
      ccig[c] = (int32_t)cig;
      ckdd[c] = (int32_t)(kd*g->d[0]);
      ckhd[c] = (int32_t)(kh*g->d[1]);
      ckwd[c] = (int32_t)(kw*g->d[2]);
    }
    memset(cc, 0, (size_t)(mpad*ldc)*sizeof(float));
    const uint8_t *xg = xb + grp*cinG*Isp*el;
    const uint8_t *dyg = dyb + grp*coutG*Osp*el;
    for (int64_t pc=0; pc < K; pc += KC) {
      int64_t kct = mag_vmin(KC, K-pc);
      mag_conv_runs_build(&runs, g, Osp, pc, kct);
      (*pack_a)(ap, dyg, mbt, kct, g->cout, Osp, m0, &runs);
      (*pack_b)(bp, xg, g, kct, nct, g->cin, &runs, ccig, ckdd, ckhd, ckwd);
      mag_conv_ukernel_sweep(mbt, nct, kct, ap, bp, cc, ldc);
    }
    (*store_c)(rb + ((grp*coutG + m0)*Kcol + n0)*el, Kcol, cc, ldc, mbt, nct);
  }
  mag_scratch_arena_reset(&mag_tls_arena, mark);
  return MAG_OK;
}

static MAG_HOTPROC mag_status_t mag_conv_gemm_transpose(mag_error_t *err, const mag_kernel_payload_t *payload, mag_dtype_t dtype, const mag_conv_geom_t *g, void *pr, const void *px, const void *pw, const void *pb) {
  int64_t el = (int64_t)mag_type_trait(dtype)->size;
  int64_t cinG = g->cin/g->groups, coutG = g->cout/g->groups;
  int64_t Kd = g->k[0], Kh = g->k[1], Kw = g->k[2];
  int64_t Kvol = Kd*Kh*Kw;
  int64_t Od = g->big[0], Oh = g->big[1], Ow = g->big[2];
  int64_t Isp = g->small[0]*g->small[1]*g->small[2];
  int64_t Osp = Od*Oh*Ow;
  int64_t N = Isp, K = cinG;
  int64_t batch = g->N*g->groups;
  int64_t tc = payload->thread_num;
  int64_t cpt = mag_vmax((int64_t)1, mag_vmin((int64_t)MAG_GEMM_MC/Kvol, coutG));
  int64_t want = (coutG*batch + 4*tc-1)/(4*tc);
  if (tc > 1 && want < cpt) cpt = mag_vmax((int64_t)1, want);
  int64_t cblocks = (coutG + cpt-1)/cpt;
  int64_t total = batch*cblocks;
  int64_t mt = cpt*Kvol;
  int64_t nt = mag_vmin((int64_t)MAG_GEMM_NC, mag_gemm_round_up(N, MAG_GEMM_NR));
  int64_t KC = mag_vmin((int64_t)MAG_GEMM_KC, K);
  int64_t apad = mag_gemm_round_up(mt, MAG_GEMM_MR);
  int64_t ldc_max = mag_gemm_round_up(nt, MAG_GEMM_NR);
  size_t ap_nb = mag_gemm_align_up((size_t)(apad*KC)*sizeof(float));
  size_t bp_nb = mag_gemm_align_up((size_t)(ldc_max*KC)*sizeof(float));
  size_t cc_nb = mag_gemm_align_up((size_t)(apad*ldc_max)*sizeof(float));
  size_t acc_nb = mag_gemm_align_up((size_t)(cpt*Osp)*sizeof(float));
  size_t tab_nb = mag_gemm_align_up((size_t)ldc_max*sizeof(int32_t));
  size_t mark = mag_scratch_arena_mark(&mag_tls_arena);
  uint8_t *blk = mag_scratch_arena_alloc(&mag_tls_arena, ap_nb + bp_nb + cc_nb + acc_nb + 3*tab_nb);
  if (mag_unlikely(!blk)) return mag_set_error(err, MAG_ERR_OOM, "conv_transpose: failed to allocate gemm scratch.");
  float *ap = (float *)blk;
  float *bp = (float *)(blk + ap_nb);
  float *cc = (float *)(blk + ap_nb + bp_nb);
  float *acc = (float *)(blk + ap_nb + bp_nb + cc_nb);
  int32_t *id0 = (int32_t *)(blk + ap_nb + bp_nb + cc_nb + acc_nb);
  int32_t *ih0 = (int32_t *)((uint8_t *)id0 + tab_nb);
  int32_t *iw0 = (int32_t *)((uint8_t *)ih0 + tab_nb);
  mag_gemm_pack_a_t *pack_a = mag_conv_lut_pack_a[dtype];
  mag_gemm_pack_b_t *pack_b = mag_conv_lut_pack_b[dtype];
  mag_gemm_store_c_t *store_c = mag_conv_lut_store_c[dtype];
  mag_conv_ld_t *ld = mag_conv_lut_ld[dtype];
  const uint8_t *xb = px;
  const uint8_t *wb = pw;
  uint8_t *rb = pr;
  int64_t sx1 = coutG*Kvol;
  for (int64_t t=payload->thread_idx; t < total; t = tc + mag_tile_sched_acquire_next(payload->tile_sched)) {
    int64_t b = t/cblocks;
    int64_t cb = t%cblocks;
    int64_t n = b/g->groups;
    int64_t grp = b%g->groups;
    int64_t co0 = cb*cpt;
    int64_t cpt_t = mag_vmin(cpt, coutG-co0);
    int64_t mbt = cpt_t*Kvol;
    int64_t mpad = mag_gemm_round_up(mbt, MAG_GEMM_MR);
    for (int64_t c=0; c < cpt_t; ++c) {
      float bias = pb ? (*ld)(pb, grp*coutG + co0 + c) : 0.f;
      float *plane = acc + c*Osp;
      for (int64_t i=0; i < Osp; ++i) plane[i] = bias;
    }
    const uint8_t *wg = wb + ((grp*cinG)*coutG + co0)*Kvol*el;
    const uint8_t *xg = xb + (n*g->cin + grp*cinG)*Isp*el;
    for (int64_t n0=0; n0 < N; n0 += nt) {
      int64_t n1 = mag_vmin(N, n0+nt);
      int64_t nct = n1-n0;
      int64_t ldc = mag_gemm_round_up(nct, MAG_GEMM_NR);
      mag_conv_col_table(g, g->small, n0, nct, id0, ih0, iw0);
      memset(cc, 0, (size_t)(mpad*ldc)*sizeof(float));
      for (int64_t pc=0; pc < K; pc += KC) {
        int64_t kct = mag_vmin(KC, K-pc);
        (*pack_a)(ap, wg + pc*sx1*el, mbt, kct, 1, sx1);
        (*pack_b)(bp, xg + (pc*Isp + n0)*el, kct, nct, Isp, 1);
        mag_conv_ukernel_sweep(mbt, nct, kct, ap, bp, cc, ldc);
      }
      for (int64_t i=0; i < mbt; ++i) {
        int64_t c = i/Kvol;
        int64_t kk = i%Kvol;
        int64_t kw = kk%Kw; kk /= Kw;
        int64_t kh = kk%Kh; kk /= Kh;
        int64_t kd = kk;
        int64_t kdd = kd*g->d[0], khd = kh*g->d[1], kwd = kw*g->d[2];
        float *plane = acc + c*Osp;
        const float *src = cc + i*ldc;
        for (int64_t j=0; j < nct; ++j) {
          int64_t od = id0[j]+kdd, oh = ih0[j]+khd, ow = iw0[j]+kwd;
          if ((uint64_t)od >= (uint64_t)Od || (uint64_t)oh >= (uint64_t)Oh || (uint64_t)ow >= (uint64_t)Ow) continue;
          plane[(od*Oh + oh)*Ow + ow] += src[j];
        }
      }
    }
    (*store_c)(rb + (n*g->cout + grp*coutG + co0)*Osp*el, Osp, acc, Osp, cpt_t, Osp);
  }
  mag_scratch_arena_reset(&mag_tls_arena, mark);
  return MAG_OK;
}


#ifndef MAG_CONV_ACCEL_SCRATCH_BYTES
  #define MAG_CONV_ACCEL_SCRATCH_BYTES (8<<20)
#endif
#ifndef MAG_CONV_ACCEL_MIN_FLOPS
  #define MAG_CONV_ACCEL_MIN_FLOPS (1<<22)
#endif

static MAG_HOTPROC void mag_conv_im2col_rows_f32(float *restrict dst, int64_t ld, const float *xg, const mag_conv_geom_t *g, int64_t k0, int64_t k1, int64_t n0, int64_t nct, const int32_t *od0, const int32_t *oh0, const int32_t *ow0) {
  int64_t Id = g->big[0], Ih = g->big[1], Iw = g->big[2];
  int64_t Kd = g->k[0], Kh = g->k[1], Kw = g->k[2];
  int64_t Isp = Id*Ih*Iw;
  int64_t Ow = g->small[2];
  int64_t s2 = g->s[2];
  int64_t cig, kd, kh, kw;
  mag_conv_kcol_decode(k0, Kd, Kh, Kw, &cig, &kd, &kh, &kw);
  for (int64_t k=k0; k < k1; ++k) {
    const float *xplane = xg + cig*Isp;
    int64_t kdd = kd*g->d[0], khd = kh*g->d[1], kwd = kw*g->d[2];
    float *row = dst + (k-k0)*ld;
    int64_t c = 0;
    while (c < nct) {
      int64_t ow = (n0+c)%Ow;
      int64_t len = mag_vmin(Ow-ow, nct-c);
      int64_t id = od0[c]+kdd, ih = oh0[c]+khd;
      if ((uint64_t)id >= (uint64_t)Id || (uint64_t)ih >= (uint64_t)Ih) {
        memset(row+c, 0, (size_t)len*sizeof(float));
      } else {
        const float *xrow = xplane + (id*Ih + ih)*Iw;
        int64_t iw0 = ow0[c]+kwd;
        int64_t lo, hi;
        mag_conv_clip_range(iw0, s2, len, Iw, &lo, &hi);
        if (lo > 0) memset(row+c, 0, (size_t)lo*sizeof(float));
        if (s2 == 1) {
          if (hi > lo) memcpy(row+c+lo, xrow+iw0+lo, (size_t)(hi-lo)*sizeof(float));
        } else {
          for (int64_t j=lo; j < hi; ++j) row[c+j] = xrow[iw0 + j*s2];
        }
        if (len > hi) memset(row+c+hi, 0, (size_t)(len-hi)*sizeof(float));
      }
      c += len;
    }
    mag_conv_kcol_next(Kd, Kh, Kw, &cig, &kd, &kh, &kw);
  }
}

static MAG_HOTPROC void mag_conv_col2im_rows_f32(float *restrict plane, int64_t Osp, const float *cols, int64_t ld, const mag_conv_geom_t *g, int64_t mbt, const int32_t *id0, const int32_t *ih0, const int32_t *iw0) {
  int64_t Od = g->big[0], Oh = g->big[1], Ow = g->big[2];
  int64_t Kd = g->k[0], Kh = g->k[1], Kw = g->k[2];
  int64_t Kvol = Kd*Kh*Kw;
  int64_t Isp = g->small[0]*g->small[1]*g->small[2];
  int64_t Iw = g->small[2];
  int64_t s2 = g->s[2];
  for (int64_t i=0; i < mbt; ++i) {
    int64_t c = i/Kvol;
    int64_t kk = i%Kvol;
    int64_t kw = kk%Kw; kk /= Kw;
    int64_t kh = kk%Kh; kk /= Kh;
    int64_t kd = kk;
    int64_t kdd = kd*g->d[0], khd = kh*g->d[1], kwd = kw*g->d[2];
    float *out = plane + c*Osp;
    const float *src = cols + i*ld;
    int64_t j = 0;
    while (j < Isp) {
      int64_t iw = j%Iw;
      int64_t len = mag_vmin(Iw-iw, Isp-j);
      int64_t od = id0[j]+kdd, oh = ih0[j]+khd;
      if ((uint64_t)od < (uint64_t)Od && (uint64_t)oh < (uint64_t)Oh) {
        float *orow = out + (od*Oh + oh)*Ow;
        int64_t ow0 = iw0[j]+kwd;
        int64_t lo, hi;
        mag_conv_clip_range(ow0, s2, len, Ow, &lo, &hi);
        if (s2 == 1) for (int64_t jj=lo; jj < hi; ++jj) orow[ow0+jj] += src[j+jj];
        else for (int64_t jj=lo; jj < hi; ++jj) orow[ow0+jj*s2] += src[j+jj];
      }
      j += len;
    }
  }
}

static MAG_HOTPROC mag_status_t mag_conv_accel_forward(mag_error_t *err, const mag_kernel_payload_t *payload, const mag_conv_geom_t *g, float *pr, const float *px, const float *pw, const float *pb) {
  int64_t cinG = g->cin/g->groups, coutG = g->cout/g->groups;
  int64_t Isp = g->big[0]*g->big[1]*g->big[2];
  int64_t Osp = g->small[0]*g->small[1]*g->small[2];
  int64_t Kcol = cinG*g->k[0]*g->k[1]*g->k[2];
  int64_t batch = g->N*g->groups;
  int64_t tc = payload->thread_num;
  int64_t nt = mag_vmax((int64_t)64, mag_vmin(Osp, (int64_t)(MAG_CONV_ACCEL_SCRATCH_BYTES/((size_t)Kcol*sizeof(float)))));
  int64_t nblocks = (Osp + nt-1)/nt;
  if (tc > 1 && batch*nblocks < tc) {
    nblocks = mag_vmin(Osp, (tc + batch-1)/batch);
    nt = (Osp + nblocks-1)/nblocks;
    nblocks = (Osp + nt-1)/nt;
  }
  int64_t total = batch*nblocks;
  size_t col_nb = mag_gemm_align_up((size_t)(Kcol*nt)*sizeof(float));
  size_t tab_nb = mag_gemm_align_up((size_t)nt*sizeof(int32_t));
  size_t mark = mag_scratch_arena_mark(&mag_tls_arena);
  uint8_t *blk = mag_scratch_arena_alloc(&mag_tls_arena, col_nb + 3*tab_nb);
  if (mag_unlikely(!blk)) return mag_set_error(err, MAG_ERR_OOM, "conv: failed to allocate im2col scratch.");
  float *col = (float *)blk;
  int32_t *od0 = (int32_t *)(blk + col_nb);
  int32_t *oh0 = (int32_t *)((uint8_t *)od0 + tab_nb);
  int32_t *ow0 = (int32_t *)((uint8_t *)oh0 + tab_nb);
  for (int64_t t=payload->thread_idx; t < total; t = tc + mag_tile_sched_acquire_next(payload->tile_sched)) {
    int64_t b = t/nblocks;
    int64_t jc = t%nblocks;
    int64_t n = b/g->groups;
    int64_t grp = b%g->groups;
    int64_t n0 = jc*nt;
    int64_t nct = mag_vmin(nt, Osp-n0);
    mag_conv_col_table(g, g->small, n0, nct, od0, oh0, ow0);
    const float *xg = px + (n*g->cin + grp*cinG)*Isp;
    mag_conv_im2col_rows_f32(col, nct, xg, g, 0, Kcol, n0, nct, od0, oh0, ow0);
    float *C = pr + (n*g->cout + grp*coutG)*Osp + n0;
    float beta = 0.f;
    if (pb) {
      for (int64_t i=0; i < coutG; ++i) {
        float bias = pb[grp*coutG + i];
        float *row = C + i*Osp;
        for (int64_t j=0; j < nct; ++j) row[j] = bias;
      }
      beta = 1.f;
    }
    mag_accel_sgemm_ex(false, false, coutG, nct, Kcol, 1.f, pw + grp*coutG*Kcol, Kcol, col, nct, beta, C, Osp);
  }
  mag_scratch_arena_reset(&mag_tls_arena, mark);
  return MAG_OK;
}

static MAG_HOTPROC mag_status_t mag_conv_accel_wgrad(mag_error_t *err, const mag_kernel_payload_t *payload, const mag_conv_geom_t *g, float *pr, const float *px, const float *pdy) {
  int64_t cinG = g->cin/g->groups, coutG = g->cout/g->groups;
  int64_t Isp = g->big[0]*g->big[1]*g->big[2];
  int64_t Osp = g->small[0]*g->small[1]*g->small[2];
  int64_t Kcol = cinG*g->k[0]*g->k[1]*g->k[2];
  int64_t tc = payload->thread_num;
  int64_t nblocks = tc > 1 ? mag_vmin(Kcol, mag_vmax((int64_t)1, (2*tc + g->groups-1)/g->groups)) : 1;
  int64_t nt = (Kcol + nblocks-1)/nblocks;
  int64_t max_nt = mag_vmax((int64_t)1, (int64_t)(MAG_CONV_ACCEL_SCRATCH_BYTES/((size_t)Osp*sizeof(float))));
  nt = mag_vmin(nt, max_nt);
  nblocks = (Kcol + nt-1)/nt;
  int64_t total = g->groups*nblocks;
  size_t col_nb = mag_gemm_align_up((size_t)(nt*Osp)*sizeof(float));
  size_t tab_nb = mag_gemm_align_up((size_t)Osp*sizeof(int32_t));
  size_t mark = mag_scratch_arena_mark(&mag_tls_arena);
  uint8_t *blk = mag_scratch_arena_alloc(&mag_tls_arena, col_nb + 3*tab_nb);
  if (mag_unlikely(!blk)) return mag_set_error(err, MAG_ERR_OOM, "conv_wgrad: failed to allocate im2col scratch.");
  float *col = (float *)blk;
  int32_t *od0 = (int32_t *)(blk + col_nb);
  int32_t *oh0 = (int32_t *)((uint8_t *)od0 + tab_nb);
  int32_t *ow0 = (int32_t *)((uint8_t *)oh0 + tab_nb);
  mag_conv_col_table(g, g->small, 0, Osp, od0, oh0, ow0);
  for (int64_t t=payload->thread_idx; t < total; t = tc + mag_tile_sched_acquire_next(payload->tile_sched)) {
    int64_t grp = t/nblocks;
    int64_t k0 = (t%nblocks)*nt;
    int64_t k1 = mag_vmin(Kcol, k0+nt);
    int64_t kn = k1-k0;
    float *C = pr + (grp*coutG)*Kcol + k0;
    for (int64_t n=0; n < g->N; ++n) {
      const float *xg = px + (n*g->cin + grp*cinG)*Isp;
      const float *dy = pdy + (n*g->cout + grp*coutG)*Osp;
      mag_conv_im2col_rows_f32(col, Osp, xg, g, k0, k1, 0, Osp, od0, oh0, ow0);
      mag_accel_sgemm_ex(false, true, coutG, kn, Osp, 1.f, dy, Osp, col, Osp, n ? 1.f : 0.f, C, Kcol);
    }
  }
  mag_scratch_arena_reset(&mag_tls_arena, mark);
  return MAG_OK;
}

static MAG_HOTPROC mag_status_t mag_conv_accel_transpose(mag_error_t *err, const mag_kernel_payload_t *payload, const mag_conv_geom_t *g, float *pr, const float *px, const float *pw, const float *pb) {
  int64_t cinG = g->cin/g->groups, coutG = g->cout/g->groups;
  int64_t Kvol = g->k[0]*g->k[1]*g->k[2];
  int64_t Isp = g->small[0]*g->small[1]*g->small[2];
  int64_t Osp = g->big[0]*g->big[1]*g->big[2];
  int64_t batch = g->N*g->groups;
  int64_t tc = payload->thread_num;
  int64_t cpt = mag_vmax((int64_t)1, mag_vmin((int64_t)(MAG_CONV_ACCEL_SCRATCH_BYTES/((size_t)(Kvol*Isp)*sizeof(float))), coutG));
  int64_t want = (coutG*batch + 2*tc-1)/(2*tc);
  if (tc > 1 && want < cpt) cpt = mag_vmax((int64_t)1, want);
  int64_t cblocks = (coutG + cpt-1)/cpt;
  int64_t total = batch*cblocks;
  int64_t sx1 = coutG*Kvol;
  size_t col_nb = mag_gemm_align_up((size_t)(cpt*Kvol*Isp)*sizeof(float));
  size_t tab_nb = mag_gemm_align_up((size_t)Isp*sizeof(int32_t));
  size_t mark = mag_scratch_arena_mark(&mag_tls_arena);
  uint8_t *blk = mag_scratch_arena_alloc(&mag_tls_arena, col_nb + 3*tab_nb);
  if (mag_unlikely(!blk)) return mag_set_error(err, MAG_ERR_OOM, "conv_transpose: failed to allocate col2im scratch.");
  float *cols = (float *)blk;
  int32_t *id0 = (int32_t *)(blk + col_nb);
  int32_t *ih0 = (int32_t *)((uint8_t *)id0 + tab_nb);
  int32_t *iw0 = (int32_t *)((uint8_t *)ih0 + tab_nb);
  mag_conv_col_table(g, g->small, 0, Isp, id0, ih0, iw0);
  for (int64_t t=payload->thread_idx; t < total; t = tc + mag_tile_sched_acquire_next(payload->tile_sched)) {
    int64_t b = t/cblocks;
    int64_t cb = t%cblocks;
    int64_t n = b/g->groups;
    int64_t grp = b%g->groups;
    int64_t co0 = cb*cpt;
    int64_t cpt_t = mag_vmin(cpt, coutG-co0);
    int64_t mbt = cpt_t*Kvol;
    const float *wg = pw + ((grp*cinG)*coutG + co0)*Kvol;
    const float *xg = px + (n*g->cin + grp*cinG)*Isp;
    mag_accel_sgemm_ex(true, false, mbt, Isp, cinG, 1.f, wg, sx1, xg, Isp, 0.f, cols, Isp);
    float *plane = pr + (n*g->cout + grp*coutG + co0)*Osp;
    for (int64_t c=0; c < cpt_t; ++c) {
      float bias = pb ? pb[grp*coutG + co0 + c] : 0.f;
      float *out = plane + c*Osp;
      for (int64_t i=0; i < Osp; ++i) out[i] = bias;
    }
    mag_conv_col2im_rows_f32(plane, Osp, cols, Isp, g, mbt, id0, ih0, iw0);
  }
  mag_scratch_arena_reset(&mag_tls_arena, mark);
  return MAG_OK;
}

static MAG_AINLINE bool mag_conv_use_accel(mag_dtype_t dtype, int64_t M, int64_t N, int64_t K, int64_t batch) {
  if (dtype != MAG_DTYPE_FLOAT32 || !mag_accel_sgemm_available()) return false;
  return 2.0*(double)M*(double)N*(double)K*(double)batch >= (double)MAG_CONV_ACCEL_MIN_FLOPS;
}

#define mag_gen_stub_conv_entry(T, TF) \
  static mag_status_t MAG_HOTPROC mag_conv_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *w = payload->cmd->in[1]; \
    const mag_tensor_t *b = payload->cmd->num_in > 2 ? payload->cmd->in[2] : NULL; \
    if (mag_unlikely(r->meta.numel == 0)) return MAG_OK; \
    mag_conv_geom_t g; \
    mag_conv_geom_init(&g, payload->cmd->params, x->meta.coords.shape+2, r->meta.coords.shape+2, w->meta.coords.shape+2); \
    g.N = x->meta.coords.shape[0]; \
    g.cin = x->meta.coords.shape[1]; \
    g.cout = r->meta.coords.shape[1]; \
    int64_t coutG = g.cout/g.groups; \
    int64_t Kcol = g.cin/g.groups*g.k[0]*g.k[1]*g.k[2]; \
    if (coutG < MAG_CONV_GEMM_MIN_M || Kcol < MAG_CONV_GEMM_MIN_K) return mag_conv_direct_##TF(err, payload); \
    if (mag_conv_use_accel(r->meta.dtype, coutG, g.small[0]*g.small[1]*g.small[2], Kcol, g.N*g.groups)) \
      return mag_conv_accel_forward(err, payload, &g, (float *)mag_tensor_data_ptr_mut(r), (const float *)mag_tensor_data_ptr(x), (const float *)mag_tensor_data_ptr(w), b ? (const float *)mag_tensor_data_ptr(b) : NULL); \
    return mag_conv_gemm_forward(err, payload, r->meta.dtype, &g, (void *)mag_tensor_data_ptr_mut(r), (const void *)mag_tensor_data_ptr(x), (const void *)mag_tensor_data_ptr(w), b ? (const void *)mag_tensor_data_ptr(b) : NULL); \
  } \
  static mag_status_t MAG_HOTPROC mag_conv_transpose_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *w = payload->cmd->in[1]; \
    const mag_tensor_t *b = payload->cmd->num_in > 2 ? payload->cmd->in[2] : NULL; \
    if (mag_unlikely(r->meta.numel == 0)) return MAG_OK; \
    mag_conv_geom_t g; \
    mag_conv_geom_init(&g, payload->cmd->params, r->meta.coords.shape+2, x->meta.coords.shape+2, w->meta.coords.shape+2); \
    g.N = x->meta.coords.shape[0]; \
    g.cin = x->meta.coords.shape[1]; \
    g.cout = r->meta.coords.shape[1]; \
    int64_t cinG = g.cin/g.groups; \
    int64_t M = g.cout/g.groups*g.k[0]*g.k[1]*g.k[2]; \
    if (cinG < MAG_CONV_GEMM_MIN_K || M < MAG_CONV_GEMM_MIN_M) return mag_conv_transpose_direct_##TF(err, payload); \
    if (mag_conv_use_accel(r->meta.dtype, M, g.small[0]*g.small[1]*g.small[2], cinG, g.N*g.groups)) \
      return mag_conv_accel_transpose(err, payload, &g, (float *)mag_tensor_data_ptr_mut(r), (const float *)mag_tensor_data_ptr(x), (const float *)mag_tensor_data_ptr(w), b ? (const float *)mag_tensor_data_ptr(b) : NULL); \
    return mag_conv_gemm_transpose(err, payload, r->meta.dtype, &g, (void *)mag_tensor_data_ptr_mut(r), (const void *)mag_tensor_data_ptr(x), (const void *)mag_tensor_data_ptr(w), b ? (const void *)mag_tensor_data_ptr(b) : NULL); \
  } \
  static mag_status_t MAG_HOTPROC mag_conv_wgrad_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *a = payload->cmd->in[0]; \
    const mag_tensor_t *bt = payload->cmd->in[1]; \
    if (mag_unlikely(r->meta.numel == 0)) return MAG_OK; \
    mag_conv_geom_t g; \
    mag_conv_geom_init(&g, payload->cmd->params, a->meta.coords.shape+2, bt->meta.coords.shape+2, r->meta.coords.shape+2); \
    g.N = a->meta.coords.shape[0]; \
    g.cin = a->meta.coords.shape[1]; \
    g.cout = bt->meta.coords.shape[1]; \
    int64_t coutG = g.cout/g.groups; \
    int64_t Kcol = g.cin/g.groups*g.k[0]*g.k[1]*g.k[2]; \
    if (coutG < MAG_CONV_GEMM_MIN_M || Kcol < MAG_CONV_GEMM_MIN_K) return mag_conv_wgrad_direct_##TF(err, payload); \
    if (mag_conv_use_accel(r->meta.dtype, coutG, Kcol, g.small[0]*g.small[1]*g.small[2], g.N*g.groups)) \
      return mag_conv_accel_wgrad(err, payload, &g, (float *)mag_tensor_data_ptr_mut(r), (const float *)mag_tensor_data_ptr(a), (const float *)mag_tensor_data_ptr(bt)); \
    return mag_conv_gemm_wgrad(err, payload, r->meta.dtype, &g, (void *)mag_tensor_data_ptr_mut(r), (const void *)mag_tensor_data_ptr(a), (const void *)mag_tensor_data_ptr(bt)); \
  }

mag_gen_stub_conv_entry(float, float32)
mag_gen_stub_conv_entry(mag_float16_t, float16)
mag_gen_stub_conv_entry(mag_bfloat16_t, bfloat16)
mag_gen_stub_conv_entry(mag_float8_e4m3fn_t, float8_e4m3fn)

#undef mag_gen_stub_conv_entry
