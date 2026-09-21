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
  static mag_status_t MAG_HOTPROC mag_conv_wgrad_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
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
