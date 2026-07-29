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

#define mag_gen_stub_cat(T, TF) \
  static MAG_HOTPROC mag_status_t mag_cat_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    int64_t dim = payload->cmd->params->cat.dim; \
    int64_t n = payload->cmd->num_in; \
    mag_assert2(r && n > 0); \
    mag_assert2(dim >= 0 && dim < r->meta.coords.rank); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    mag_assert2(mag_tensor_is_contiguous(r)); \
    int64_t inner = 1; \
    for (int64_t d = dim+1; d < r->meta.coords.rank; ++d) inner *= r->meta.coords.shape[d]; \
    int64_t outer = 1; \
    for (int64_t d = 0; d < dim; ++d) outer *= r->meta.coords.shape[d]; \
    int64_t out_dim = r->meta.coords.shape[dim]; \
    int64_t out_outer_stride = out_dim * inner; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (outer + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_xmin(oa + chunk, outer); \
    bool all_contig = true; \
    for (int64_t i = 0; i < n && all_contig; ++i) \
      if (!mag_tensor_is_contiguous(payload->cmd->in[i])) all_contig = false; \
    if (mag_likely(all_contig)) { \
      int64_t *xi_outer = mag_scratch_arena_alloc(&mag_tls_arena, n*sizeof(*xi_outer)); \
      const T **bxi = mag_scratch_arena_alloc(&mag_tls_arena, n*sizeof(*bxi)); \
      for (int64_t i=0; i < n; ++i) { \
        const mag_tensor_t *x = payload->cmd->in[i]; \
        xi_outer[i] = x->meta.coords.shape[dim] * inner; \
        bxi[i] = (const T *)mag_tensor_data_ptr(x); \
      } \
      for (int64_t p = oa; p < ob; ++p) { \
        T *dst = br + p * out_outer_stride; \
        for (int64_t i = 0; i < n; ++i) { \
          size_t nb = (size_t)xi_outer[i] * sizeof(T); \
          memcpy(dst, bxi[i] + p*xi_outer[i], nb); \
          dst += xi_outer[i]; \
        } \
      } \
      mag_scratch_arena_clear(&mag_tls_arena); \
    } else { \
      int64_t mult[MAG_MAX_DIMS]; \
      for (int64_t d = 0; d < dim; ++d) { \
        int64_t m = 1; \
        for (int64_t k = d+1; k < dim; ++k) m *= r->meta.coords.shape[k]; \
        mult[d] = m; \
      } \
      for (int64_t p = oa; p < ob; ++p) { \
        int64_t idx_prefix[MAG_MAX_DIMS]; \
        int64_t rtmp = p; \
        for (int64_t d = 0; d < dim; ++d) { \
          int64_t q = mult[d] ? rtmp/mult[d] : 0; \
          if (mult[d]) rtmp %= mult[d]; \
          idx_prefix[d] = q; \
        } \
        int64_t moff = 0; \
        for (int64_t d = 0; d < dim; ++d) moff += idx_prefix[d]*r->meta.coords.strides[d]; \
        int64_t cur = 0; \
        for (int64_t i = 0; i < n; ++i) { \
          const mag_tensor_t *x = payload->cmd->in[i]; \
          int64_t smoff = 0; \
          for (int64_t d = 0; d < dim; ++d) smoff += idx_prefix[d]*x->meta.coords.strides[d]; \
          int64_t cl = x->meta.coords.shape[dim]; \
          const T *bx = (const T *)mag_tensor_data_ptr(x); \
          memcpy(br + moff + cur*r->meta.coords.strides[dim], bx + smoff, (size_t)(cl*inner)*sizeof(T)); \
          cur += cl; \
        } \
      } \
    } \
    return MAG_OK; \
  }

mag_gen_stub_cat(float, float32)
mag_gen_stub_cat(mag_float16_t, float16)
mag_gen_stub_cat(mag_bfloat16_t, bfloat16)
mag_gen_stub_cat(mag_float8_e4m3fn_t, float8_e4m3fn)
mag_gen_stub_cat(uint8_t, uint8)
mag_gen_stub_cat(int8_t, int8)
mag_gen_stub_cat(uint16_t, uint16)
mag_gen_stub_cat(int16_t, int16)
mag_gen_stub_cat(uint32_t, uint32)
mag_gen_stub_cat(int32_t, int32)
mag_gen_stub_cat(uint64_t, uint64)
mag_gen_stub_cat(int64_t, int64)

#undef mag_gen_stub_cat


typedef struct mag_rb_group_t {
  int64_t size;      /* number of elements along the collapsed group */
  int64_t xstride;   /* element stride into x for this group */
  int64_t rstride;   /* element stride into r (kept groups only, else 0) */
} mag_rb_group_t;

typedef enum mag_rb_mode_t {
  MAG_RB_GENERAL = 0,     /* arbitrary strides: strided scatter-free per-output reduction */
  MAG_RB_INNER_KEPT,      /* contiguous, innermost axis kept -> contiguous output blocks */
  MAG_RB_INNER_REDUCED    /* contiguous, innermost axis reduced -> contiguous inner runs */
} mag_rb_mode_t;

typedef struct mag_rb_plan_t {
  mag_rb_mode_t mode;
  int64_t red_prod;                    /* total number of reduced elements per output */
  int64_t inner;                       /* size of the innermost collapsed group (fast paths) */
  mag_rb_group_t kept[MAG_MAX_DIMS];   /* collapsed kept groups, outer -> inner */
  int64_t nkept;
  mag_rb_group_t red[MAG_MAX_DIMS];    /* collapsed reduced groups, outer -> inner */
  int64_t nred;
  int64_t g_rank;                      /* general path: r rank */
  int64_t g_rshape[MAG_MAX_DIMS];      /* general path: r shape */
  int64_t g_rstride[MAG_MAX_DIMS];     /* general path: r strides */
  int64_t g_xkept[MAG_MAX_DIMS];       /* general path: x stride if this r axis is kept, else 0 */
  mag_rb_group_t g_red[MAG_MAX_DIMS];  /* general path: reduced x axes (uncollapsed) */
  int64_t g_nred;
} mag_rb_plan_t;

static void mag_rb_build_plan(mag_rb_plan_t *p, const mag_tensor_t *r, const mag_tensor_t *x) {
  int64_t rr = r->meta.coords.rank;
  int64_t rx = x->meta.coords.rank;
  const int64_t *rd = r->meta.coords.shape;
  const int64_t *rs = r->meta.coords.strides;
  const int64_t *xd = x->meta.coords.shape;
  const int64_t *xs = x->meta.coords.strides;
  int64_t delta = rx - rr;

  p->g_rank = rr;
  for (int64_t kd=0; kd < rr; ++kd) {
    p->g_rshape[kd] = rd[kd];
    p->g_rstride[kd] = rs[kd];
    p->g_xkept[kd] = (rd[kd] > 1) ? xs[kd+delta] : 0;
  }
  p->g_nred = 0;
  int64_t red_prod = 1;
  for (int64_t k=0; k < rx; ++k) {
    int64_t kd = k - delta;
    bool reduced = (kd < 0) || (rd[kd] == 1);
    if (reduced && xd[k] > 1) {
      p->g_red[p->g_nred].size = xd[k];
      p->g_red[p->g_nred].xstride = xs[k];
      p->g_red[p->g_nred].rstride = 0;
      ++p->g_nred;
      red_prod *= xd[k];
    }
  }
  p->red_prod = red_prod;

  if (!(mag_tensor_is_contiguous(r) && mag_tensor_is_contiguous(x))) {
    p->mode = MAG_RB_GENERAL;
    return;
  }

  int64_t gsize[MAG_MAX_DIMS];
  bool gkept[MAG_MAX_DIMS];
  int64_t ng = 0;
  for (int64_t k=0; k < rx; ++k) {
    int64_t kd = k - delta;
    bool kept = (kd >= 0) && (rd[kd] > 1);
    int64_t sz = xd[k];
    if (sz == 1) continue;
    if (ng > 0 && gkept[ng-1] == kept) gsize[ng-1] *= sz;
    else { gsize[ng] = sz; gkept[ng] = kept; ++ng; }
  }
  if (ng == 0) {
    p->mode = MAG_RB_INNER_KEPT;
    p->inner = 1;
    p->nkept = 1; p->kept[0].size = 1; p->kept[0].xstride = 1; p->kept[0].rstride = 1;
    p->nred = 0; p->red_prod = 1;
    return;
  }
  int64_t gxstride[MAG_MAX_DIMS];
  gxstride[ng-1] = 1;
  for (int64_t j=ng-2; j >= 0; --j) gxstride[j] = gxstride[j+1]*gsize[j+1];
  int64_t grstride[MAG_MAX_DIMS];
  int64_t racc = 1;
  for (int64_t j=ng-1; j >= 0; --j) {
    if (gkept[j]) { grstride[j] = racc; racc *= gsize[j]; }
    else grstride[j] = 0;
  }
  p->nkept = 0; p->nred = 0;
  for (int64_t j=0; j < ng; ++j) {
    if (gkept[j]) {
      p->kept[p->nkept].size = gsize[j];
      p->kept[p->nkept].xstride = gxstride[j];
      p->kept[p->nkept].rstride = grstride[j];
      ++p->nkept;
    } else {
      p->red[p->nred].size = gsize[j];
      p->red[p->nred].xstride = gxstride[j];
      p->red[p->nred].rstride = 0;
      ++p->nred;
    }
  }
  int64_t rp = 1;
  for (int64_t j=0; j < p->nred; ++j) rp *= p->red[j].size;
  p->red_prod = rp;
  if (gkept[ng-1]) { p->mode = MAG_RB_INNER_KEPT; p->inner = p->kept[p->nkept-1].size; }
  else { p->mode = MAG_RB_INNER_REDUCED; p->inner = p->red[p->nred-1].size; }
}

#define mag_gen_stub_repeat_back(T, TF, CVT, RCVT, LDV, STV) \
  static mag_status_t MAG_HOTPROC mag_repeat_back_##TF(mag_error_t *err,const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t rnumel = r->meta.numel; \
    if (mag_unlikely(rnumel == 0)) return MAG_OK; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    const int64_t LANES = MAG_VF32_LANES; \
    mag_rb_plan_t plan; \
    mag_rb_build_plan(&plan, r, x); \
    if (plan.mode == MAG_RB_INNER_KEPT) { \
      int64_t chunk = (rnumel + tc - 1)/tc; \
      int64_t ra = ti*chunk; \
      int64_t rb = mag_xmin(ra + chunk, rnumel); \
      if (ra >= rb) return MAG_OK; \
      if (plan.red_prod == 1) { \
        for (int64_t i=ra; i < rb; ++i) br[i] = bx[i]; \
        return MAG_OK; \
      } \
      int64_t IN = plan.inner; \
      size_t mark = mag_scratch_arena_mark(&mag_tls_arena); \
      float *accf = mag_scratch_arena_alloc(&mag_tls_arena, (size_t)IN*sizeof(float)); \
      if (mag_unlikely(!accf)) return mag_set_error(err, MAG_ERR_OOM, "repeat_back: failed to allocate %" PRIi64 " f32 accumulators.", (int64_t)IN); \
      int64_t ob0 = ra / IN; \
      int64_t ob1 = (rb - 1) / IN; \
      for (int64_t oo=ob0; oo <= ob1; ++oo) { \
        int64_t blk_lo = oo*IN; \
        int64_t c0 = mag_xmax(ra, blk_lo) - blk_lo; \
        int64_t c1 = mag_xmin(rb, blk_lo + IN) - blk_lo; \
        int64_t w = c1 - c0; \
        int64_t x_base = 0; \
        { int64_t tmp = oo; for (int64_t g=plan.nkept-2; g >= 0; --g) { int64_t idx = tmp % plan.kept[g].size; tmp /= plan.kept[g].size; x_base += idx*plan.kept[g].xstride; } } \
        for (int64_t c=0; c < w; ++c) accf[c] = .0f; \
        for (int64_t j=0; j < plan.red_prod; ++j) { \
          int64_t redoff = 0; \
          { int64_t rt = j; for (int64_t g=plan.nred-1; g >= 0; --g) { int64_t idx = rt % plan.red[g].size; rt /= plan.red[g].size; redoff += idx*plan.red[g].xstride; } } \
          const T *restrict xp = bx + x_base + redoff + c0; \
          int64_t c = 0; \
          for (; c + LANES <= w; c += LANES) mag_vf32_storeu(accf+c, mag_vf32_add(mag_vf32_loadu(accf+c), LDV(xp+c))); \
          for (; c < w; ++c) accf[c] += CVT(xp[c]); \
        } \
        T *restrict rp = br + blk_lo + c0; \
        int64_t c = 0; \
        for (; c + LANES <= w; c += LANES) STV(rp+c, mag_vf32_loadu(accf+c)); \
        for (; c < w; ++c) rp[c] = RCVT(accf[c]); \
      } \
      mag_scratch_arena_reset(&mag_tls_arena, mark); \
      return MAG_OK; \
    } \
    if (plan.mode == MAG_RB_INNER_REDUCED) { \
      int64_t IN = plan.inner; \
      int64_t red_outer = plan.red_prod / IN; \
      int64_t chunk = (rnumel + tc - 1)/tc; \
      int64_t ra = ti*chunk; \
      int64_t rb = mag_xmin(ra + chunk, rnumel); \
      for (int64_t o=ra; o < rb; ++o) { \
        int64_t x_base = 0, r_off = 0; \
        { int64_t tmp = o; for (int64_t g=plan.nkept-1; g >= 0; --g) { int64_t idx = tmp % plan.kept[g].size; tmp /= plan.kept[g].size; x_base += idx*plan.kept[g].xstride; r_off += idx*plan.kept[g].rstride; } } \
        mag_vf32_t vacc = mag_vf32_zero(); \
        float sacc = .0f; \
        for (int64_t jo=0; jo < red_outer; ++jo) { \
          int64_t redoff = 0; \
          { int64_t rt = jo; for (int64_t g=plan.nred-2; g >= 0; --g) { int64_t idx = rt % plan.red[g].size; rt /= plan.red[g].size; redoff += idx*plan.red[g].xstride; } } \
          const T *restrict xp = bx + x_base + redoff; \
          int64_t c = 0; \
          for (; c + LANES <= IN; c += LANES) vacc = mag_vf32_add(vacc, LDV(xp+c)); \
          for (; c < IN; ++c) sacc += CVT(xp[c]); \
        } \
        br[r_off] = RCVT(mag_vf32_reduce_add(vacc) + sacc); \
      } \
      return MAG_OK; \
    } \
    { \
      int64_t chunk = (rnumel + tc - 1)/tc; \
      int64_t ra = ti*chunk; \
      int64_t rb = mag_xmin(ra + chunk, rnumel); \
      for (int64_t oi=ra; oi < rb; ++oi) { \
        int64_t r_off = 0, x_base = 0; \
        { int64_t tmp = oi; for (int64_t kd=plan.g_rank-1; kd >= 0; --kd) { int64_t dim = plan.g_rshape[kd]; int64_t ax = tmp % dim; tmp /= dim; r_off += ax*plan.g_rstride[kd]; x_base += ax*plan.g_xkept[kd]; } } \
        float acc = .0f; \
        for (int64_t j=0; j < plan.red_prod; ++j) { \
          int64_t redoff = 0; \
          { int64_t rt = j; for (int64_t g=plan.g_nred-1; g >= 0; --g) { int64_t idx = rt % plan.g_red[g].size; rt /= plan.g_red[g].size; redoff += idx*plan.g_red[g].xstride; } } \
          mag_bnd_chk(bx+x_base+redoff, x->storage->base, mag_tensor_numbytes(x)); \
          acc += CVT(bx[x_base+redoff]); \
        } \
        mag_bnd_chk(br+r_off, r->storage->base, mag_tensor_numbytes(r)); \
        br[r_off] = RCVT(acc); \
      } \
      return MAG_OK; \
    } \
  }

mag_gen_stub_repeat_back(float, float32, mag_cvt_nop, mag_cvt_nop, mag_vf32_loadu, mag_vf32_storeu)
mag_gen_stub_repeat_back(mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16, mag_vf32_loadu_f16, mag_vf32_storeu_f16)
mag_gen_stub_repeat_back(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16, mag_vf32_storeu_bf16)
mag_gen_stub_repeat_back(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn, mag_vf32_loadu_float8_e4m3fn, mag_vf32_storeu_float8_e4m3fn)

#undef mag_gen_stub_repeat_back

#define mag_gen_stub_tri_mask(T, TF, S, Z, CMP) \
  static mag_status_t MAG_HOTPROC mag_tri##S##_##TF(mag_error_t *err,const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    mag_coords_iter_t cr, cx; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    mag_coords_iter_init(&cx, &x->meta.coords); \
    int64_t diag = payload->cmd->params->trilu.diag; \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    int64_t cols = r->meta.coords.shape[r->meta.coords.rank-1]; \
    int64_t rows = r->meta.coords.shape[r->meta.coords.rank-2]; \
    int64_t mat = rows*cols; \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t inner = i % mat; \
      int64_t row = inner / cols; \
      int64_t col = inner - row*cols; \
      int64_t ri, xi; \
      mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
      mag_bnd_chk(bx+xi, x->storage->base, mag_tensor_numbytes(x)); \
      mag_bnd_chk(br+ri, r->storage->base, mag_tensor_numbytes(r)); \
      br[ri] = ((col-row) CMP diag) ? bx[xi] : (Z); \
    }  \
    return MAG_OK; \
  }

mag_gen_stub_tri_mask(float, float32, l, 0.f, <=)
mag_gen_stub_tri_mask(mag_float16_t, float16, l, MAG_FLOAT16_ZERO, <=)
mag_gen_stub_tri_mask(mag_bfloat16_t, bfloat16, l, MAG_BFLOAT16_ZERO, <=)
mag_gen_stub_tri_mask(mag_float8_e4m3fn_t, float8_e4m3fn, l, MAG_FLOAT8_E4M3FN_ZERO, <=)
mag_gen_stub_tri_mask(uint8_t, uint8, l, 0, <=)
mag_gen_stub_tri_mask(int8_t, int8, l, 0, <=)
mag_gen_stub_tri_mask(uint16_t, uint16, l, 0, <=)
mag_gen_stub_tri_mask(int16_t, int16, l, 0, <=)
mag_gen_stub_tri_mask(uint32_t, uint32, l, 0, <=)
mag_gen_stub_tri_mask(int32_t, int32, l, 0, <=)
mag_gen_stub_tri_mask(uint64_t, uint64, l, 0, <=)
mag_gen_stub_tri_mask(int64_t, int64, l, 0, <=)

mag_gen_stub_tri_mask(float, float32, u, 0.f, >=)
mag_gen_stub_tri_mask(mag_float16_t, float16, u, MAG_FLOAT16_ZERO, >=)
mag_gen_stub_tri_mask(mag_bfloat16_t, bfloat16, u, MAG_BFLOAT16_ZERO, >=)
mag_gen_stub_tri_mask(mag_float8_e4m3fn_t, float8_e4m3fn, u, MAG_FLOAT8_E4M3FN_ZERO, >=)
mag_gen_stub_tri_mask(uint8_t, uint8, u, 0, >=)
mag_gen_stub_tri_mask(int8_t, int8, u, 0, >=)
mag_gen_stub_tri_mask(uint16_t, uint16, u, 0, >=)
mag_gen_stub_tri_mask(int16_t, int16, u, 0, >=)
mag_gen_stub_tri_mask(uint32_t, uint32, u, 0, >=)
mag_gen_stub_tri_mask(int32_t, int32, u, 0, >=)
mag_gen_stub_tri_mask(uint64_t, uint64, u, 0, >=)
mag_gen_stub_tri_mask(int64_t, int64, u, 0, >=)

#undef mag_gen_stub_tri_mask

#define mag_gen_stub_topk(T, TF, CVT) \
  static MAG_HOTPROC mag_status_t mag_topk_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    mag_tensor_t *v = payload->cmd->out[0]; \
    mag_tensor_t *idx = payload->cmd->out[1]; \
    int64_t k = payload->cmd->params->topk.k; \
    int64_t dim = payload->cmd->params->topk.dim; \
    bool largest = payload->cmd->params->topk.largest; \
    bool sorted = payload->cmd->params->topk.sorted; \
    (void)sorted; /* This implementation always emits sorted top-k, which is valid for sorted=false too. */ \
    const int64_t R = x->meta.coords.rank; \
    mag_assert2(R > 0); \
    mag_assert2(dim >= 0 && dim < R); \
    const int64_t *shape_x = x->meta.coords.shape; \
    const int64_t *shape_v = v->meta.coords.shape; \
    const int64_t *shape_i = idx->meta.coords.shape; \
    const int64_t dim_size = shape_x[dim]; \
    mag_assert2(k > 0 && k <= dim_size); \
    for (int64_t d=0; d < R; ++d) { \
      const int64_t expected = (d == dim) ? k : shape_x[d]; \
      mag_assert2(shape_v[d] == expected); \
      mag_assert2(shape_i[d] == expected); \
    } \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    T *bv = (T *)mag_tensor_data_ptr_mut(v); \
    int64_t *bi = (int64_t *)mag_tensor_data_ptr_mut(idx); \
    const int64_t tc = payload->thread_num; \
    const int64_t ti = payload->thread_idx; \
    const int64_t outer_count = x->meta.numel / dim_size; \
    if (outer_count <= 0) return MAG_OK; \
    const int64_t stride_x_dim = x->meta.coords.strides[dim]; \
    const int64_t stride_v_dim = v->meta.coords.strides[dim]; \
    const int64_t outer_rank = R - 1; \
    int64_t shape_outer[MAG_MAX_DIMS]; \
    int64_t mult_outer[MAG_MAX_DIMS]; \
    int64_t outer_to_full[MAG_MAX_DIMS]; \
    { \
      int64_t t = 0; \
      for (int64_t d=0; d < R; ++d) { \
        if (d == dim) continue; \
        shape_outer[t] = shape_x[d]; \
        outer_to_full[t] = d; \
        ++t; \
      } \
      for (int64_t t2=0; t2 < outer_rank; ++t2) { \
        int64_t m = 1; \
        for (int64_t k2=t2+1; k2 < outer_rank; ++k2) { \
          m *= shape_outer[k2]; \
        } \
        mult_outer[t2] = m; \
      } \
    } \
    const int64_t chunk = (outer_count + tc - 1) / tc; \
    const int64_t oa = ti * chunk; \
    const int64_t ob = mag_xmin(oa + chunk, outer_count); \
    for (int64_t row=oa; row < ob; ++row) { \
      size_t mark = mag_scratch_arena_mark(&mag_tls_arena); \
      int64_t base_idx[MAG_MAX_DIMS]; \
      for (int64_t d=0; d < R; ++d) base_idx[d] = 0; \
      int64_t rtmp = row; \
      for (int64_t t=0; t < outer_rank; ++t) { \
        const int64_t q = (mult_outer[t] == 0) ? 0 : (rtmp / mult_outer[t]); \
        if (mult_outer[t] != 0) rtmp %= mult_outer[t]; \
        base_idx[outer_to_full[t]] = q; \
      } \
      base_idx[dim] = 0; \
      int64_t off_x0 = 0; \
      int64_t off_v0 = 0; \
      for (int64_t d=0; d < R; ++d) { \
        off_x0 += base_idx[d] * x->meta.coords.strides[d]; \
        off_v0 += base_idx[d] * v->meta.coords.strides[d]; \
      } \
      T *best_vals = mag_scratch_arena_alloc(&mag_tls_arena, (size_t)k * sizeof(*best_vals)); \
      int64_t *best_idx = mag_scratch_arena_alloc(&mag_tls_arena, (size_t)k * sizeof(*best_idx)); \
      if (mag_unlikely(!best_vals || !best_idx)) \
        return mag_set_error(err, MAG_ERR_OOM, "topk: failed to allocate scratch buffer for k=%" PRIi64 ".", (int64_t)k); \
      int64_t filled = 0; \
      \
      for (int64_t p=0; p < dim_size; ++p) { \
        const int64_t off_x = off_x0 + p * stride_x_dim; \
        mag_bnd_chk(bx + off_x, x->storage->base, mag_tensor_numbytes(x)); \
        const T xv = bx[off_x]; \
        const double xvc = (double)CVT(xv); \
        if (filled < k) { \
          int64_t ins = filled; \
          while (ins > 0) { \
            const double prevc = (double)CVT(best_vals[ins - 1]); \
            const int64_t previ = best_idx[ins - 1]; \
            bool better; \
            if (largest) better = (xvc > prevc) || ((xvc == prevc) && (p < previ)); \
            else         better = (xvc < prevc) || ((xvc == prevc) && (p < previ)); \
            if (!better) break; \
            best_vals[ins] = best_vals[ins - 1]; \
            best_idx[ins] = best_idx[ins - 1]; \
            --ins; \
          } \
          best_vals[ins] = xv; \
          best_idx[ins] = p; \
          ++filled; \
          continue; \
        } \
        { \
          const double worstc = (double)CVT(best_vals[k - 1]); \
          const int64_t worsti = best_idx[k - 1]; \
          bool better; \
          if (largest) better = (xvc > worstc) || ((xvc == worstc) && (p < worsti)); \
          else         better = (xvc < worstc) || ((xvc == worstc) && (p < worsti)); \
          if (!better) continue; \
        } \
        int64_t ins = k - 1; \
        while (ins > 0) { \
          const double prevc = (double)CVT(best_vals[ins - 1]); \
          const int64_t previ = best_idx[ins - 1]; \
          bool better; \
          if (largest) better = (xvc > prevc) || ((xvc == prevc) && (p < previ)); \
          else         better = (xvc < prevc) || ((xvc == prevc) && (p < previ)); \
          if (!better) break; \
          best_vals[ins] = best_vals[ins - 1]; \
          best_idx[ins] = best_idx[ins - 1]; \
          --ins; \
        } \
        best_vals[ins] = xv; \
        best_idx[ins] = p; \
      } \
      mag_assert2(filled == k); \
      for (int64_t r=0; r < k; ++r) { \
        const int64_t off_v = off_v0 + r * stride_v_dim; \
        mag_bnd_chk(bv + off_v, v->storage->base, mag_tensor_numbytes(v)); \
        mag_bnd_chk(bi + off_v, idx->storage->base, mag_tensor_numbytes(idx)); \
        bv[off_v] = best_vals[r]; \
        bi[off_v] = best_idx[r]; \
      } \
      mag_scratch_arena_reset(&mag_tls_arena, mark); \
    } \
    return MAG_OK; \
  }

mag_gen_stub_topk(float, float32, mag_cvt_nop)
mag_gen_stub_topk(mag_float16_t, float16, mag_float16_to_float32)
mag_gen_stub_topk(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32)
mag_gen_stub_topk(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32)
mag_gen_stub_topk(uint8_t, uint8, mag_cvt_nop)
mag_gen_stub_topk(int8_t, int8, mag_cvt_nop)
mag_gen_stub_topk(uint16_t, uint16, mag_cvt_nop)
mag_gen_stub_topk(int16_t, int16, mag_cvt_nop)
mag_gen_stub_topk(uint32_t, uint32, mag_cvt_nop)
mag_gen_stub_topk(int32_t, int32, mag_cvt_nop)
mag_gen_stub_topk(uint64_t, uint64, mag_cvt_nop)
mag_gen_stub_topk(int64_t, int64, mag_cvt_nop)

#undef mag_gen_stub_topk

#define mag_gen_stub_where(T, TF) \
  static mag_status_t MAG_HOTPROC mag_where_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *cond = payload->cmd->in[0]; \
    const mag_tensor_t *x = payload->cmd->in[1]; \
    const mag_tensor_t *y = payload->cmd->in[2]; \
    mag_assert2(cond->meta.dtype == MAG_DTYPE_BOOLEAN); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const uint8_t *bc = (const uint8_t *)mag_tensor_data_ptr(cond); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *by = (const T *)mag_tensor_data_ptr(y); \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t total = r->meta.numel; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    mag_coords_iter_t cr, cc, cx, cy; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    mag_coords_iter_init(&cc, &cond->meta.coords); \
    mag_coords_iter_init(&cx, &x->meta.coords); \
    mag_coords_iter_init(&cy, &y->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri, ci, xi, yi; \
      mag_coords_iter_offset4(&cr, &cc, &cx, &cy, i, &ri, &ci, &xi, &yi); \
      mag_bnd_chk(bc+ci, cond->storage->base, mag_tensor_numbytes(cond)); \
      mag_bnd_chk(bx+xi, x->storage->base, mag_tensor_numbytes(x)); \
      mag_bnd_chk(by+yi, y->storage->base, mag_tensor_numbytes(y)); \
      mag_bnd_chk(br+ri, r->storage->base, mag_tensor_numbytes(r)); \
      br[ri] = bc[ci] ? bx[xi] : by[yi]; \
    } \
    return MAG_OK; \
  }

mag_gen_stub_where(float, float32)
mag_gen_stub_where(mag_float16_t, float16)
mag_gen_stub_where(mag_bfloat16_t, bfloat16)
mag_gen_stub_where(mag_float8_e4m3fn_t, float8_e4m3fn)
mag_gen_stub_where(uint8_t, uint8)
mag_gen_stub_where(int8_t, int8)
mag_gen_stub_where(uint16_t, uint16)
mag_gen_stub_where(int16_t, int16)
mag_gen_stub_where(uint32_t, uint32)
mag_gen_stub_where(int32_t, int32)
mag_gen_stub_where(uint64_t, uint64)
mag_gen_stub_where(int64_t, int64)

#undef mag_gen_stub_where

#define mag_gen_stub_clamp_cvt(T, TF, CVT, FROMF32) \
  static mag_status_t MAG_HOTPROC mag_clamp_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *mn = payload->cmd->in[1]; \
    const mag_tensor_t *mx = payload->cmd->in[2]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *bmn = (const T *)mag_tensor_data_ptr(mn); \
    const T *bmx = (const T *)mag_tensor_data_ptr(mx); \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t total = r->meta.numel; \
    int64_t chunk = (total + tc - 1) / tc; \
    int64_t ra = ti * chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    mag_coords_iter_t cr, cx, cmn, cmx; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    mag_coords_iter_init(&cx, &x->meta.coords); \
    mag_coords_iter_init(&cmn, &mn->meta.coords); \
    mag_coords_iter_init(&cmx, &mx->meta.coords); \
    for (int64_t i = ra; i < rb; ++i) { \
      int64_t ri, xi, mni, mxi; \
      mag_coords_iter_offset4(&cr, &cx, &cmn, &cmx, i, &ri, &xi, &mni, &mxi); \
      float v = CVT(bx[xi]); \
      float lo = CVT(bmn[mni]); \
      float hi = CVT(bmx[mxi]); \
      float o = v < lo ? lo : (v > hi ? hi : v); \
      br[ri] = FROMF32(o); \
    } \
    return MAG_OK; \
  }

mag_gen_stub_clamp_cvt(float, float32, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_clamp_cvt(mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_stub_clamp_cvt(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gen_stub_clamp_cvt(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)

#undef mag_gen_stub_clamp_cvt

#define mag_gen_stub_clamp_ord(T, TF) \
  static mag_status_t MAG_HOTPROC mag_clamp_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *mn = payload->cmd->in[1]; \
    const mag_tensor_t *mx = payload->cmd->in[2]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *bmn = (const T *)mag_tensor_data_ptr(mn); \
    const T *bmx = (const T *)mag_tensor_data_ptr(mx); \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t total = r->meta.numel; \
    int64_t chunk = (total + tc - 1) / tc; \
    int64_t ra = ti * chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    mag_coords_iter_t cr, cx, cmn, cmx; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    mag_coords_iter_init(&cx, &x->meta.coords); \
    mag_coords_iter_init(&cmn, &mn->meta.coords); \
    mag_coords_iter_init(&cmx, &mx->meta.coords); \
    for (int64_t i = ra; i < rb; ++i) { \
      int64_t ri, xi, mni, mxi; \
      mag_coords_iter_offset4(&cr, &cx, &cmn, &cmx, i, &ri, &xi, &mni, &mxi); \
      T v = bx[xi]; \
      T lo = bmn[mni]; \
      T hi = bmx[mxi]; \
      br[ri] = v < lo ? lo : (v > hi ? hi : v); \
    } \
    return MAG_OK; \
  }

mag_gen_stub_clamp_ord(uint8_t, uint8)
mag_gen_stub_clamp_ord(int8_t, int8)
mag_gen_stub_clamp_ord(uint16_t, uint16)
mag_gen_stub_clamp_ord(int16_t, int16)
mag_gen_stub_clamp_ord(uint32_t, uint32)
mag_gen_stub_clamp_ord(int32_t, int32)
mag_gen_stub_clamp_ord(uint64_t, uint64)
mag_gen_stub_clamp_ord(int64_t, int64)

typedef struct mag_discrete_sample_pair_t {
  float score;
  int64_t idx;
} mag_discrete_sample_pair_t;

static int mag_discrete_sample_pair_cmp(const void *a, const void *b) {
  const mag_discrete_sample_pair_t *A = a;
  const mag_discrete_sample_pair_t *B = b;
  return A->score < B->score ? 1 : A->score > B->score ? -1 : 0;
}

#define mag_gen_stub_multinomial(T, TF, CVT) \
  static mag_status_t MAG_HOTPROC mag_multinomial_##TF(mag_error_t *err,const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    mag_assert2(r->meta.dtype == MAG_DTYPE_INT64); \
    int64_t *br = (int64_t *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t num_samples = payload->cmd->params->multinomial.samples; \
    mag_philox4x32_stream_t *rng = payload->prng; \
    int64_t K = x->meta.coords.shape[x->meta.coords.rank-1]; \
    if (mag_unlikely(K <= 0)) return MAG_OK; \
    int64_t B = x->meta.numel / K; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (B + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, B); \
    for (int64_t b=ra; b < rb; ++b) { \
      const T *w = bx + b*K; \
      int64_t *o = br + b*num_samples; \
      float sumw = .0f; \
      int64_t nnz = 0; \
      for (int64_t i=0; i < K; ++i) { \
        float wi = CVT(w[i]); \
        if (!isfinite(wi) || wi <= .0f) wi = .0f; \
        sumw += wi; \
        if (wi > .0f) ++nnz; \
      } \
      if (!(sumw > .0f) || nnz == 0) { \
        for (int64_t s=0; s < num_samples; ++s) o[s] = -1; \
        continue; \
      } \
      int64_t k = num_samples; \
      if (k > nnz) k = nnz; \
      if (mag_unlikely(k <= 0)) { \
        for (int64_t s=0; s < num_samples; ++s) o[s] = -1; \
        continue; \
      } \
      size_t mark = mag_scratch_arena_mark(&mag_tls_arena); \
      mag_discrete_sample_pair_t *arr = mag_scratch_arena_alloc(&mag_tls_arena, (size_t)nnz*sizeof(*arr)); \
      if (mag_unlikely(!arr)) \
        return mag_set_error(err, MAG_ERR_OOM, "multinomial: failed to allocate scratch buffer for %" PRIi64 " entries.", (int64_t)nnz); \
      int64_t m=0; \
      for (int64_t i=0; i < K; ++i) { \
        float wi = CVT(w[i]); \
        if (mag_unlikely(!isfinite(wi) || wi <= .0f)) continue; \
        float u = mag_philox4x32_next_float32(rng); \
        float g = -logf(-logf(u)); \
        arr[m].score = logf(wi) + g; \
        arr[m].idx = i; \
        ++m; \
      } \
      qsort(arr, (size_t)m, sizeof(*arr), mag_discrete_sample_pair_cmp); \
      for (int64_t s=0; s < k; ++s) o[s] = arr[s].idx; \
      for (int64_t s=k; s < num_samples; ++s) o[s] = -1; \
      mag_scratch_arena_reset(&mag_tls_arena, mark); \
    } \
    return MAG_OK; \
  }

mag_gen_stub_multinomial(float, float32, mag_cvt_nop)
mag_gen_stub_multinomial(mag_float16_t, float16, mag_float16_to_float32)
mag_gen_stub_multinomial(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32)
mag_gen_stub_multinomial(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32)

#undef mag_gen_stub_multinomial

static int64_t mag_pad_reflect_index(int64_t i, int64_t size) {
  if (size <= 1) return 0;
  int64_t period = 2*(size - 1);
  i %= period;
  if (i < 0) i += period;
  if (i >= size) i = period - i;
  return i;
}

static int64_t mag_pad_replicate_index(int64_t i, int64_t size) {
  if (size <= 0) return 0;
  if (i < 0) return 0;
  if (i >= size) return size - 1;
  return i;
}

static int64_t mag_pad_circular_index(int64_t i, int64_t size) {
  if (size <= 0) return 0;
  i %= size;
  if (i < 0) i += size;
  return i;
}

static int64_t mag_pad_map_index(int64_t i, int64_t size, mag_pad_mode_t mode) {
  switch (mode) {
    case MAG_PAD_MODE_REFLECT: return mag_pad_reflect_index(i, size);
    case MAG_PAD_MODE_REPLICATE: return mag_pad_replicate_index(i, size);
    case MAG_PAD_MODE_CIRCULAR: return mag_pad_circular_index(i, size);
    default: mag_panic("pad: invalid mode %d.", (int)mode);
  }
}

#define mag_gen_stub_pad(T, TF, CVT) \
  static MAG_HOTPROC mag_status_t mag_pad_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    T fill = (CVT(payload->cmd->params->pad.value)); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t R = payload->cmd->params->pad.rank; \
    const int64_t *in_shape = x->meta.coords.shape; \
    const int64_t *in_stride = x->meta.coords.strides; \
    const int64_t *out_shape = r->meta.coords.shape; \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    mag_coords_iter_t cr; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri = mag_coords_iter_to_offset(&cr, i); \
      int64_t tmp = i; \
      int64_t oc[MAG_MAX_DIMS]; \
      for (int64_t d = R - 1; d >= 0; --d) { \
        oc[d] = tmp % out_shape[d]; \
        tmp /= out_shape[d]; \
      } \
      bool use_constant = payload->cmd->params->pad.mode == MAG_PAD_MODE_CONSTANT; \
      int64_t si[MAG_MAX_DIMS]; \
      if (payload->cmd->params->pad.mode == MAG_PAD_MODE_CONSTANT) { \
        use_constant = false; \
        for (int64_t d=0; d < R; ++d) { \
          int64_t ic = oc[d] - payload->cmd->params->pad.pad_before[d]; \
          if (ic < 0 || ic >= in_shape[d]) { \
            use_constant = true; \
            break; \
          } \
          si[d] = ic; \
        } \
      } else { \
        for (int64_t d=0; d < R; ++d) { \
          int64_t ic = oc[d] - payload->cmd->params->pad.pad_before[d]; \
          si[d] = mag_pad_map_index(ic, in_shape[d], payload->cmd->params->pad.mode); \
        } \
      } \
      mag_bnd_chk(br+ri, r->storage->base, mag_tensor_numbytes(r)); \
      if (use_constant) { \
        br[ri] = fill; \
      } else { \
        int64_t xi = 0; \
        for (int64_t d=0; d < R; ++d) xi += si[d]*in_stride[d]; \
        mag_bnd_chk(bx+xi, x->storage->base, mag_tensor_numbytes(x)); \
        br[ri] = bx[xi]; \
      } \
    } \
    return MAG_OK; \
  }

mag_gen_stub_pad(float, float32, mag_scalar_to_float32)
mag_gen_stub_pad(mag_float16_t, float16, mag_scalar_to_float16)
mag_gen_stub_pad(mag_bfloat16_t, bfloat16, mag_scalar_to_bfloat16)
mag_gen_stub_pad(mag_float8_e4m3fn_t, float8_e4m3fn, mag_scalar_to_float8_e4m3fn)
mag_gen_stub_pad(uint8_t, uint8, mag_scalar_to_uint8)
mag_gen_stub_pad(int8_t, int8, mag_scalar_to_int8)
mag_gen_stub_pad(uint16_t, uint16, mag_scalar_to_uint16)
mag_gen_stub_pad(int16_t, int16, mag_scalar_to_int16)
mag_gen_stub_pad(uint32_t, uint32, mag_scalar_to_uint32)
mag_gen_stub_pad(int32_t, int32, mag_scalar_to_int32)
mag_gen_stub_pad(uint64_t, uint64, mag_scalar_to_uint64)
mag_gen_stub_pad(int64_t, int64, mag_scalar_to_int64)

#undef mag_gen_stub_pad
