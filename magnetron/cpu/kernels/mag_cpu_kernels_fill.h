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

#define mag_cvt_nop(x) (x)

/* Generate N bernoulli distributed booleans. */
static void MAG_AINLINE mag_vrand_bernoulli_bool(mag_philox4x32_stream_t *prng, int64_t numel, uint8_t *restrict o, float p) {
  if (mag_unlikely(p <= 0.0f)) {
    memset(o, 0, sizeof(*o)*numel);
    return;
  }
  if (mag_unlikely(p >= 1.0f)) {
    for (int64_t i=0; i < numel; ++i) o[i] = 1;
    return;
  }
  uint32_t thresh = (uint32_t)(p*4294967296.f); /* 2^32 */
  int64_t i=0;
  for (; i+3 < numel; i += 4) {
    mag_philox4x32_uint32x4_t r = mag_philox4x32_next_uint32x4(prng);
    for (int j=0; j < 4; ++j)
      o[i+j] = r.v[j] < thresh;
  }
  if (i < numel) {
    mag_philox4x32_uint32x4_t r = mag_philox4x32_next_uint32x4(prng);
    for (int64_t t=0; i < numel; ++i, ++t)
      o[i] = r.v[t] < thresh;
  }
}

#define mag_gen_vrand_uniform_int(T, UT) \
  static MAG_HOTPROC void mag_vrand_uniform_##T(mag_philox4x32_stream_t *prng, int64_t numel, T *restrict o, T min, T max) {                                                                  \
    UT umin = (UT)min; \
    UT umax = (UT)max; \
    uint64_t span64 = (uint64_t)((UT)(umax - umin))+1ull; \
    if (!span64) { \
      for (int64_t i=0; i < numel; ++i) { \
        UT r = (UT)mag_philox4x32_next_uint64(prng); \
        o[i] = (T)r; \
      } \
      return; \
    } \
    if (sizeof(UT) <= 4) { \
      uint32_t span = (uint32_t)span64; \
      uint32_t thresh = (uint32_t)(0u-span)%span; \
      for (int64_t i=0; i < numel; ++i) { \
        for (;;) { \
          uint32_t x = mag_philox4x32_next_uint32(prng); \
          uint64_t m = (uint64_t)x * (uint64_t)span; \
          uint32_t lo = (uint32_t)m; \
          if (mag_unlikely(lo < thresh)) continue; \
          uint32_t hi = (uint32_t)(m>>32); \
          UT v = (UT)((uint32_t)umin + hi); \
          o[i] = (T)v; \
          break; \
        } \
      } \
    } else { \
      uint64_t span = span64; \
      uint64_t thresh = (uint64_t)(0ull-span)%span; \
      for (int64_t i=0; i < numel; ++i) { \
        for (;;) { \
          uint64_t x = mag_philox4x32_next_uint64(prng); \
          mag_uint128_t m = mag_uint128_mul128(x, span); \
          uint64_t lo = mag_uint128_lo(m), hi = mag_uint128_hi(m); \
          if (mag_unlikely(lo < thresh)) continue; \
          UT v = (UT)(umin + hi); \
          o[i] = (T)v; \
          break; \
        } \
      } \
    } \
  }

mag_gen_vrand_uniform_int(uint8_t, uint8_t)
mag_gen_vrand_uniform_int(int8_t, uint8_t)
mag_gen_vrand_uniform_int(uint16_t, uint16_t)
mag_gen_vrand_uniform_int(int16_t, uint16_t)
mag_gen_vrand_uniform_int(uint32_t, uint32_t)
mag_gen_vrand_uniform_int(int32_t, uint32_t)
mag_gen_vrand_uniform_int(uint64_t, uint64_t)
mag_gen_vrand_uniform_int(int64_t, uint64_t)

#undef mag_gen_vrand_uniform_int

#define mag_G(x) (x)                    /* Get scalar value */
#define mag_G_underlying(x) (x.bits)    /* Get underlying storage scalar */

#define mag_gen_stub_fill(T, TF, G, UT, CVT) \
  static MAG_HOTPROC mag_status_t mag_fill_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    T val = (T)CVT(mag_scalar_as_##UT(payload->cmd->params->fill.value)); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra+chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_tensor_is_contiguous(r)) { \
      for (int64_t ri=ra; ri < rb; ++ri) { \
        mag_bnd_chk(br+ri, r->storage->base, r->storage->size); \
        br[ri] = val; \
      } \
      return MAG_OK; \
    } \
    mag_coords_iter_t cr; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri = mag_coords_iter_to_offset(&cr, i); \
      mag_bnd_chk(br+ri, r->storage->base, r->storage->size); \
      br[ri] = val; \
    } \
    return MAG_OK; \
  }

mag_gen_stub_fill(float, float32, mag_G, float64, mag_cvt_nop)
mag_gen_stub_fill(mag_float16_t, float16, mag_G_underlying, float64, mag_float32_to_float16)
mag_gen_stub_fill(mag_bfloat16_t, bfloat16, mag_G_underlying, float64, mag_float32_to_bfloat16)
mag_gen_stub_fill(mag_float8_e4m3fn_t, float8_e4m3fn, mag_G_underlying, float64, mag_float32_to_float8_e4m3fn)
mag_gen_stub_fill(uint8_t, uint8, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_fill(int8_t, int8, mag_G, int64, mag_cvt_nop)
mag_gen_stub_fill(uint16_t, uint16, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_fill(int16_t, int16, mag_G, int64, mag_cvt_nop)
mag_gen_stub_fill(uint32_t, uint32, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_fill(int32_t, int32, mag_G, int64, mag_cvt_nop)
mag_gen_stub_fill(uint64_t, uint64, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_fill(int64_t, int64, mag_G, int64, mag_cvt_nop)

#undef mag_gen_stub_fill

#define mag_gen_stub_masked_fill(T, TF, G, UT, CVT) \
  static MAG_HOTPROC mag_status_t mag_masked_fill_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *mask = payload->cmd->in[1]; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    T val = (T)CVT(mag_scalar_as_##UT(payload->cmd->params->fill.value)); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const uint8_t *bm = (const uint8_t *)mag_tensor_data_ptr(mask); \
    mag_coords_iter_t cr, cx, cm; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    mag_coords_iter_init(&cx, &x->meta.coords); \
    mag_coords_iter_init(&cm, &mask->meta.coords); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra+chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_tensor_is_contiguous(r) && mag_tensor_is_contiguous(x)) { \
      for (int64_t ri=ra; ri < rb; ++ri) { \
        int64_t mi = mag_coords_iter_broadcast(&cr, &cm, ri); \
        mag_bnd_chk(br+ri, r->storage->base, r->storage->size); \
        mag_bnd_chk(bx+ri, x->storage->base, x->storage->size); \
        mag_bnd_chk(bm+mi, mask->storage->base, mask->storage->size); \
        br[ri] = bm[mi] ? val : bx[ri]; \
      } \
      return MAG_OK; \
    } \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri, xi, mi; \
      mag_coords_iter_offset3(&cr, &cx, &cm, i, &ri, &xi, &mi); \
      mag_bnd_chk(br+ri, r->storage->base, r->storage->size); \
      mag_bnd_chk(bx+xi, x->storage->base, x->storage->size); \
      mag_bnd_chk(bm+mi, mask->storage->base, mask->storage->size); \
      br[ri] = bm[mi] ? val : bx[xi]; \
    } \
    return MAG_OK; \
  }

mag_gen_stub_masked_fill(float, float32, mag_G, float64, mag_cvt_nop)
mag_gen_stub_masked_fill(mag_float16_t, float16, mag_G_underlying, float64, mag_float32_to_float16)
mag_gen_stub_masked_fill(mag_bfloat16_t, bfloat16, mag_G_underlying, float64, mag_float32_to_bfloat16)
mag_gen_stub_masked_fill(mag_float8_e4m3fn_t, float8_e4m3fn, mag_G_underlying, float64, mag_float32_to_float8_e4m3fn)
mag_gen_stub_masked_fill(uint8_t, uint8, mag_G, uint64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(int8_t, int8, mag_G, int64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(uint16_t, uint16, mag_G, uint64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(int16_t, int16, mag_G, int64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(uint32_t, uint32, mag_G, uint64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(int32_t, int32, mag_G, int64, mag_cvt_int32_to_int64)
mag_gen_stub_masked_fill(uint64_t, uint64, mag_G, uint64, mag_cvt_nop)
mag_gen_stub_masked_fill(int64_t, int64, mag_G, int64, mag_cvt_nop)

#undef mag_gen_stub_masked_fill

#define mag_gen_stub_fill_rand(D, T, TS, UT, TF) \
  static MAG_HOTPROC mag_status_t mag_fill_rand_##D##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    TS min = (TS)mag_scalar_as_##UT(payload->cmd->params->uniform.low); \
    TS max = (TS)mag_scalar_as_##UT(payload->cmd->params->uniform.high); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    mag_philox4x32_stream_t *prng = payload->prng; \
    mag_coords_iter_t cr; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra + chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_tensor_is_contiguous(r)) { \
      mag_vrand_##D##_##T(prng, rb-ra, br+ra, min, max); \
      return MAG_OK; \
    } \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri = mag_coords_iter_to_offset(&cr, i); \
      mag_bnd_chk(br+ri, r->storage->base, r->storage->size); \
      mag_vrand_##D##_##T(prng, 1, br+ri, min, max); \
    } \
    return MAG_OK; \
  }

mag_gen_stub_fill_rand(uniform, uint8_t, uint64_t, uint64, uint8)
mag_gen_stub_fill_rand(uniform, int8_t, int64_t, int64, int8)
mag_gen_stub_fill_rand(uniform, uint16_t, uint64_t, uint64, uint16)
mag_gen_stub_fill_rand(uniform, int16_t, int64_t, int64, int16)
mag_gen_stub_fill_rand(uniform, uint32_t, uint64_t, uint64, uint32)
mag_gen_stub_fill_rand(uniform, int32_t, int64_t, int64, int32)
mag_gen_stub_fill_rand(uniform, uint64_t, uint64_t, uint64, uint64)
mag_gen_stub_fill_rand(uniform, int64_t, int64_t, int64, int64)

#undef mag_gen_stub_fill_rand

#define mag_gen_stub_fill_rand_fp_simd_only(D, T, TS, UT, TF)                    \
  static MAG_HOTPROC mag_status_t mag_fill_rand_##D##_##TF(                     \
    mag_error_t *err, const mag_kernel_payload_t *payload                        \
  ) {                                                                           \
    (void)err;                                                                  \
    mag_tensor_t *r = payload->cmd->out[0];                                            \
    TS min = (TS)mag_scalar_as_##UT(payload->cmd->params->uniform.low);    /* todo: this works because normal params has name bit layout  ut wtf, fix it */                   \
    TS max = (TS)mag_scalar_as_##UT(payload->cmd->params->uniform.high);                       \
    T *br = (T *)mag_tensor_data_ptr_mut(r);                                     \
    mag_philox4x32_stream_t *prng = payload->prng;                               \
                                                                                \
    int64_t total = r->meta.numel;                                                    \
    int64_t tc = payload->thread_num;                                            \
    int64_t ti = payload->thread_idx;                                            \
                                                                                \
    int64_t block = 4 * MAG_VF32_LANES;                                          \
    int64_t blocks = (total + block - 1) / block;                                \
    int64_t chunk_blocks = (blocks + tc - 1) / tc;                               \
                                                                                \
    int64_t ba = ti * chunk_blocks;                                              \
    int64_t bb = mag_vmin(ba + chunk_blocks, blocks);                            \
    int64_t ra = ba * block;                                                     \
    int64_t rb = mag_vmin(bb * block, total);                                    \
                                                                                \
    if (mag_unlikely(rb <= ra)) return MAG_OK;                            \
                                                                                \
    uint64_t seed = ((uint64_t)prng->key.v[1] << 32) | prng->key.v[0];            \
    uint64_t subseq = ((uint64_t)prng->ctr.v[3] << 32) | prng->ctr.v[2];          \
    uint64_t counter = (uint64_t)ba * MAG_VF32_LANES;                            \
                                                                                \
    if (mag_tensor_is_contiguous(r)) {                                           \
      mag_vrand_##D##_##T##_simd(seed, subseq, counter, rb - ra, br + ra, min, max); \
      return MAG_OK;                                                      \
    }                                                                           \
                                                                                \
    mag_coords_iter_t cr;                                                        \
    mag_coords_iter_init(&cr, &r->meta.coords);                                       \
    for (int64_t i = ra; i < rb; ++i) {                                          \
      int64_t ri = mag_coords_iter_to_offset(&cr, i);                            \
      mag_bnd_chk(br + ri, r->storage->base, r->storage->size);                          \
      uint64_t elem_block = (uint64_t)i / (uint64_t)(4 * MAG_VF32_LANES);         \
      uint64_t elem_off = (uint64_t)i % (uint64_t)(4 * MAG_VF32_LANES);           \
      T tmp[4 * MAG_VF32_LANES];                                                 \
      mag_vrand_##D##_##T##_simd(                                                \
        seed, subseq, elem_block * MAG_VF32_LANES,                               \
        4 * MAG_VF32_LANES, tmp, min, max                                        \
      );                                                                         \
      br[ri] = tmp[elem_off];                                                    \
    }                                                                           \
    return MAG_OK;                                                        \
  }

mag_gen_stub_fill_rand_fp_simd_only(uniform, float, float, float64, float32)
mag_gen_stub_fill_rand_fp_simd_only(uniform, mag_float16_t, float, float64, float16)
mag_gen_stub_fill_rand_fp_simd_only(uniform, mag_bfloat16_t, float, float64, bfloat16)
mag_gen_stub_fill_rand_fp_simd_only(uniform, mag_float8_e4m3fn_t, float, float64, float8_e4m3fn)

mag_gen_stub_fill_rand_fp_simd_only(normal, float, float, float64, float32)
mag_gen_stub_fill_rand_fp_simd_only(normal, mag_float16_t, float, float64, float16)
mag_gen_stub_fill_rand_fp_simd_only(normal, mag_bfloat16_t, float, float64, bfloat16)
mag_gen_stub_fill_rand_fp_simd_only(normal, mag_float8_e4m3fn_t, float, float64, float8_e4m3fn)


#define mag_gen_stub_arange(T, TF, PT, UNWRAP, AT, CVT) \
  static MAG_HOTPROC mag_status_t mag_arange_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    PT start = UNWRAP(payload->cmd->params->arange.start); \
    PT step = UNWRAP(payload->cmd->params->arange.step); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra + chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_tensor_is_contiguous(r)) { \
      for (int64_t ri=ra; ri < rb; ++ri) { \
        mag_bnd_chk(br+ri, r->storage->base, r->storage->size); \
        br[ri] = CVT((AT)start + (AT)ri*(AT)step); \
      } \
      return MAG_OK; \
    } \
    mag_coords_iter_t cr; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri = mag_coords_iter_to_offset(&cr, i); \
      mag_bnd_chk(br+ri, r->storage->base, r->storage->size); \
      br[ri] = CVT((AT)start + (AT)i*(AT)step); \
    } \
    return MAG_OK; \
  }

mag_gen_stub_arange(float, float32, double, mag_scalar_as_float64, double, mag_cvt_nop)
mag_gen_stub_arange(mag_float16_t, float16, double, mag_scalar_as_float64, double, mag_float32_to_float16)
mag_gen_stub_arange(mag_bfloat16_t, bfloat16, double, mag_scalar_as_float64, double, mag_float32_to_bfloat16)
mag_gen_stub_arange(mag_float8_e4m3fn_t, float8_e4m3fn, double, mag_scalar_as_float64, double, mag_float32_to_float8_e4m3fn)
mag_gen_stub_arange(uint8_t, uint8, uint64_t, mag_scalar_as_uint64, uint64_t, mag_cvt_nop)
mag_gen_stub_arange(int8_t, int8, int64_t, mag_scalar_as_int64, uint64_t, mag_cvt_nop)
mag_gen_stub_arange(uint16_t, uint16, uint64_t, mag_scalar_as_uint64, uint64_t, mag_cvt_nop)
mag_gen_stub_arange(int16_t, int16, int64_t, mag_scalar_as_int64, uint64_t, mag_cvt_nop)
mag_gen_stub_arange(uint32_t, uint32, uint64_t, mag_scalar_as_uint64, uint64_t, mag_cvt_nop)
mag_gen_stub_arange(int32_t, int32, int64_t, mag_scalar_as_int64, uint64_t, mag_cvt_nop)
mag_gen_stub_arange(uint64_t, uint64, uint64_t, mag_scalar_as_uint64, uint64_t, mag_cvt_nop)
mag_gen_stub_arange(int64_t, int64, int64_t, mag_scalar_as_int64, uint64_t, mag_cvt_nop)

#undef mag_gen_stub_arange

#define mag_gen_stub_eye(T, TF, ONE, ZERO) \
  static MAG_HOTPROC mag_status_t mag_eye_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    mag_assert2(r->meta.coords.rank == 2); \
    int64_t cols = r->meta.coords.shape[1]; \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra + chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_tensor_is_contiguous(r)) { \
      for (int64_t i=ra; i < rb; ++i) { \
        int64_t row = i / cols; \
        int64_t col = i - row*cols; \
        mag_bnd_chk(br+i, r->storage->base, r->storage->size); \
        br[i] = row == col ? (ONE) : (ZERO); \
      } \
      return MAG_OK; \
    } \
    mag_coords_iter_t cr; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri = mag_coords_iter_to_offset(&cr, i); \
      int64_t row = i / cols; \
      int64_t col = i - row*cols; \
      mag_bnd_chk(br+ri, r->storage->base, r->storage->size); \
      br[ri] = row == col ? (ONE) : (ZERO); \
    } \
    return MAG_OK; \
  }

mag_gen_stub_eye(float, float32, 1.f, 0.f)
mag_gen_stub_eye(mag_float16_t, float16, MAG_FLOAT16_ONE, MAG_FLOAT16_ZERO)
mag_gen_stub_eye(mag_bfloat16_t, bfloat16, MAG_BFLOAT16_ONE, MAG_BFLOAT16_ZERO)
mag_gen_stub_eye(mag_float8_e4m3fn_t, float8_e4m3fn, MAG_FLOAT8_E4M3FN_ONE, MAG_FLOAT8_E4M3FN_ZERO)
mag_gen_stub_eye(uint8_t, uint8, 1, 0)
mag_gen_stub_eye(int8_t, int8, 1, 0)
mag_gen_stub_eye(uint16_t, uint16, 1, 0)
mag_gen_stub_eye(int16_t, int16, 1, 0)
mag_gen_stub_eye(uint32_t, uint32, 1, 0)
mag_gen_stub_eye(int32_t, int32, 1, 0)
mag_gen_stub_eye(uint64_t, uint64, 1, 0)
mag_gen_stub_eye(int64_t, int64, 1, 0)

#undef mag_gen_stub_eye

static MAG_HOTPROC mag_status_t mag_one_hot_int64(mag_error_t *err, const mag_kernel_payload_t *payload) {
  (void)err;
  mag_tensor_t *r = payload->cmd->out[0];
  mag_tensor_t *idx = payload->cmd->in[0];
  int64_t *restrict pr = (int64_t *)mag_tensor_data_ptr_mut(r);
  const int64_t *restrict pidx = (const int64_t *)mag_tensor_data_ptr(idx);
  int64_t nc = payload->cmd->params->one_hot.num_classes;
  int64_t total = idx->meta.numel;
  int64_t tc = payload->thread_num;
  int64_t ti = payload->thread_idx;
  int64_t chunk = (total + tc - 1)/tc;
  int64_t ra = ti*chunk;
  int64_t rb = mag_vmin(ra + chunk, total);
  if (mag_unlikely(rb <= ra)) return MAG_OK;
  if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[2]){r, idx}, 2)) {
    for (int64_t i=ra; i < rb; ++i) {
      int64_t cls = pidx[i];
      if ((uint64_t)cls < (uint64_t)nc) {
        int64_t off = i*nc + cls;
        mag_bnd_chk(pr+off, r->storage->base, r->storage->size);
        pr[off] = 1;
      }
    }
    return MAG_OK;
  }
  mag_coords_iter_t it;
  mag_coords_iter_init(&it, &idx->meta.coords);
  for (int64_t i=ra; i < rb; ++i) {
    int64_t ridx = mag_coords_iter_to_offset(&it, i);
    mag_bnd_chk(pidx+ridx, idx->storage->base, idx->storage->size);
    int64_t cls = pidx[ridx];
    if ((uint64_t)cls < (uint64_t)nc) {
      int64_t off = i*nc + cls;
      mag_bnd_chk(pr+off, r->storage->base, r->storage->size);
      pr[off] = 1;
    }
  }
  return MAG_OK;
}

#undef mag_G
#undef mag_G_underlying

static MAG_HOTPROC mag_status_t mag_fill_rand_bernoulli_bool(mag_error_t *err, const mag_kernel_payload_t *payload) {
  (void)err;
  mag_tensor_t *r = payload->cmd->out[0];
  float p = (float)payload->cmd->params->bernoulli.p;
  uint8_t *b_r = (uint8_t *)mag_tensor_data_ptr_mut(r);
  int64_t numel = r->meta.numel;
  mag_vrand_bernoulli_bool(payload->prng, numel, b_r, p);
  return MAG_OK;
}

#define mag_gen_stub_rand_perm(T, TF, CVT) \
  static MAG_HOTPROC mag_status_t mag_rand_perm_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    int64_t numel = r->meta.numel; \
    mag_philox4x32_stream_t *prng = payload->prng; \
    if (mag_tensor_is_contiguous(r)) { \
      for (int64_t i=0; i < numel; ++i) { \
        mag_bnd_chk(br+i, r->storage->base, r->storage->size); \
        br[i] = CVT((int64_t)i); \
      } \
      for (int64_t i=0; i < numel-1; ++i) { \
        int64_t j = i+(int64_t)(mag_philox4x32_next_uint64(prng)%(uint64_t)(numel-i)); \
        mag_bnd_chk(br+j, r->storage->base, r->storage->size); \
        T tmp = br[i]; \
        br[i] = br[j]; \
        br[j] = tmp; \
      } \
      return MAG_OK; \
    } \
    mag_coords_iter_t it; \
    mag_coords_iter_init(&it, &r->meta.coords); \
    for (int64_t i=0; i < numel; ++i) { \
      int64_t off = mag_coords_iter_to_offset(&it, i); \
      mag_bnd_chk(br+off, r->storage->base, r->storage->size); \
      br[off] = CVT((int64_t)i); \
    } \
    for (int64_t i=0; i < numel-1; ++i) { \
      int64_t j = i+(int64_t)(mag_philox4x32_next_uint64(prng)%(uint64_t)(numel-i)); \
      int64_t off_i = mag_coords_iter_to_offset(&it, i); \
      int64_t off_j = mag_coords_iter_to_offset(&it, j); \
      mag_bnd_chk(br+off_i, r->storage->base, r->storage->size); \
      mag_bnd_chk(br+off_j, r->storage->base, r->storage->size); \
      T tmp = br[off_i]; \
      br[off_i] = br[off_j]; \
      br[off_j] = tmp; \
    } \
    return MAG_OK; \
  }

mag_gen_stub_rand_perm(uint8_t, uint8, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(int8_t, int8, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(uint16_t, uint16, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(int16_t, int16, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(uint32_t, uint32, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(int32_t, int32, mag_cvt_int32_to_int64)
mag_gen_stub_rand_perm(uint64_t, uint64, mag_cvt_nop)
mag_gen_stub_rand_perm(int64_t, int64, mag_cvt_nop)

#undef mag_gen_stub_randperm
