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

static MAG_AINLINE int64_t mag_repeat_in_elem_offset(
  int64_t flat_out,
  const mag_repeat_plan_t *plan,
  const mag_coords_t *in_coords
) {
  int64_t tmp = flat_out;
  int64_t off = 0;
  for (int64_t d = plan->rank - 1; d >= 0; --d) {
    int64_t oc = tmp % plan->out_shape[d];
    tmp /= plan->out_shape[d];
    int64_t ic = oc % plan->in_shape[d];
    int64_t id = d - (plan->rank - plan->in_rank);
    if (id >= 0)
      off += ic * in_coords->strides[id];
  }
  return off;
}

#define mag_gen_stub_repeat(T, TF) \
  static MAG_HOTPROC mag_status_t mag_repeat_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *x = mag_cmd_in(0); \
    const mag_repeat_plan_t *plan = (const mag_repeat_plan_t *)mag_op_attr_unwrap_ptr(mag_cmd_attr(0)); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t total = r->numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    mag_coords_iter_t cr; \
    mag_coords_iter_init(&cr, &r->coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri = mag_coords_iter_to_offset(&cr, i); \
      int64_t xi = mag_repeat_in_elem_offset(i, plan, &x->coords); \
      mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
      mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
      br[ri] = bx[xi]; \
    } \
    return MAG_STATUS_OK; \
  }

mag_gen_stub_repeat(float, float32)
mag_gen_stub_repeat(mag_float16_t, float16)
mag_gen_stub_repeat(mag_bfloat16_t, bfloat16)
mag_gen_stub_repeat(mag_float8_e4m3fn_t, float8_e4m3fn)
mag_gen_stub_repeat(uint8_t, uint8)
mag_gen_stub_repeat(int8_t, int8)
mag_gen_stub_repeat(uint16_t, uint16)
mag_gen_stub_repeat(int16_t, int16)
mag_gen_stub_repeat(uint32_t, uint32)
mag_gen_stub_repeat(int32_t, int32)
mag_gen_stub_repeat(uint64_t, uint64)
mag_gen_stub_repeat(int64_t, int64)

#undef mag_gen_stub_repeat

#define mag_gen_stub_repeat_interleave(T, TF) \
  static MAG_HOTPROC mag_status_t mag_repeat_interleave_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *x = mag_cmd_in(0); \
    const mag_repeat_interleave_plan_t *plan = (const mag_repeat_interleave_plan_t *)mag_op_attr_unwrap_ptr(mag_cmd_attr(0)); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    mag_assert2(mag_tensor_is_contiguous(r) && mag_tensor_is_contiguous(x)); \
    if (plan->flatten) { \
      int64_t n = x->numel; \
      mag_assert2(plan->count_len == 1 || plan->count_len == n); \
      int64_t out_i = 0; \
      for (int64_t i=0; i < n; ++i) { \
        int64_t rep = plan->count_len == 1 ? plan->counts[0] : plan->counts[i]; \
        mag_assert2(rep >= 0); \
        for (int64_t k=0; k < rep; ++k) { \
          mag_bnd_chk(br+out_i, br, mag_tensor_numbytes(r)); \
          mag_bnd_chk(bx+i, bx, mag_tensor_numbytes(x)); \
          br[out_i++] = bx[i]; \
        } \
      } \
      mag_assert2(out_i == r->numel); \
      return MAG_STATUS_OK; \
    } \
    int64_t dim = plan->dim; \
    int64_t R = x->coords.rank; \
    int64_t inner_block = 1; \
    for (int64_t d = dim+1; d < R; ++d) inner_block *= x->coords.shape[d]; \
    int64_t outer_count = 1; \
    for (int64_t d=0; d < dim; ++d) outer_count *= x->coords.shape[d]; \
    int64_t axis_len = x->coords.shape[dim]; \
    mag_assert2(plan->count_len == 1 || plan->count_len == axis_len); \
    int64_t mult[MAG_MAX_DIMS]; \
    for (int64_t d = 0; d < dim; ++d) { \
      int64_t m = 1; \
      for (int64_t k = d + 1; k < dim; ++k) m *= x->coords.shape[k]; \
      mult[d] = m; \
    } \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (outer_count + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_xmin(oa + chunk, outer_count); \
    for (int64_t p=oa; p < ob; ++p) { \
      int64_t idx_prefix[MAG_MAX_DIMS]; \
      int64_t rtmp = p; \
      for (int64_t d = 0; d < dim; ++d) { \
        int64_t q = !mult[d] ? 0 : rtmp/mult[d]; \
        if (mult[d] != 0) rtmp = rtmp%mult[d]; \
        idx_prefix[d] = q; \
      } \
      int64_t moff = 0; \
      for (int64_t d=0; d < dim; ++d) moff += idx_prefix[d]*r->coords.strides[d]; \
      int64_t smoff = 0; \
      for (int64_t d=0; d < dim; ++d) smoff += idx_prefix[d]*x->coords.strides[d]; \
      int64_t cur = 0; \
      for (int64_t a=0; a < axis_len; ++a) { \
        int64_t rep = plan->count_len == 1 ? plan->counts[0] : plan->counts[a]; \
        mag_assert2(rep >= 0); \
        int64_t oel = moff + cur*r->coords.strides[dim]; \
        int64_t sel = smoff + a*x->coords.strides[dim]; \
        const T *restrict src_ptr = bx + sel; \
        T *restrict dst_ptr = br + oel; \
        mag_bnd_chk(bx + sel, bx, mag_tensor_numbytes(x)); \
        mag_bnd_chk(br + oel, br, mag_tensor_numbytes(r)); \
        for (int64_t k=0; k < rep; ++k) { \
          mag_bnd_chk(dst_ptr + k*inner_block, br, mag_tensor_numbytes(r)); \
          memcpy(dst_ptr + k*inner_block, src_ptr, (size_t)inner_block*sizeof(T)); \
        } \
        cur += rep; \
      } \
    } \
    return MAG_STATUS_OK; \
  }

mag_gen_stub_repeat_interleave(float, float32)
mag_gen_stub_repeat_interleave(mag_float16_t, float16)
mag_gen_stub_repeat_interleave(mag_bfloat16_t, bfloat16)
mag_gen_stub_repeat_interleave(mag_float8_e4m3fn_t, float8_e4m3fn)
mag_gen_stub_repeat_interleave(uint8_t, uint8)
mag_gen_stub_repeat_interleave(int8_t, int8)
mag_gen_stub_repeat_interleave(uint16_t, uint16)
mag_gen_stub_repeat_interleave(int16_t, int16)
mag_gen_stub_repeat_interleave(uint32_t, uint32)
mag_gen_stub_repeat_interleave(int32_t, int32)
mag_gen_stub_repeat_interleave(uint64_t, uint64)
mag_gen_stub_repeat_interleave(int64_t, int64)

#undef mag_gen_stub_repeat_interleave
