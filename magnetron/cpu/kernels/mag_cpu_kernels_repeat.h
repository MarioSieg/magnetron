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
  const mag_op_params_t *plan,
  const mag_coords_t *in_coords
) {
  int64_t tmp = flat_out;
  int64_t off = 0;
  for (int64_t d = plan->repeat.rank - 1; d >= 0; --d) {
    int64_t oc = tmp % plan->repeat.out_shape[d];
    tmp /= plan->repeat.out_shape[d];
    int64_t ic = oc % plan->repeat.in_shape[d];
    int64_t id = d - (plan->repeat.rank - plan->repeat.in_rank);
    if (id >= 0)
      off += ic * in_coords->strides[id];
  }
  return off;
}

#define mag_gen_stub_repeat(T, TF) \
  static MAG_HOTPROC mag_status_t mag_repeat_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_vmin(ra + chunk, total); \
    mag_coords_iter_t cr; \
    mag_coords_iter_init(&cr, &r->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri = mag_coords_iter_to_offset(&cr, i); \
      int64_t xi = mag_repeat_in_elem_offset(i, payload->cmd->params, &x->meta.coords); \
      mag_bnd_chk(br+ri, r->storage->base, r->storage->size); \
      mag_bnd_chk(bx+xi, x->storage->base, x->storage->size); \
      br[ri] = bx[xi]; \
    } \
    return MAG_OK; \
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
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    mag_assert2(mag_tensor_is_contiguous(r) && mag_tensor_is_contiguous(x)); \
    if (payload->cmd->params->repeat_interleave.flatten) { \
      int64_t n = x->meta.numel; \
      mag_assert2(payload->cmd->params->repeat_interleave.count_len == 1 || payload->cmd->params->repeat_interleave.count_len == n); \
      int64_t out_i = 0; \
      for (int64_t i=0; i < n; ++i) { \
        int64_t rep = payload->cmd->params->repeat_interleave.count_len == 1 ? payload->cmd->params->repeat_interleave.counts[0] : payload->cmd->params->repeat_interleave.counts[i]; \
        mag_assert2(rep >= 0); \
        for (int64_t k=0; k < rep; ++k) { \
          mag_bnd_chk(br+out_i, r->storage->base, r->storage->size); \
          mag_bnd_chk(bx+i, x->storage->base, x->storage->size); \
          br[out_i++] = bx[i]; \
        } \
      } \
      mag_assert2(out_i == r->meta.numel); \
      return MAG_OK; \
    } \
    int64_t dim = payload->cmd->params->repeat_interleave.dim; \
    int64_t R = x->meta.coords.rank; \
    int64_t inner_block = 1; \
    for (int64_t d = dim+1; d < R; ++d) inner_block *= x->meta.coords.shape[d]; \
    int64_t outer_count = 1; \
    for (int64_t d=0; d < dim; ++d) outer_count *= x->meta.coords.shape[d]; \
    int64_t axis_len = x->meta.coords.shape[dim]; \
    mag_assert2(payload->cmd->params->repeat_interleave.count_len == 1 || payload->cmd->params->repeat_interleave.count_len == axis_len); \
    int64_t mult[MAG_MAX_DIMS]; \
    for (int64_t d = 0; d < dim; ++d) { \
      int64_t m = 1; \
      for (int64_t k = d + 1; k < dim; ++k) m *= x->meta.coords.shape[k]; \
      mult[d] = m; \
    } \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (outer_count + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_vmin(oa + chunk, outer_count); \
    for (int64_t p=oa; p < ob; ++p) { \
      int64_t idx_prefix[MAG_MAX_DIMS]; \
      int64_t rtmp = p; \
      for (int64_t d = 0; d < dim; ++d) { \
        int64_t q = !mult[d] ? 0 : rtmp/mult[d]; \
        if (mult[d] != 0) rtmp = rtmp%mult[d]; \
        idx_prefix[d] = q; \
      } \
      int64_t moff = 0; \
      for (int64_t d=0; d < dim; ++d) moff += idx_prefix[d]*r->meta.coords.strides[d]; \
      int64_t smoff = 0; \
      for (int64_t d=0; d < dim; ++d) smoff += idx_prefix[d]*x->meta.coords.strides[d]; \
      int64_t cur = 0; \
      for (int64_t a=0; a < axis_len; ++a) { \
        int64_t rep = payload->cmd->params->repeat_interleave.count_len == 1 ? payload->cmd->params->repeat_interleave.counts[0] : payload->cmd->params->repeat_interleave.counts[a]; \
        mag_assert2(rep >= 0); \
        int64_t oel = moff + cur*r->meta.coords.strides[dim]; \
        int64_t sel = smoff + a*x->meta.coords.strides[dim]; \
        const T *restrict src_ptr = bx + sel; \
        T *restrict dst_ptr = br + oel; \
        mag_bnd_chk(bx + sel, x->storage->base, x->storage->size); \
        mag_bnd_chk(br + oel, r->storage->base, r->storage->size); \
        for (int64_t k=0; k < rep; ++k) { \
          mag_bnd_chk(dst_ptr + k*inner_block, r->storage->base, r->storage->size); \
          memcpy(dst_ptr + k*inner_block, src_ptr, (size_t)inner_block*sizeof(T)); \
        } \
        cur += rep; \
      } \
    } \
    return MAG_OK; \
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
