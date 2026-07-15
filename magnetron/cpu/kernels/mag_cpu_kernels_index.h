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

#define mag_gen_stub_gather(T, TF) \
  static MAG_HOTPROC mag_status_t mag_gather_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *src = payload->cmd->in[0]; \
    const mag_tensor_t *index = payload->cmd->in[1]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(src); \
    const int64_t *bi = (const int64_t *)mag_tensor_data_ptr(index); \
    int64_t axis = payload->cmd->params->gather.dim; \
    if (axis < 0) axis += src->meta.coords.rank; \
    int64_t ax = src->meta.coords.shape[axis]; \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    int64_t inner = 1; \
    for (int64_t d = axis+1; d < src->meta.coords.rank; ++d) inner *= src->meta.coords.shape[d]; \
    int64_t out_ax = r->meta.coords.shape[axis]; \
    if (mag_likely(mag_tensor_is_contiguous(src) && mag_tensor_is_contiguous(r) && mag_tensor_is_contiguous(index))) { \
      int64_t cur_k, cur_j, cur_o; \
      { int64_t tmp = ra; cur_k = tmp % inner; tmp /= inner; cur_j = tmp % out_ax; cur_o = tmp / out_ax; } \
      if (inner == 1) { \
        for (int64_t flat = ra; flat < rb; ++flat) { \
          int64_t g = bi[flat]; \
          if (g < 0) g += ax; \
          if (mag_unlikely(!(g >= 0 && g < ax))) { \
            return mag_set_error(err, MAG_ERR_KERNEL, "gather: index %" PRIi64 " is out of range [0, %" PRIi64 ").", g, ax); \
          } \
          br[flat] = bx[cur_o * ax + g]; \
          if (++cur_j == out_ax) { cur_j = 0; ++cur_o; } \
        } \
      } else { \
        for (int64_t flat = ra; flat < rb; ++flat) { \
          int64_t g = bi[flat]; \
          if (g < 0) g += ax; \
          if (mag_unlikely(!(g >= 0 && g < ax))) { \
            return mag_set_error(err, MAG_ERR_KERNEL, "gather: index %" PRIi64 " is out of range [0, %" PRIi64 ").", g, ax); \
          } \
          br[flat] = bx[(cur_o * ax + g) * inner + cur_k]; \
          if (++cur_k == inner) { cur_k = 0; if (++cur_j == out_ax) { cur_j = 0; ++cur_o; } } \
        } \
      } \
      return MAG_OK; \
    } \
    int64_t oc[MAG_MAX_DIMS]; \
    for (int64_t flat = ra; flat < rb; ++flat) { \
      int64_t tmp = flat; \
      for (int64_t d = r->meta.coords.rank-1; d >= 0; --d) { oc[d] = tmp % r->meta.coords.shape[d]; tmp /= r->meta.coords.shape[d]; } \
      int64_t index_offset = 0; \
      for (int64_t d = 0; d < index->meta.coords.rank; ++d) index_offset += oc[d] * index->meta.coords.strides[d]; \
      int64_t g = bi[index_offset]; \
      if (g < 0) g += ax; \
      if (mag_unlikely(!(g >= 0 && g < ax))) { \
        return mag_set_error(err, MAG_ERR_KERNEL, "gather: index %" PRIi64 " is out of range [0, %" PRIi64 ").", g, ax); \
      } \
      int64_t src_off = 0, dst_off = 0; \
      for (int64_t d = 0; d < src->meta.coords.rank; ++d) src_off += (d == axis ? g : oc[d]) * src->meta.coords.strides[d]; \
      for (int64_t d = 0; d < r->meta.coords.rank; ++d) dst_off += oc[d] * r->meta.coords.strides[d]; \
      br[dst_off] = bx[src_off]; \
    } \
    return MAG_OK; \
  }

mag_gen_stub_gather(float, float32)
mag_gen_stub_gather(mag_float16_t, float16)
mag_gen_stub_gather(mag_bfloat16_t, bfloat16)
mag_gen_stub_gather(mag_float8_e4m3fn_t, float8_e4m3fn)
mag_gen_stub_gather(uint8_t, uint8)
mag_gen_stub_gather(int8_t, int8)
mag_gen_stub_gather(uint16_t, uint16)
mag_gen_stub_gather(int16_t, int16)
mag_gen_stub_gather(uint32_t, uint32)
mag_gen_stub_gather(int32_t, int32)
mag_gen_stub_gather(uint64_t, uint64)
mag_gen_stub_gather(int64_t, int64)

#undef mag_gen_stub_gather

#define mag_gen_stub_embedding(T, TF) \
  static MAG_HOTPROC mag_status_t mag_embedding_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *weight = payload->cmd->in[0]; \
    const mag_tensor_t *indices = payload->cmd->in[1]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(weight); \
    const int64_t *bi = (const int64_t *)mag_tensor_data_ptr(indices); \
    int64_t vocab_size = weight->meta.coords.shape[0]; \
    int64_t row_size = weight->meta.numel / vocab_size; \
    int64_t n_idx = indices->meta.numel; \
    int64_t total = n_idx * row_size; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_likely(mag_tensor_is_contiguous(weight) && mag_tensor_is_contiguous(indices))) { \
      int64_t row_start = ra / row_size; \
      int64_t row_end   = (rb - 1) / row_size; \
      for (int64_t row = row_start; row <= row_end; ++row) { \
        int64_t g = bi[row]; \
        if (g < 0) g += vocab_size; \
        if (mag_unlikely(!(g >= 0 && g < vocab_size))) { \
          return mag_set_error(err, MAG_ERR_KERNEL, "embedding: index %" PRIi64 " is out of range [0, %" PRIi64 ").", g, vocab_size); \
        } \
        int64_t dst_off = row * row_size; \
        int64_t src_off = g  * row_size; \
        int64_t a = (row == row_start) ? (ra - dst_off) : 0; \
        int64_t b = (row == row_end)   ? (rb - dst_off) : row_size; \
        memcpy(br + dst_off + a, bx + src_off + a, (size_t)(b - a) * sizeof(T)); \
      } \
      return MAG_OK; \
    } \
    int64_t cur_col = ra % row_size; \
    int64_t cur_row = ra / row_size; \
    int64_t cur_g; \
    { int64_t idx_off = 0, tmp = cur_row; \
      for (int64_t d = indices->meta.coords.rank-1; d >= 0; --d) { idx_off += (tmp % indices->meta.coords.shape[d]) * indices->meta.coords.strides[d]; tmp /= indices->meta.coords.shape[d]; } \
      cur_g = bi[idx_off]; if (cur_g < 0) cur_g += vocab_size; \
      if (mag_unlikely(!(cur_g >= 0 && cur_g < vocab_size))) { \
        return mag_set_error(err, MAG_ERR_KERNEL, "embedding: index %" PRIi64 " is out of range [0, %" PRIi64 ").", cur_g, vocab_size); \
      } \
    } \
    for (int64_t flat = ra; flat < rb; ++flat) { \
      int64_t w_off = cur_g * weight->meta.coords.strides[0]; \
      { int64_t tmp2 = cur_col; \
        for (int64_t d = weight->meta.coords.rank-1; d >= 1; --d) { w_off += (tmp2 % weight->meta.coords.shape[d]) * weight->meta.coords.strides[d]; tmp2 /= weight->meta.coords.shape[d]; } \
      } \
      br[flat] = bx[w_off]; \
      if (++cur_col == row_size) { \
        cur_col = 0; ++cur_row; \
        if (mag_likely(cur_row < n_idx)) { \
          int64_t idx_off = 0, tmp = cur_row; \
          for (int64_t d = indices->meta.coords.rank-1; d >= 0; --d) { idx_off += (tmp % indices->meta.coords.shape[d]) * indices->meta.coords.strides[d]; tmp /= indices->meta.coords.shape[d]; } \
          cur_g = bi[idx_off]; if (cur_g < 0) cur_g += vocab_size; \
          if (mag_unlikely(!(cur_g >= 0 && cur_g < vocab_size))) { \
        return mag_set_error(err, MAG_ERR_KERNEL, "embedding: index %" PRIi64 " is out of range [0, %" PRIi64 ").", cur_g, vocab_size); \
      } \
        } \
      } \
    } \
    return MAG_OK; \
  }

mag_gen_stub_embedding(float, float32)
mag_gen_stub_embedding(mag_float16_t, float16)
mag_gen_stub_embedding(mag_bfloat16_t, bfloat16)
mag_gen_stub_embedding(mag_float8_e4m3fn_t, float8_e4m3fn)

#undef mag_gen_stub_embedding

#define mag_gen_stub_index_add(T, TF, CVT, ACC_T, MUL, FROM_ACC) \
  static MAG_HOTPROC mag_status_t mag_index_add_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)payload; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *src = payload->cmd->in[1]; \
    const mag_tensor_t *idx = payload->cmd->in[2]; \
    T *bs = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(src); \
    const int64_t *bi = (const int64_t *)mag_tensor_data_ptr(idx); \
    int64_t ax = payload->cmd->params->index_add.dim; \
    double alpha = payload->cmd->params->index_add.alpha; \
    if (ax < 0) ax += r->meta.coords.rank; \
    mag_assert2(ax >= 0 && ax < r->meta.coords.rank); \
    int64_t ra = r->meta.coords.rank; \
    int64_t rra = r->meta.coords.shape[ax]; \
    int64_t total = src->meta.numel; \
    if (payload->thread_idx != 0) return MAG_OK; \
    for (int64_t flat=0; flat < total; ++flat) { \
      int64_t tmp = flat; \
      int64_t sc[MAG_MAX_DIMS]; \
      for (int64_t dx = ra-1; dx >= 0; --dx) { \
        sc[dx] = tmp % src->meta.coords.shape[dx]; \
        tmp /= src->meta.coords.shape[dx]; \
      } \
      int64_t j = sc[ax]; \
      int64_t idx_off = j*idx->meta.coords.strides[0]; \
      int64_t g = bi[idx_off]; \
      if (g < 0) g += rra; \
      if (mag_unlikely(!(g >= 0 && g < rra))) { \
        return mag_set_error(err, MAG_ERR_KERNEL, "index_add_: idx %" PRIi64 " is out of range [0, %" PRIi64 ").", g, rra); \
      } \
      int64_t src_off = 0; \
      for (int64_t dx=0; dx < ra; ++dx) src_off += sc[dx]*src->meta.coords.strides[dx]; \
      sc[ax] = g; \
      int64_t dst_off = 0; \
      for (int64_t dx=0; dx < ra; ++dx) dst_off += sc[dx]*r->meta.coords.strides[dx]; \
      mag_bnd_chk(bs+dst_off, r->storage->base, mag_tensor_numbytes(r)); \
      mag_bnd_chk(bx+src_off, src->storage->base, mag_tensor_numbytes(src)); \
      bs[dst_off] = FROM_ACC((ACC_T)(CVT(bs[dst_off])) + (ACC_T)(MUL(CVT(bx[src_off]), alpha))); \
    } \
    return MAG_OK; \
  }

#define mag_index_add_mul_float(x, a) ((double)(x)*(a))
#define mag_index_add_mul_int(x, a) ((int64_t)(x)*(int64_t)(a))

mag_gen_stub_index_add(float, float32, mag_cvt_nop, double, mag_index_add_mul_float, mag_cvt_nop)
mag_gen_stub_index_add(mag_float16_t, float16, mag_float16_to_float32, float, mag_index_add_mul_float, mag_float32_to_float16)
mag_gen_stub_index_add(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, float, mag_index_add_mul_float, mag_float32_to_bfloat16)
mag_gen_stub_index_add(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, float, mag_index_add_mul_float, mag_float32_to_float8_e4m3fn)
mag_gen_stub_index_add(uint8_t, uint8, mag_cvt_nop, int64_t, mag_index_add_mul_int, mag_cvt_nop)
mag_gen_stub_index_add(int8_t, int8, mag_cvt_nop, int64_t, mag_index_add_mul_int, mag_cvt_nop)
mag_gen_stub_index_add(uint16_t, uint16, mag_cvt_nop, int64_t, mag_index_add_mul_int, mag_cvt_nop)
mag_gen_stub_index_add(int16_t, int16, mag_cvt_nop, int64_t, mag_index_add_mul_int, mag_cvt_nop)
mag_gen_stub_index_add(uint32_t, uint32, mag_cvt_nop, int64_t, mag_index_add_mul_int, mag_cvt_nop)
mag_gen_stub_index_add(int32_t, int32, mag_cvt_nop, int64_t, mag_index_add_mul_int, mag_cvt_nop)
mag_gen_stub_index_add(uint64_t, uint64, mag_cvt_nop, int64_t, mag_index_add_mul_int, mag_cvt_nop)
mag_gen_stub_index_add(int64_t, int64, mag_cvt_nop, int64_t, mag_index_add_mul_int, mag_cvt_nop)

#undef mag_index_add_mul_int
#undef mag_index_add_mul_float
#undef mag_gen_stub_index_add

#define mag_gen_stub_scatter_body(T, OP) \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *src = payload->cmd->in[1]; \
    const mag_tensor_t *idx = payload->cmd->in[2]; \
    T *bs = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(src); \
    const int64_t *bi = (const int64_t *)mag_tensor_data_ptr(idx); \
    int64_t axis = payload->cmd->params->scatter.dim; \
    if (axis < 0) axis += r->meta.coords.rank; \
    int64_t rank = idx->meta.coords.rank; \
    int64_t self_ax = r->meta.coords.shape[axis]; \
    int64_t s_axis = idx->meta.coords.shape[axis]; \
    int64_t total = idx->meta.numel; \
    if (mag_unlikely(total == 0)) return MAG_OK; \
    int64_t num_rows = total/s_axis; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (num_rows + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, num_rows); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    int64_t ist = idx->meta.coords.strides[axis]; \
    int64_t xst = src->meta.coords.strides[axis]; \
    int64_t rst = r->meta.coords.strides[axis]; \
    int64_t c[MAG_MAX_DIMS]; \
    for (int64_t row=ra; row < rb; ++row) { \
      int64_t rem = row; \
      for (int64_t d = rank-1; d >= 0; --d) { if (d == axis) continue; c[d] = rem % idx->meta.coords.shape[d]; rem /= idx->meta.coords.shape[d]; } \
      int64_t idx_row = 0, src_row = 0, dst_row = 0; \
      for (int64_t d=0; d < rank; ++d) { if (d == axis) continue; idx_row += c[d]*idx->meta.coords.strides[d]; src_row += c[d]*src->meta.coords.strides[d]; dst_row += c[d]*r->meta.coords.strides[d]; } \
      for (int64_t j=0; j < s_axis; ++j) { \
        int64_t g = bi[idx_row + j*ist]; \
        if (g < 0) g += self_ax; \
        if (mag_unlikely(!(g >= 0 && g < self_ax))) { \
          return mag_set_error(err, MAG_ERR_KERNEL, "scatter: index %" PRIi64 " is out of range [0, %" PRIi64 ").", g, self_ax); \
        } \
        int64_t dst_off = dst_row + g*rst; \
        int64_t src_off = src_row + j*xst; \
        mag_bnd_chk(bs+dst_off, r->storage->base, mag_tensor_numbytes(r)); \
        mag_bnd_chk(bx+src_off, src->storage->base, mag_tensor_numbytes(src)); \
        OP; \
      } \
    } \
    return MAG_OK;

#define mag_gen_stub_scatter(T, TF) \
  static MAG_HOTPROC mag_status_t mag_scatter_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_gen_stub_scatter_body(T, bs[dst_off] = bx[src_off]) \
  }

mag_gen_stub_scatter(float, float32)
mag_gen_stub_scatter(mag_float16_t, float16)
mag_gen_stub_scatter(mag_bfloat16_t, bfloat16)
mag_gen_stub_scatter(mag_float8_e4m3fn_t, float8_e4m3fn)
mag_gen_stub_scatter(uint8_t, uint8)
mag_gen_stub_scatter(int8_t, int8)
mag_gen_stub_scatter(uint16_t, uint16)
mag_gen_stub_scatter(int16_t, int16)
mag_gen_stub_scatter(uint32_t, uint32)
mag_gen_stub_scatter(int32_t, int32)
mag_gen_stub_scatter(uint64_t, uint64)
mag_gen_stub_scatter(int64_t, int64)

#undef mag_gen_stub_scatter

#define mag_gen_stub_scatter_add(T, TF, CVT, ACC_T, FROM_ACC) \
  static MAG_HOTPROC mag_status_t mag_scatter_add_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_gen_stub_scatter_body(T, bs[dst_off] = FROM_ACC((ACC_T)(CVT(bs[dst_off])) + (ACC_T)(CVT(bx[src_off])))) \
  }

mag_gen_stub_scatter_add(float, float32, mag_cvt_nop, double, mag_cvt_nop)
mag_gen_stub_scatter_add(mag_float16_t, float16, mag_float16_to_float32, float, mag_float32_to_float16)
mag_gen_stub_scatter_add(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, float, mag_float32_to_bfloat16)
mag_gen_stub_scatter_add(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, float, mag_float32_to_float8_e4m3fn)
mag_gen_stub_scatter_add(uint8_t, uint8, mag_cvt_nop, int64_t, mag_cvt_nop)
mag_gen_stub_scatter_add(int8_t, int8, mag_cvt_nop, int64_t, mag_cvt_nop)
mag_gen_stub_scatter_add(uint16_t, uint16, mag_cvt_nop, int64_t, mag_cvt_nop)
mag_gen_stub_scatter_add(int16_t, int16, mag_cvt_nop, int64_t, mag_cvt_nop)
mag_gen_stub_scatter_add(uint32_t, uint32, mag_cvt_nop, int64_t, mag_cvt_nop)
mag_gen_stub_scatter_add(int32_t, int32, mag_cvt_nop, int64_t, mag_cvt_nop)
mag_gen_stub_scatter_add(uint64_t, uint64, mag_cvt_nop, int64_t, mag_cvt_nop)
mag_gen_stub_scatter_add(int64_t, int64, mag_cvt_nop, int64_t, mag_cvt_nop)

#undef mag_gen_stub_scatter_add
#undef mag_gen_stub_scatter_body
