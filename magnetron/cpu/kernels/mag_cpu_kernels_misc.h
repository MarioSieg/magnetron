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
    mag_tensor_t *r = mag_cmd_out(0); \
    const int64_t dim = mag_op_attr_unwrap_int64(mag_cmd_attr(0)); \
    const int64_t n = payload->cmd->num_in; \
    mag_assert2(r && n > 0); \
    mag_assert2(dim >= 0 && dim < r->coords.rank); \
    \
    int64_t R = r->coords.rank; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    mag_assert2(mag_tensor_is_contiguous(r)); \
    \
    int64_t inner_block = 1; \
    for (int64_t d = dim+1; d < R; ++d) inner_block *= r->coords.shape[d]; \
    int64_t outer_count = 1; \
    for (int64_t d=0; d < dim; ++d) outer_count *= r->coords.shape[d]; \
    \
    int64_t mult[MAG_MAX_DIMS]; \
    for (int64_t d = 0; d < dim; ++d) { \
      int64_t m = 1; \
      for (int64_t k = d + 1; k < dim; ++k) m *= r->coords.shape[k]; \
      mult[d] = m; \
    } \
    \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (outer_count + tc - 1)/tc; \
    int64_t oa = ti*chunk; \
    int64_t ob = mag_xmin(oa + chunk, outer_count); \
    \
    for (int64_t p=oa; p < ob; ++p) { \
      int64_t idx_prefix[MAG_MAX_DIMS]; \
      int64_t rtmp = p; \
      for (int64_t d = 0; d < dim; ++d) { \
        int64_t q = !mult[d] ? 0 : rtmp/mult[d]; \
        if (mult[d] != 0) rtmp = rtmp%mult[d]; \
        idx_prefix[d] = q; \
      } \
      \
      int64_t moff = 0; \
      for (int64_t d=0; d < dim; ++d) moff += idx_prefix[d]*r->coords.strides[d]; \
      int64_t cur = 0; \
      \
      for (int64_t i=0; i < n; ++i) { \
        const mag_tensor_t *x = mag_cmd_in(i); \
        int64_t smoff=0; \
        for (int64_t d=0; d < dim; ++d) \
          smoff += idx_prefix[d]*x->coords.strides[d]; \
        int64_t cl = x->coords.shape[dim]; \
        int64_t numel = cl*inner_block; \
        int64_t oel = moff + cur*r->coords.strides[dim]; \
        int64_t sel = smoff; \
        const T *restrict bx = (const T *)mag_tensor_data_ptr(x); \
        const uint8_t *restrict src_ptr = (const uint8_t *)(bx+sel); \
        uint8_t *restrict dst_ptr = (uint8_t *)(br+oel); \
        mag_bnd_chk(bx + sel, bx, mag_tensor_numbytes(x)); \
        mag_bnd_chk(br + oel, br, mag_tensor_numbytes(r)); \
        memcpy(dst_ptr, src_ptr, (size_t)numel*sizeof(T)); \
        cur += cl; \
      } \
    } \
    return MAG_STATUS_OK; \
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

#define mag_gen_stub_repeat_back(T, TF, Z, CVT, RCVT) \
  static mag_status_t MAG_HOTPROC mag_repeat_back_##TF(mag_error_t *err,const mag_kernel_payload_t *payload) { \
    (void)err; \
    if (payload->thread_idx != 0) return MAG_STATUS_OK; \
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *x = mag_cmd_in(0); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    mag_coords_iter_t cr, cx; \
    mag_coords_iter_init(&cr, &r->coords); \
    mag_coords_iter_init(&cx, &x->coords); \
    for (int64_t i=0; i < r->numel; ++i) { \
      int64_t ri = mag_coords_iter_to_offset(&cr, i); \
      mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
      br[ri] = (Z); \
    } \
    for (int64_t i=0; i < x->numel; ++i) { \
      int64_t xi = mag_coords_iter_to_offset(&cx, i); \
      int64_t ri = mag_coords_iter_repeat(&cr, &cx, i); \
      mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
      mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
      br[ri] = RCVT(CVT(br[ri]) + CVT(bx[xi])); \
    } \
    return MAG_STATUS_OK; \
  }

mag_gen_stub_repeat_back(float, float32, .0f, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_repeat_back(mag_float16_t, float16, MAG_FLOAT16_ZERO, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_stub_repeat_back(mag_bfloat16_t, bfloat16, MAG_BFLOAT16_ZERO, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gen_stub_repeat_back(mag_float8_e4m3fn_t, float8_e4m3fn, MAG_FLOAT8_E4M3FN_ZERO, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)

#undef mag_gen_stub_repeat_back

static inline bool mag_coords_is_contig_dense(const mag_coords_t *c) {
  int64_t s = 1;
  for (int64_t dim = c->rank - 1; dim >= 0; --dim) {
    if (c->strides[dim] != s) return false;
    s *= c->shape[dim];
  }
  return true;
}

#define mag_gen_stub_gather(T, TF) \
  static MAG_HOTPROC mag_status_t mag_gather_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *src = mag_cmd_in(0); \
    const mag_tensor_t *index = mag_cmd_in(1); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(src); \
    const int64_t *bi = (const int64_t *)mag_tensor_data_ptr(index); \
    int64_t axis = mag_op_attr_unwrap_int64(mag_cmd_attr(0)); \
    if (axis < 0) axis += src->coords.rank; \
    int64_t ax = src->coords.shape[axis]; \
    int64_t total = r->numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    if (mag_unlikely(rb <= ra)) return MAG_STATUS_OK; \
    bool src_contig = mag_tensor_is_contiguous(src); \
    bool r_contig = mag_tensor_is_contiguous(r); \
    bool i_contig = mag_tensor_is_contiguous(index); \
    int64_t inner=1; \
    for (int64_t dim=axis+1; dim < src->coords.rank; ++dim) \
      inner*=src->coords.shape[dim]; \
    int64_t outer=1; \
    for (int64_t dim=0; dim < axis; ++dim) \
      outer*=src->coords.shape[dim]; \
    bool full_index = index->coords.rank == r->coords.rank && index->numel == r->numel; \
    if (mag_likely(src_contig && r_contig && i_contig && full_index)) { \
      int64_t out_ax = r->coords.shape[axis]; \
      for (int64_t flat=ra; flat < rb; ++flat) { \
        int64_t g = bi[flat]; \
        if (g < 0) g += ax; \
        mag_contract(err, ERR_KERNEL_FAILURE, {}, g >= 0 && g < ax, "gather: index %" PRIi64 " is out of range [0, %" PRIi64 ").", g, ax); \
        int64_t k = flat%inner; \
        int64_t t = flat/inner; \
        int64_t o = t/out_ax; \
        br[flat] = bx[o*ax*inner + g*inner + k]; \
      } \
      return MAG_STATUS_OK; \
    } \
    if (mag_likely(src_contig && r_contig && i_contig && index->coords.rank == 1)) { \
      int64_t idx_len = index->coords.shape[0]; \
      mag_contract(err, ERR_KERNEL_FAILURE, {}, r->coords.shape[axis] == idx_len, "gather: output shape along axis %" PRIi64 " (%" PRIi64 ") does not match index length %" PRIi64 ".", (int64_t)axis, r->coords.shape[axis], idx_len); \
      int64_t block = inner; \
      int64_t groups = outer*idx_len; \
      int64_t gra = ra/block; \
      int64_t grb = (rb + block - 1)/block; \
      for (int64_t group=gra; group < grb && group < groups; ++group) { \
        int64_t o = group/idx_len; \
        int64_t j = group%idx_len; \
        int64_t g = bi[j]; \
        if (g < 0) g += ax; \
        mag_contract(err, ERR_KERNEL_FAILURE, {}, g >= 0 && g < ax, "gather: index %" PRIi64 " is out of range [0, %" PRIi64 ").", g, ax); \
        int64_t dst_base = (o*idx_len + j)*inner; \
        int64_t src_base = (o*ax + g)*inner; \
        int64_t a = 0; \
        int64_t b = inner; \
        if (group == gra) a = ra - dst_base; \
        if (group == grb-1) b = rb - dst_base; \
        a = mag_xmax(a, 0); \
        b = mag_xmin(b, inner); \
        if (mag_likely(b > a)) \
          memcpy(br+dst_base+a, bx+src_base+a, (size_t)(b-a)*sizeof(T)); \
      } \
      return MAG_STATUS_OK; \
    } \
    int64_t oc[MAG_MAX_DIMS]; \
    int64_t sc[MAG_MAX_DIMS]; \
    bool full = true; \
    if (index->coords.rank != src->coords.rank) full = false; \
    else { \
      for (int64_t dim=0; dim < src->coords.rank; ++dim) { \
        if (dim == axis) continue; \
        if (index->coords.shape[dim] != src->coords.shape[dim]) { \
          full = false; \
          break; \
        } \
      } \
    } \
    mag_coords_iter_t ci; \
    mag_coords_iter_init(&ci, &index->coords); \
    for (int64_t flat = ra; flat < rb; ++flat) { \
      int64_t tmp = flat; \
      for (int64_t dim = r->coords.rank - 1; dim >= 0; --dim) { \
        oc[dim] = tmp % r->coords.shape[dim]; \
        tmp /= r->coords.shape[dim]; \
      } \
      int64_t gather_idx; \
      if (full) { \
        int64_t index_offset = mag_coords_iter_to_offset(&ci, flat); \
        gather_idx = bi[index_offset]; \
      } else if (index->coords.rank == 1) { \
        int64_t idx_pos = oc[axis]; \
        mag_contract(err, ERR_KERNEL_FAILURE, {}, idx_pos >= 0 && idx_pos < index->coords.shape[0], "gather: index position %" PRIi64 " is out of range [0, %" PRIi64 ").", idx_pos, index->coords.shape[0]); \
        gather_idx = bi[idx_pos * index->coords.strides[0]]; \
      } else { \
        int64_t index_offset = 0; \
        for (int64_t dim=0; dim < index->coords.rank; ++dim) \
          index_offset += oc[axis + dim] * index->coords.strides[dim]; \
        gather_idx = bi[index_offset]; \
      } \
      if (gather_idx < 0) gather_idx += ax; \
      mag_contract(err, ERR_KERNEL_FAILURE, {}, gather_idx >= 0 && gather_idx < ax, "gather: index %" PRIi64 " is out of range [0, %" PRIi64 ").", gather_idx, ax); \
      if (full || index->coords.rank == 1) { \
        for (int64_t dim=0; dim < src->coords.rank; ++dim) sc[dim] = oc[dim]; \
        sc[axis] = gather_idx; \
      } else { \
        for (int64_t dim=0; dim < axis; ++dim) sc[dim] = oc[dim]; \
        sc[axis] = gather_idx; \
        for (int64_t dim = axis + 1; dim < src->coords.rank; ++dim) \
          sc[dim] = oc[index->coords.rank + dim - 1]; \
      } \
      int64_t src_offset = 0; \
      int64_t dst_offset = 0; \
      for (int64_t dim=0; dim < src->coords.rank; ++dim) \
        src_offset += sc[dim]*src->coords.strides[dim]; \
      for (int64_t dim=0; dim < r->coords.rank; ++dim) \
        dst_offset += oc[dim]*r->coords.strides[dim]; \
      br[dst_offset] = bx[src_offset]; \
    } \
    return MAG_STATUS_OK; \
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

#define mag_gen_stub_tri_mask(T, TF, S, Z, CMP) \
  static mag_status_t MAG_HOTPROC mag_tri##S##_##TF(mag_error_t *err,const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *x = mag_cmd_in(0); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    mag_coords_iter_t cr, cx; \
    mag_coords_iter_init(&cr, &r->coords); \
    mag_coords_iter_init(&cx, &x->coords); \
    int64_t diag = mag_op_attr_unwrap_int64(mag_cmd_attr(0)); \
    int64_t total = r->numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    int64_t cols = r->coords.shape[r->coords.rank-1]; \
    int64_t rows = r->coords.shape[r->coords.rank-2]; \
    int64_t mat = rows*cols; \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t inner = i % mat; \
      int64_t row = inner / cols; \
      int64_t col = inner - row*cols; \
      int64_t ri, xi; \
      mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
      mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
      mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
      br[ri] = ((col-row) CMP diag) ? bx[xi] : (Z); \
    }  \
    return MAG_STATUS_OK; \
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
    const mag_tensor_t *x = mag_cmd_in(0); \
    mag_tensor_t *v = mag_cmd_out(0); \
    mag_tensor_t *idx = mag_cmd_out(1); \
    const int64_t k = mag_op_attr_unwrap_int64(mag_cmd_attr(0)); \
    int64_t dim = mag_op_attr_unwrap_int64(mag_cmd_attr(1)); \
    const bool largest = mag_op_attr_unwrap_bool(mag_cmd_attr(2)); \
    const bool sorted = mag_op_attr_unwrap_bool(mag_cmd_attr(3)); \
    (void)sorted; /* This implementation always emits sorted top-k, which is valid for sorted=false too. */ \
    const int64_t R = x->coords.rank; \
    mag_assert2(R > 0); \
    mag_assert2(dim >= 0 && dim < R); \
    const int64_t *shape_x = x->coords.shape; \
    const int64_t *shape_v = v->coords.shape; \
    const int64_t *shape_i = idx->coords.shape; \
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
    const int64_t outer_count = x->numel / dim_size; \
    if (outer_count <= 0) return MAG_STATUS_OK; \
    const int64_t stride_x_dim = x->coords.strides[dim]; \
    const int64_t stride_v_dim = v->coords.strides[dim]; \
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
        off_x0 += base_idx[d] * x->coords.strides[d]; \
        off_v0 += base_idx[d] * v->coords.strides[d]; \
      } \
      T *best_vals = mag_scratch_arena_alloc(&mag_tls_arena, (size_t)k * sizeof(*best_vals)); \
      int64_t *best_idx = mag_scratch_arena_alloc(&mag_tls_arena, (size_t)k * sizeof(*best_idx)); \
      int64_t filled = 0; \
      \
      for (int64_t p=0; p < dim_size; ++p) { \
        const int64_t off_x = off_x0 + p * stride_x_dim; \
        mag_bnd_chk(bx + off_x, bx, mag_tensor_numbytes(x)); \
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
        mag_bnd_chk(bv + off_v, bv, mag_tensor_numbytes(v)); \
        mag_bnd_chk(bi + off_v, bi, mag_tensor_numbytes(idx)); \
        bv[off_v] = best_vals[r]; \
        bi[off_v] = best_idx[r]; \
      } \
      mag_scratch_arena_reset(&mag_tls_arena, mark); \
    } \
    return MAG_STATUS_OK; \
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
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *cond = mag_cmd_in(0); \
    const mag_tensor_t *x = mag_cmd_in(1); \
    const mag_tensor_t *y = mag_cmd_in(2); \
    mag_assert2(cond->dtype == MAG_DTYPE_BOOLEAN); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const uint8_t *bc = (const uint8_t *)mag_tensor_data_ptr(cond); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *by = (const T *)mag_tensor_data_ptr(y); \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t total = r->numel; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    mag_coords_iter_t cr, cc, cx, cy; \
    mag_coords_iter_init(&cr, &r->coords); \
    mag_coords_iter_init(&cc, &cond->coords); \
    mag_coords_iter_init(&cx, &x->coords); \
    mag_coords_iter_init(&cy, &y->coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri, ci, xi, yi; \
      mag_coords_iter_offset4(&cr, &cc, &cx, &cy, i, &ri, &ci, &xi, &yi); \
      mag_bnd_chk(bc+ci, bc, mag_tensor_numbytes(cond)); \
      mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
      mag_bnd_chk(by+yi, by, mag_tensor_numbytes(y)); \
      mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
      br[ri] = bc[ci] ? bx[xi] : by[yi]; \
    } \
    return MAG_STATUS_OK; \
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
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *x = mag_cmd_in(0); \
    const mag_tensor_t *mn = mag_cmd_in(1); \
    const mag_tensor_t *mx = mag_cmd_in(2); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *bmn = (const T *)mag_tensor_data_ptr(mn); \
    const T *bmx = (const T *)mag_tensor_data_ptr(mx); \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t total = r->numel; \
    int64_t chunk = (total + tc - 1) / tc; \
    int64_t ra = ti * chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    mag_coords_iter_t cr, cx, cmn, cmx; \
    mag_coords_iter_init(&cr, &r->coords); \
    mag_coords_iter_init(&cx, &x->coords); \
    mag_coords_iter_init(&cmn, &mn->coords); \
    mag_coords_iter_init(&cmx, &mx->coords); \
    for (int64_t i = ra; i < rb; ++i) { \
      int64_t ri, xi, mni, mxi; \
      mag_coords_iter_offset4(&cr, &cx, &cmn, &cmx, i, &ri, &xi, &mni, &mxi); \
      float v = CVT(bx[xi]); \
      float lo = CVT(bmn[mni]); \
      float hi = CVT(bmx[mxi]); \
      float o = v < lo ? lo : (v > hi ? hi : v); \
      br[ri] = FROMF32(o); \
    } \
    return MAG_STATUS_OK; \
  }

mag_gen_stub_clamp_cvt(float, float32, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_clamp_cvt(mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_stub_clamp_cvt(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16)
mag_gen_stub_clamp_cvt(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32, mag_float32_to_float8_e4m3fn)

#undef mag_gen_stub_clamp_cvt

#define mag_gen_stub_clamp_ord(T, TF) \
  static mag_status_t MAG_HOTPROC mag_clamp_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *x = mag_cmd_in(0); \
    const mag_tensor_t *mn = mag_cmd_in(1); \
    const mag_tensor_t *mx = mag_cmd_in(2); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *bmn = (const T *)mag_tensor_data_ptr(mn); \
    const T *bmx = (const T *)mag_tensor_data_ptr(mx); \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t total = r->numel; \
    int64_t chunk = (total + tc - 1) / tc; \
    int64_t ra = ti * chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    mag_coords_iter_t cr, cx, cmn, cmx; \
    mag_coords_iter_init(&cr, &r->coords); \
    mag_coords_iter_init(&cx, &x->coords); \
    mag_coords_iter_init(&cmn, &mn->coords); \
    mag_coords_iter_init(&cmx, &mx->coords); \
    for (int64_t i = ra; i < rb; ++i) { \
      int64_t ri, xi, mni, mxi; \
      mag_coords_iter_offset4(&cr, &cx, &cmn, &cmx, i, &ri, &xi, &mni, &mxi); \
      T v = bx[xi]; \
      T lo = bmn[mni]; \
      T hi = bmx[mxi]; \
      br[ri] = v < lo ? lo : (v > hi ? hi : v); \
    } \
    return MAG_STATUS_OK; \
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
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *x = mag_cmd_in(0); \
    mag_assert2(r->dtype == MAG_DTYPE_INT64); \
    int64_t *br = (int64_t *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t num_samples = mag_op_attr_unwrap_int64(mag_cmd_attr(0)); \
    mag_philox4x32_stream_t *rng = payload->prng; \
    int64_t K = x->coords.shape[x->coords.rank-1]; \
    if (mag_unlikely(K <= 0)) return MAG_STATUS_OK; \
    int64_t B = x->numel / K; \
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
    return MAG_STATUS_OK; \
  }

mag_gen_stub_multinomial(float, float32, mag_cvt_nop)
mag_gen_stub_multinomial(mag_float16_t, float16, mag_float16_to_float32)
mag_gen_stub_multinomial(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32)
mag_gen_stub_multinomial(mag_float8_e4m3fn_t, float8_e4m3fn, mag_float8_e4m3fn_to_float32)
