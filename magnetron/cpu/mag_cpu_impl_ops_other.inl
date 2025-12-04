/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#define mag_gen_stub_cat(T, TF) \
    static MAG_HOTPROC void mag_cat_##TF(const mag_kernel_payload_t *payload) { \
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
    }

mag_gen_stub_cat(float, float32)
mag_gen_stub_cat(mag_float16_t, float16)
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
    static void MAG_HOTPROC mag_repeat_back_##TF(const mag_kernel_payload_t *payload) { \
        if (payload->thread_idx != 0) return; \
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
    }

mag_gen_stub_repeat_back(float, float32, .0f, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_repeat_back(mag_float16_t, float16, MAG_FLOAT16_ZERO, mag_float16_to_float32, mag_float32_to_float16)

#undef mag_gen_stub_repeat_back

#define mag_gen_stub_gather(T, TF) \
    static MAG_HOTPROC void mag_gather_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *src = mag_cmd_in(0); \
        const mag_tensor_t *index = mag_cmd_in(1); \
        mag_assert2(index->dtype == MAG_DTYPE_INT64); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        const T *bx = (const T *)mag_tensor_data_ptr(src); \
        const int64_t *bi = (const int64_t *)mag_tensor_data_ptr(index); \
        int64_t axis = mag_op_attr_unwrap_int64(mag_cmd_attr(0)); \
        if (axis < 0) axis += src->coords.rank; \
        mag_assert2(axis >= 0 && axis < src->coords.rank); \
        mag_assert2(index->coords.rank >= 1); \
        int64_t ax = src->coords.shape[axis]; \
        int64_t on = r->numel; \
        int64_t oc[MAG_MAX_DIMS]; \
        int64_t sc[MAG_MAX_DIMS]; \
        bool full = true; \
        for (int64_t d = 0; d < src->coords.rank; ++d) { \
            if (d == axis) continue; \
            if (index->coords.shape[d] != src->coords.shape[d]) { \
                full = false; \
                break; \
            } \
        } \
        mag_coords_iter_t ci; \
        mag_coords_iter_init(&ci, &index->coords); \
        for (int64_t flat=0; flat < on; ++flat) { \
            int64_t tmp = flat; \
            for (int64_t d=r->coords.rank-1; d >= 0; --d) { \
                oc[d] = tmp % r->coords.shape[d]; \
                tmp /= r->coords.shape[d]; \
            } \
            int64_t gather_idx; \
            if (full) { \
                int64_t index_offset = mag_coords_iter_to_offset(&ci, flat); \
                gather_idx = bi[index_offset]; \
            } else if (index->coords.rank == 1) { \
                int64_t idx_pos = oc[axis]; \
                mag_assert2(idx_pos >= 0 && idx_pos < index->coords.shape[0]); \
                int64_t index_offset = idx_pos*index->coords.strides[0]; \
                gather_idx = bi[index_offset]; \
            } else { \
                int64_t idx_coords[MAG_MAX_DIMS]; \
                for (int64_t i=0; i < index->coords.rank; ++i) idx_coords[i] = oc[axis+i]; \
                int64_t index_offset = 0; \
                for (int64_t d=0; d < index->coords.rank; ++d) index_offset += idx_coords[d]*index->coords.strides[d]; \
                gather_idx = bi[index_offset]; \
            } \
            if (gather_idx < 0) gather_idx += ax; \
            mag_assert2(gather_idx >= 0 && gather_idx < ax); \
            if (full) { \
                for (int64_t d=0; d < src->coords.rank; ++d) sc[d] = oc[d]; \
                sc[axis] = gather_idx; \
            } else if (index->coords.rank == 1) { \
                for (int64_t d=0; d < src->coords.rank; ++d) sc[d] = oc[d]; \
                sc[axis] = gather_idx; \
            } else { \
                for (int64_t d=0; d < axis; ++d) sc[d] = oc[d]; \
                sc[axis] = gather_idx; \
                for (int64_t d=axis+1; d < src->coords.rank; ++d) sc[d] = oc[index->coords.rank+d-1]; \
            } \
            int64_t src_offset = 0, dest_offset = 0; \
            for (int64_t d=0; d < src->coords.rank; ++d) src_offset += sc[d]*src->coords.strides[d]; \
            for (int64_t d=0; d < r->coords.rank; ++d) dest_offset += oc[d]*r->coords.strides[d]; \
            mag_bnd_chk(bx + src_offset, bx, mag_tensor_numbytes(src)); \
            mag_bnd_chk(br + dest_offset, br, mag_tensor_numbytes(r)); \
            br[dest_offset] = bx[src_offset]; \
        } \
    }

mag_gen_stub_gather(float, float32)
mag_gen_stub_gather(mag_float16_t, float16)
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
    static void MAG_HOTPROC mag_tri##S##_##TF(const mag_kernel_payload_t *payload) { \
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
    }

mag_gen_stub_tri_mask(float, float32, l, 0.f, <=)
mag_gen_stub_tri_mask(mag_float16_t, float16, l, MAG_FLOAT16_ZERO, <=)
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
mag_gen_stub_tri_mask(uint8_t, uint8, u, 0, >=)
mag_gen_stub_tri_mask(int8_t, int8, u, 0, >=)
mag_gen_stub_tri_mask(uint16_t, uint16, u, 0, >=)
mag_gen_stub_tri_mask(int16_t, int16, u, 0, >=)
mag_gen_stub_tri_mask(uint32_t, uint32, u, 0, >=)
mag_gen_stub_tri_mask(int32_t, int32, u, 0, >=)
mag_gen_stub_tri_mask(uint64_t, uint64, u, 0, >=)
mag_gen_stub_tri_mask(int64_t, int64, u, 0, >=)

#undef mag_gen_stub_tri_mask
