/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** +---------------------------------------------------------------------+
*/

#include "mag_cuda_misc.cuh"

#include "mag_cuda_unary.cuh"

#include <core/mag_prng_philox4x32.h>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace mag {

    constexpr int MISC_BLOCK_SIZE = 256;

    static void cuda_check(cudaError_t e, const char *what) {
        if (mag_unlikely(e != cudaSuccess))
            mag_panic("%s: %s", what, cudaGetErrorString(e));
    }

    /* --- one_hot --- */
    template <const bool same_layout>
    __global__ static void one_hot_kernel2(
        int64_t total,
        int64_t nc,
        int64_t *__restrict__ pr,
        const int64_t *__restrict__ pidx,
        mag_coords_iter_t it
    ) {
        int64_t t = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        for (; t < total; t += step) {
            int64_t cls;
            if constexpr (same_layout)
                cls = pidx[t];
            else {
                int64_t ridx = mag_coords_iter_to_offset(&it, t);
                cls = pidx[ridx];
            }
            if ((uint64_t)cls < (uint64_t)nc) {
                int64_t off = t*nc + cls;
                pr[off] = 1;
            }
        }
    }

    void misc_op_one_hot(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        mag_tensor_t *idx = cmd.in[0];
        mag_assert2(r->dtype == MAG_DTYPE_INT64 && idx->dtype == MAG_DTYPE_INT64);
        int64_t nc = mag_op_attr_unwrap_int64(cmd.attrs[0]);
        int64_t total = idx->numel;
        auto *pr = reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(r));
        const auto *pidx = reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(idx));
        int64_t blocks = (total + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
        mag_coords_iter_t it;
        mag_coords_iter_init(&it, &idx->coords);
        if (std::array<const mag_tensor_t *, 2> tensors{r, idx}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size()))
            one_hot_kernel2<true><<<blocks, MISC_BLOCK_SIZE>>>(total, nc, pr, pidx, it);
        else
            one_hot_kernel2<false><<<blocks, MISC_BLOCK_SIZE>>>(total, nc, pr, pidx, it);
    }

    /* --- tril / triu --- */
    template <typename T, const bool upper>
    __global__ static void tri_mask_kernel(
        int64_t total,
        T *__restrict__ br,
        const T *__restrict__ bx,
        mag_coords_iter_t cr,
        mag_coords_iter_t cx,
        int64_t diag
    ) {
        T z{};
        if constexpr (std::is_same_v<T, float>) z = 0.f;
        else if constexpr (std::is_same_v<T, half>) z = __float2half(0.f);
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) z = __float2bfloat16(0.f);
        else z = T{};
        int64_t ti = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        int64_t cols = cr.shape[cr.rank-1];
        int64_t rows = cr.shape[cr.rank-2];
        int64_t mat = rows*cols;
        for (; ti < total; ti += step) {
            int64_t inner = ti % mat;
            int64_t row = inner / cols;
            int64_t col = inner - row*cols;
            int64_t ri, xi;
            mag_coords_iter_offset2(&cr, &cx, ti, &ri, &xi);
            bool keep = upper ? (col - row) >= diag : (col - row) <= diag;
            br[ri] = keep ? bx[xi] : z;
        }
    }

    template <typename T, const bool upper>
    static void launch_tri_mask(mag_tensor_t *r, const mag_tensor_t *x, int64_t diag) {
        int64_t n = mag_tensor_numel(r);
        int64_t blocks = (n + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
        mag_coords_iter_t cr, cx;
        mag_coords_iter_init(&cr, &r->coords);
        mag_coords_iter_init(&cx, &x->coords);
        auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
        const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        tri_mask_kernel<T, upper><<<blocks, MISC_BLOCK_SIZE>>>(n, br, bx, cr, cx, diag);
    }

    void misc_op_tril(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        int64_t diag = mag_op_attr_unwrap_int64(cmd.attrs[0]);
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_tri_mask<float, false>(r, x, diag); break;
            case MAG_DTYPE_FLOAT16: launch_tri_mask<half, false>(r, x, diag); break;
            case MAG_DTYPE_BFLOAT16: launch_tri_mask<__nv_bfloat16, false>(r, x, diag); break;
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_tri_mask<uint8_t, false>(r, x, diag); break;
            case MAG_DTYPE_INT8: launch_tri_mask<int8_t, false>(r, x, diag); break;
            case MAG_DTYPE_UINT16: launch_tri_mask<uint16_t, false>(r, x, diag); break;
            case MAG_DTYPE_INT16: launch_tri_mask<int16_t, false>(r, x, diag); break;
            case MAG_DTYPE_UINT32: launch_tri_mask<uint32_t, false>(r, x, diag); break;
            case MAG_DTYPE_INT32: launch_tri_mask<int32_t, false>(r, x, diag); break;
            case MAG_DTYPE_UINT64: launch_tri_mask<uint64_t, false>(r, x, diag); break;
            case MAG_DTYPE_INT64: launch_tri_mask<int64_t, false>(r, x, diag); break;
            default: mag_assert(false, "tril: unsupported dtype");
        }
    }

    void misc_op_triu(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        int64_t diag = mag_op_attr_unwrap_int64(cmd.attrs[0]);
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_tri_mask<float, true>(r, x, diag); break;
            case MAG_DTYPE_FLOAT16: launch_tri_mask<half, true>(r, x, diag); break;
            case MAG_DTYPE_BFLOAT16: launch_tri_mask<__nv_bfloat16, true>(r, x, diag); break;
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_tri_mask<uint8_t, true>(r, x, diag); break;
            case MAG_DTYPE_INT8: launch_tri_mask<int8_t, true>(r, x, diag); break;
            case MAG_DTYPE_UINT16: launch_tri_mask<uint16_t, true>(r, x, diag); break;
            case MAG_DTYPE_INT16: launch_tri_mask<int16_t, true>(r, x, diag); break;
            case MAG_DTYPE_UINT32: launch_tri_mask<uint32_t, true>(r, x, diag); break;
            case MAG_DTYPE_INT32: launch_tri_mask<int32_t, true>(r, x, diag); break;
            case MAG_DTYPE_UINT64: launch_tri_mask<uint64_t, true>(r, x, diag); break;
            case MAG_DTYPE_INT64: launch_tri_mask<int64_t, true>(r, x, diag); break;
            default: mag_assert(false, "triu: unsupported dtype");
        }
    }

    /* --- where --- */
    template <typename T, const bool contig>
    __global__ static void where_kernel(
        int64_t n,
        T *__restrict__ br,
        const uint8_t *__restrict__ bc,
        const T *__restrict__ bx,
        const T *__restrict__ by,
        mag_coords_iter_t cr,
        mag_coords_iter_t cc,
        mag_coords_iter_t cx,
        mag_coords_iter_t cy
    ) {
        int64_t i = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        if constexpr (contig) {
            for (; i < n; i += step) {
                br[i] = bc[i] ? bx[i] : by[i];
            }
        } else {
            for (; i < n; i += step) {
                int64_t ri, ci, xi, yi;
                mag_coords_iter_offset4(&cr, &cc, &cx, &cy, i, &ri, &ci, &xi, &yi);
                br[ri] = bc[ci] ? bx[xi] : by[yi];
            }
        }
    }

    template <typename T>
    static void launch_where(mag_tensor_t *r, const mag_tensor_t *cond, const mag_tensor_t *x, const mag_tensor_t *y) {
        int64_t n = mag_tensor_numel(r);
        int64_t blocks = (n + UNARY_BLOCK_SIZE - 1) / UNARY_BLOCK_SIZE;
        mag_coords_iter_t cr, cc, cx, cy;
        mag_coords_iter_init(&cr, &r->coords);
        mag_coords_iter_init(&cc, &cond->coords);
        mag_coords_iter_init(&cx, &x->coords);
        mag_coords_iter_init(&cy, &y->coords);
        auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
        const auto *bc = reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(cond));
        const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        const auto *by = reinterpret_cast<const T *>(mag_tensor_data_ptr(y));
        if (std::array<const mag_tensor_t *, 4> tensors{r, cond, x, y}; mag_all_shapes_equal_and_contig(tensors.data(), tensors.size()))
            where_kernel<T, true><<<blocks, UNARY_BLOCK_SIZE>>>(n, br, bc, bx, by, cr, cc, cx, cy);
        else
            where_kernel<T, false><<<blocks, UNARY_BLOCK_SIZE>>>(n, br, bc, bx, by, cr, cc, cx, cy);
    }

    void misc_op_where(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *cond = cmd.in[0];
        const mag_tensor_t *x = cmd.in[1];
        const mag_tensor_t *y = cmd.in[2];
        mag_assert2(cond->dtype == MAG_DTYPE_BOOLEAN);
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_where<float>(r, cond, x, y); break;
            case MAG_DTYPE_FLOAT16: launch_where<half>(r, cond, x, y); break;
            case MAG_DTYPE_BFLOAT16: launch_where<__nv_bfloat16>(r, cond, x, y); break;
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_where<uint8_t>(r, cond, x, y); break;
            case MAG_DTYPE_INT8: launch_where<int8_t>(r, cond, x, y); break;
            case MAG_DTYPE_UINT16: launch_where<uint16_t>(r, cond, x, y); break;
            case MAG_DTYPE_INT16: launch_where<int16_t>(r, cond, x, y); break;
            case MAG_DTYPE_UINT32: launch_where<uint32_t>(r, cond, x, y); break;
            case MAG_DTYPE_INT32: launch_where<int32_t>(r, cond, x, y); break;
            case MAG_DTYPE_UINT64: launch_where<uint64_t>(r, cond, x, y); break;
            case MAG_DTYPE_INT64: launch_where<int64_t>(r, cond, x, y); break;
            default: mag_assert(false, "where: unsupported dtype");
        }
    }

    /* --- repeat_back (single-thread; matches CPU accumulation order) --- */
    __global__ static void repeat_back_f32_kernel(
        mag_coords_iter_t cr,
        mag_coords_iter_t cx,
        int64_t rn,
        int64_t xn,
        float *__restrict__ br,
        const float *__restrict__ bx
    ) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        for (int64_t i=0; i < rn; ++i) {
            int64_t ri = mag_coords_iter_to_offset(&cr, i);
            br[ri] = 0.f;
        }
        for (int64_t i=0; i < xn; ++i) {
            int64_t xi = mag_coords_iter_to_offset(&cx, i);
            int64_t ri = mag_coords_iter_repeat(&cr, &cx, i);
            br[ri] = br[ri] + bx[xi];
        }
    }

    __global__ static void repeat_back_f16_kernel(
        mag_coords_iter_t cr,
        mag_coords_iter_t cx,
        int64_t rn,
        int64_t xn,
        half *__restrict__ br,
        const half *__restrict__ bx
    ) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        for (int64_t i=0; i < rn; ++i) {
            int64_t ri = mag_coords_iter_to_offset(&cr, i);
            br[ri] = __float2half(0.f);
        }
        for (int64_t i=0; i < xn; ++i) {
            int64_t xi = mag_coords_iter_to_offset(&cx, i);
            int64_t ri = mag_coords_iter_repeat(&cr, &cx, i);
            float s = __half2float(br[ri]) + __half2float(bx[xi]);
            br[ri] = __float2half(s);
        }
    }

    __global__ static void repeat_back_bf16_kernel(
        mag_coords_iter_t cr,
        mag_coords_iter_t cx,
        int64_t rn,
        int64_t xn,
        __nv_bfloat16 *__restrict__ br,
        const __nv_bfloat16 *__restrict__ bx
    ) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        for (int64_t i=0; i < rn; ++i) {
            int64_t ri = mag_coords_iter_to_offset(&cr, i);
            br[ri] = __float2bfloat16(0.f);
        }
        for (int64_t i=0; i < xn; ++i) {
            int64_t xi = mag_coords_iter_to_offset(&cx, i);
            int64_t ri = mag_coords_iter_repeat(&cr, &cx, i);
            float s = __bfloat162float(br[ri]) + __bfloat162float(bx[xi]);
            br[ri] = __float2bfloat16(s);
        }
    }

    void misc_op_repeat_back(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        mag_coords_iter_t cr, cx;
        mag_coords_iter_init(&cr, &r->coords);
        mag_coords_iter_init(&cx, &x->coords);
        int64_t rn = r->numel;
        int64_t xn = x->numel;
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32:
                repeat_back_f32_kernel<<<1, 1>>>(cr, cx, rn, xn,
                    reinterpret_cast<float *>(mag_tensor_data_ptr_mut(r)),
                    reinterpret_cast<const float *>(mag_tensor_data_ptr(x)));
                break;
            case MAG_DTYPE_FLOAT16:
                repeat_back_f16_kernel<<<1, 1>>>(cr, cx, rn, xn,
                    reinterpret_cast<half *>(mag_tensor_data_ptr_mut(r)),
                    reinterpret_cast<const half *>(mag_tensor_data_ptr(x)));
                break;
            case MAG_DTYPE_BFLOAT16:
                repeat_back_bf16_kernel<<<1, 1>>>(cr, cx, rn, xn,
                    reinterpret_cast<__nv_bfloat16 *>(mag_tensor_data_ptr_mut(r)),
                    reinterpret_cast<const __nv_bfloat16 *>(mag_tensor_data_ptr(x)));
                break;
            default: mag_assert(false, "repeat_back: unsupported dtype");
        }
    }

    /* --- gather --- */
    template <typename T>
    __global__ static void gather_kernel(
        int64_t on,
        T *__restrict__ br,
        const T *__restrict__ bx,
        const int64_t *__restrict__ bi,
        mag_tensor_t src,
        mag_tensor_t index,
        mag_tensor_t out,
        int64_t axis_in,
        mag_coords_iter_t ci
    ) {
        int64_t flat = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(blockIdx.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        int64_t ax = src.coords.shape[axis_in];
        for (; flat < on; flat += step) {
            int64_t oc[MAG_MAX_DIMS];
            int64_t sc[MAG_MAX_DIMS];
            int64_t tmp = flat;
            for (int64_t d = out.coords.rank-1; d >= 0; --d) {
                oc[d] = tmp % out.coords.shape[d];
                tmp /= out.coords.shape[d];
            }
            bool full = true;
            for (int64_t d = 0; d < src.coords.rank; ++d) {
                if (d == axis_in) continue;
                if (index.coords.shape[d] != src.coords.shape[d]) {
                    full = false;
                    break;
                }
            }
            int64_t gather_idx;
            if (full) {
                int64_t index_offset = mag_coords_iter_to_offset(&ci, flat);
                gather_idx = bi[index_offset];
            } else if (index.coords.rank == 1) {
                int64_t idx_pos = oc[axis_in];
                int64_t index_offset = idx_pos*index.coords.strides[0];
                gather_idx = bi[index_offset];
            } else {
                int64_t idx_coords[MAG_MAX_DIMS];
                for (int64_t i=0; i < index.coords.rank; ++i) idx_coords[i] = oc[axis_in+i];
                int64_t index_offset = 0;
                for (int64_t d=0; d < index.coords.rank; ++d) index_offset += idx_coords[d]*index.coords.strides[d];
                gather_idx = bi[index_offset];
            }
            if (gather_idx < 0) gather_idx += ax;
            if (full) {
                for (int64_t d=0; d < src.coords.rank; ++d) sc[d] = oc[d];
                sc[axis_in] = gather_idx;
            } else if (index.coords.rank == 1) {
                for (int64_t d=0; d < src.coords.rank; ++d) sc[d] = oc[d];
                sc[axis_in] = gather_idx;
            } else {
                for (int64_t d=0; d < axis_in; ++d) sc[d] = oc[d];
                sc[axis_in] = gather_idx;
                for (int64_t d=axis_in+1; d < src.coords.rank; ++d) sc[d] = oc[index.coords.rank+d-1];
            }
            int64_t src_offset = 0, dest_offset = 0;
            for (int64_t d=0; d < src.coords.rank; ++d) src_offset += sc[d]*src.coords.strides[d];
            for (int64_t d=0; d < out.coords.rank; ++d) dest_offset += oc[d]*out.coords.strides[d];
            br[dest_offset] = bx[src_offset];
        }
    }

    template <typename T>
    static void launch_gather(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *src = cmd.in[0];
        const mag_tensor_t *index = cmd.in[1];
        int64_t axis = mag_op_attr_unwrap_int64(cmd.attrs[0]);
        if (axis < 0) axis += src->coords.rank;
        mag_assert2(axis >= 0 && axis < src->coords.rank);
        int64_t on = r->numel;
        int64_t blocks = (on + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
        auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
        const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(src));
        const auto *bi = reinterpret_cast<const int64_t *>(mag_tensor_data_ptr(index));
        mag_coords_iter_t ci;
        mag_coords_iter_init(&ci, &index->coords);
        gather_kernel<T><<<blocks, MISC_BLOCK_SIZE>>>(on, br, bx, bi, *src, *index, *r, axis, ci);
    }

    void misc_op_gather(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_gather<float>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_gather<half>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_gather<__nv_bfloat16>(cmd); break;
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_gather<uint8_t>(cmd); break;
            case MAG_DTYPE_INT8: launch_gather<int8_t>(cmd); break;
            case MAG_DTYPE_UINT16: launch_gather<uint16_t>(cmd); break;
            case MAG_DTYPE_INT16: launch_gather<int16_t>(cmd); break;
            case MAG_DTYPE_UINT32: launch_gather<uint32_t>(cmd); break;
            case MAG_DTYPE_INT32: launch_gather<int32_t>(cmd); break;
            case MAG_DTYPE_UINT64: launch_gather<uint64_t>(cmd); break;
            case MAG_DTYPE_INT64: launch_gather<int64_t>(cmd); break;
            default: mag_assert(false, "gather: unsupported dtype");
        }
    }

    /* --- cat (output contiguous) --- */
    void misc_op_cat(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        mag_assert2(mag_tensor_is_contiguous(r));
        const int64_t dim = mag_op_attr_unwrap_int64(cmd.attrs[0]);
        const uint32_t n = cmd.num_in;
        mag_assert2(n > 0 && dim >= 0 && dim < r->coords.rank);
        int64_t R = r->coords.rank;
        int64_t inner_block = 1;
        for (int64_t d = dim+1; d < R; ++d) inner_block *= r->coords.shape[d];
        int64_t outer_count = 1;
        for (int64_t d=0; d < dim; ++d) outer_count *= r->coords.shape[d];
        int64_t mult[MAG_MAX_DIMS];
        for (int64_t d = 0; d < dim; ++d) {
            int64_t m = 1;
            for (int64_t k = d + 1; k < dim; ++k) m *= r->coords.shape[k];
            mult[d] = m;
        }
        for (int64_t p=0; p < outer_count; ++p) {
            int64_t idx_prefix[MAG_MAX_DIMS];
            int64_t rtmp = p;
            for (int64_t d = 0; d < dim; ++d) {
                int64_t q = !mult[d] ? 0 : rtmp/mult[d];
                if (mult[d] != 0) rtmp = rtmp%mult[d];
                idx_prefix[d] = q;
            }
            int64_t moff = 0;
            for (int64_t d=0; d < dim; ++d) moff += idx_prefix[d]*r->coords.strides[d];
            int64_t cur = 0;
            for (uint32_t i=0; i < n; ++i) {
                const mag_tensor_t *x = cmd.in[i];
                int64_t smoff=0;
                for (int64_t d=0; d < dim; ++d)
                    smoff += idx_prefix[d]*x->coords.strides[d];
                int64_t cl = x->coords.shape[dim];
                int64_t numel = cl*inner_block;
                int64_t oel = moff + cur*r->coords.strides[dim];
                int64_t sel = smoff;
                mag_assert2(r->dtype == x->dtype);
                size_t elsz = static_cast<size_t>(mag_tensor_numbytes(r) / mag_tensor_numel(r));
                const uint8_t *src_ptr = reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(x)) + sel * static_cast<int64_t>(elsz);
                uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(mag_tensor_data_ptr_mut(r)) + oel * static_cast<int64_t>(elsz);
                cuda_check(cudaMemcpy(dst_ptr, src_ptr, static_cast<size_t>(numel) * elsz, cudaMemcpyDeviceToDevice), "cat D2D");
                cur += cl;
            }
        }
    }

    /* --- matmul --- */
    struct matmul_desc_t {
        int64_t rank;
        int64_t shape[MAG_MAX_DIMS];
        int64_t stride[MAG_MAX_DIMS];
    };

    static matmul_desc_t desc_from(const mag_tensor_t *t) {
        matmul_desc_t d{};
        d.rank = t->coords.rank;
        for (int64_t i=0; i < d.rank; ++i) {
            d.shape[i] = t->coords.shape[i];
            d.stride[i] = t->coords.strides[i];
        }
        return d;
    }

    __device__ __forceinline__ int64_t mag_d_offset_rmn(matmul_desc_t t, int64_t flat, int64_t i, int64_t j) {
        int64_t ra = t.rank;
        const int64_t *td = t.shape;
        const int64_t *ts = t.stride;
        if (ra <= 3) {
            switch (ra) {
                case 0: return 0;
                case 1: return flat*ts[0];
                case 2: return i*ts[0] + j*ts[1];
                case 3: return flat*ts[0] + i*ts[1] + j*ts[2];
            }
        }
        int64_t off = 0, rem = flat;
        for (int64_t d = ra-3; d >= 0; --d) {
            int64_t idx = rem % td[d];
            rem /= td[d];
            off += idx*ts[d];
        }
        off += i*ts[ra-2];
        off += j*ts[ra-1];
        return off;
    }

    template <typename T>
    __device__ __forceinline__ float d_mm_to_f32(T x) {
        if constexpr (std::is_same_v<T, float>) return x;
        else if constexpr (std::is_same_v<T, half>) return __half2float(x);
        else return __bfloat162float(x);
    }

    template <typename T>
    __device__ __forceinline__ T d_mm_from_f32(float v) {
        if constexpr (std::is_same_v<T, float>) return v;
        else if constexpr (std::is_same_v<T, half>) return __float2half(v);
        else return __float2bfloat16(v);
    }

    template <typename T>
    __device__ float d_load_x(const matmul_desc_t &x, const T *bx, int64_t xb_flat, int64_t i, int64_t k) {
        int64_t off = (x.rank == 1) ? mag_d_offset_rmn(x, k, 0, 0) : mag_d_offset_rmn(x, xb_flat, i, k);
        return d_mm_to_f32(bx[off]);
    }

    template <typename T>
    __device__ float d_load_y(const matmul_desc_t &y, const T *by, int64_t yb_flat, int64_t k, int64_t n) {
        int64_t off = (y.rank == 1) ? mag_d_offset_rmn(y, k, 0, 0) : mag_d_offset_rmn(y, yb_flat, k, n);
        return d_mm_to_f32(by[off]);
    }

    template <typename T>
    __device__ void d_store_r(const matmul_desc_t &r, T *br, int64_t rb_flat, int64_t i, int64_t n, float v) {
        if (r.rank == 0)
            br[0] = d_mm_from_f32<T>(v);
        else if (r.rank == 1) {
            int64_t off = mag_d_offset_rmn(r, n, 0, 0);
            br[off] = d_mm_from_f32<T>(v);
        } else {
            int64_t off = mag_d_offset_rmn(r, rb_flat, i, n);
            br[off] = d_mm_from_f32<T>(v);
        }
    }

    template <typename T>
    __global__ static void matmul_kernel(
        int64_t M, int64_t N, int64_t K, int64_t batch_total,
        int64_t bdx, int64_t bdy, int64_t bdr,
        matmul_desc_t dx, matmul_desc_t dy, matmul_desc_t dr,
        T *br, const T *bx, const T *by
    ) {
        int64_t tid = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t total = batch_total * M * N;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        for (int64_t w = tid; w < total; w += step) {
            int64_t mn = M * N;
            int64_t batch_idx = w / mn;
            int64_t rem = w % mn;
            int64_t i = rem / N;
            int64_t n = rem % N;
            int64_t idx_r[MAG_MAX_DIMS] = {0};
            {
                int64_t t = batch_idx;
                for (int64_t d = bdr-1; d >= 0; --d) {
                    idx_r[d] = t % dr.shape[d];
                    t /= dr.shape[d];
                }
            }
            int64_t xb_flat = 0;
            for (int64_t d = 0; d < bdx; ++d) {
                int64_t rd = bdr - bdx + d;
                int64_t idx = (dx.shape[d] == 1) ? 0 : idx_r[rd];
                xb_flat = xb_flat * dx.shape[d] + idx;
            }
            int64_t yb_flat = 0;
            for (int64_t d = 0; d < bdy; ++d) {
                int64_t rd = bdr - bdy + d;
                int64_t idx = (dy.shape[d] == 1) ? 0 : idx_r[rd];
                yb_flat = yb_flat * dy.shape[d] + idx;
            }
            int64_t rb_flat = 0;
            for (int64_t d = 0; d < bdr; ++d)
                rb_flat = rb_flat * dr.shape[d] + idx_r[d];
            float sum = 0.f;
            for (int64_t k = 0; k < K; ++k) {
                float ax = d_load_x(dx, bx, xb_flat, i, k);
                float byv = d_load_y(dy, by, yb_flat, k, n);
                sum += ax * byv;
            }
            d_store_r(dr, br, rb_flat, i, n, sum);
        }
    }

    template <typename T>
    static void launch_matmul(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        const mag_tensor_t *y = cmd.in[1];
        int64_t M = (x->coords.rank == 1) ? 1 : x->coords.shape[x->coords.rank - 2];
        int64_t N = (y->coords.rank == 1) ? 1 : y->coords.shape[y->coords.rank - 1];
        int64_t K = x->coords.shape[x->coords.rank - 1];
        int64_t bdr = (r->coords.rank > 2) ? (r->coords.rank - 2) : 0;
        int64_t batch_total = 1;
        for (int64_t d = 0; d < bdr; ++d) batch_total *= r->coords.shape[d];
        int64_t bdx = (x->coords.rank > 2) ? (x->coords.rank - 2) : 0;
        int64_t bdy = (y->coords.rank > 2) ? (y->coords.rank - 2) : 0;
        matmul_desc_t dx = desc_from(x);
        matmul_desc_t dy = desc_from(y);
        matmul_desc_t dr = desc_from(r);
        auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
        const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        const auto *by = reinterpret_cast<const T *>(mag_tensor_data_ptr(y));
        int64_t total = batch_total * M * N;
        int64_t blocks = (total + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
        matmul_kernel<T><<<blocks, MISC_BLOCK_SIZE>>>(M, N, K, batch_total, bdx, bdy, bdr, dx, dy, dr, br, bx, by);
    }

    void misc_op_matmul(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32: launch_matmul<float>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_matmul<half>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_matmul<__nv_bfloat16>(cmd); break;
            default: mag_assert(false, "matmul: unsupported dtype");
        }
    }

    /* --- topk --- */
    template <typename T>
    __device__ __forceinline__ float topk_cmp_val(T x) {
        if constexpr (std::is_same_v<T, float>) return x;
        else if constexpr (std::is_same_v<T, half>) return __half2float(x);
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(x);
        else return static_cast<float>(x);
    }

    template <typename T>
    __global__ static void topk_rows_kernel(
        int64_t outer_count,
        int64_t dim_size,
        int64_t k,
        bool largest,
        int64_t R,
        int64_t dim,
        int64_t stride_x_dim,
        int64_t stride_v_dim,
        mag_tensor_t x_t,
        mag_tensor_t v_t,
        mag_tensor_t i_t,
        const T *bx,
        T *bv,
        int64_t *bi,
        char *scratch_base,
        size_t row_bytes
    ) {
        int64_t row = static_cast<int64_t>(blockIdx.x);
        if (row >= outer_count || threadIdx.x != 0) return;
        const int64_t *shape_x = x_t.coords.shape;
        const int64_t *str_x = x_t.coords.strides;
        const int64_t *str_v = v_t.coords.strides;
        (void)i_t;
        T *vals_buf = reinterpret_cast<T *>(scratch_base + static_cast<size_t>(row) * row_bytes);
        int64_t *idx_buf = reinterpret_cast<int64_t *>(vals_buf + dim_size);
        int64_t outer_rank = R - 1;
        int64_t shape_outer[MAG_MAX_DIMS];
        int64_t mult_outer[MAG_MAX_DIMS];
        int64_t outer_to_full[MAG_MAX_DIMS];
        {
            int64_t t=0;
            for (int64_t d=0; d < R; ++d) {
                if (d == dim) continue;
                shape_outer[t] = shape_x[d];
                outer_to_full[t] = d;
                ++t;
            }
            for (int64_t t2=0; t2 < outer_rank; ++t2) {
                int64_t m=1;
                for (int64_t k2=t2+1; k2 < outer_rank; ++k2)
                    m *= shape_outer[k2];
                mult_outer[t2] = m;
            }
        }
        int64_t rtmp = row;
        int64_t base_idx[MAG_MAX_DIMS] = {0};
        for (int64_t d=0; d < R; ++d) base_idx[d] = 0;
        for (int64_t t=0; t < outer_rank; ++t) {
            int64_t q = mult_outer[t] == 0 ? 0 : rtmp / mult_outer[t];
            if (mult_outer[t] != 0) rtmp = rtmp % mult_outer[t];
            int64_t fd = outer_to_full[t];
            base_idx[fd] = q;
        }
        base_idx[dim] = 0;
        int64_t off_x0=0;
        int64_t off_v0=0;
        for (int64_t d=0; d < R; ++d) {
            off_x0 += base_idx[d] * str_x[d];
            off_v0 += base_idx[d] * str_v[d];
        }
        for (int64_t p = 0; p < dim_size; ++p) {
            int64_t off_x = off_x0 + p * stride_x_dim;
            vals_buf[p] = bx[off_x];
            idx_buf[p] = p;
        }
        for (int64_t rr=0; rr < k; ++rr) {
            int64_t best = rr;
            for (int64_t p = rr+1; p < dim_size; ++p) {
                T vp = vals_buf[p];
                T vb = vals_buf[best];
                float fvp = topk_cmp_val(vp);
                float fvb = topk_cmp_val(vb);
                bool better;
                if (largest) better = (fvp > fvb) || ((fvp == fvb) && (idx_buf[p] < idx_buf[best]));
                else better = (fvp < fvb) || ((fvp == fvb) && (idx_buf[p] < idx_buf[best]));
                if (better) best = p;
            }
            if (best != rr) {
                T tv = vals_buf[rr];
                vals_buf[rr] = vals_buf[best];
                vals_buf[best] = tv;
                int64_t ti2 = idx_buf[rr];
                idx_buf[rr] = idx_buf[best];
                idx_buf[best] = ti2;
            }
        }
        for (int64_t rr=0; rr < k; ++rr) {
            int64_t off_v = off_v0 + rr*stride_v_dim;
            bv[off_v] = vals_buf[rr];
            bi[off_v] = idx_buf[rr];
        }
    }

    template <typename T>
    static void launch_topk(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        mag_tensor_t *v = cmd.out[0];
        mag_tensor_t *idx = cmd.out[1];
        const int64_t k = mag_op_attr_unwrap_int64(cmd.attrs[0]);
        int64_t dim = mag_op_attr_unwrap_int64(cmd.attrs[1]);
        bool largest = mag_op_attr_unwrap_bool(cmd.attrs[2]);
        (void)mag_op_attr_unwrap_bool(cmd.attrs[3]);
        int64_t R = x->coords.rank;
        mag_assert2(dim >= 0 && dim < R);
        const int64_t dim_size = x->coords.shape[dim];
        mag_assert2(k > 0 && k <= dim_size);
        int64_t outer_count = x->numel / dim_size;
        if (outer_count <= 0) return;
        size_t row_bytes = static_cast<size_t>(dim_size) * (sizeof(T) + sizeof(int64_t));
        void *d_scratch = nullptr;
        cuda_check(cudaMalloc(&d_scratch, row_bytes * static_cast<size_t>(outer_count)), "topk cudaMalloc");
        const T *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        T *bv = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(v));
        int64_t *bi = reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(idx));
        int64_t stride_x_dim = x->coords.strides[dim];
        int64_t stride_v_dim = v->coords.strides[dim];
        topk_rows_kernel<T><<<static_cast<unsigned>(outer_count), 1>>>(
            outer_count, dim_size, k, largest, R, dim, stride_x_dim, stride_v_dim,
            *x, *v, *idx, bx, bv, bi, reinterpret_cast<char *>(d_scratch), row_bytes);
        cuda_check(cudaFree(d_scratch), "topk cudaFree");
    }

    void misc_op_topk(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32: launch_topk<float>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_topk<half>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_topk<__nv_bfloat16>(cmd); break;
            case MAG_DTYPE_UINT8: launch_topk<uint8_t>(cmd); break;
            case MAG_DTYPE_INT8: launch_topk<int8_t>(cmd); break;
            case MAG_DTYPE_UINT16: launch_topk<uint16_t>(cmd); break;
            case MAG_DTYPE_INT16: launch_topk<int16_t>(cmd); break;
            case MAG_DTYPE_UINT32: launch_topk<uint32_t>(cmd); break;
            case MAG_DTYPE_INT32: launch_topk<int32_t>(cmd); break;
            case MAG_DTYPE_UINT64: launch_topk<uint64_t>(cmd); break;
            case MAG_DTYPE_INT64: launch_topk<int64_t>(cmd); break;
            default: mag_assert(false, "topk: unsupported dtype");
        }
    }

    /* --- multinomial --- */
    struct mag_discrete_sample_pair_d {
        float score;
        int64_t idx;
    };

    __device__ __forceinline__ void insertion_sort_pairs_desc(mag_discrete_sample_pair_d *arr, int64_t n) {
        for (int64_t i = 1; i < n; ++i) {
            mag_discrete_sample_pair_d key = arr[i];
            int64_t j = i - 1;
            while (j >= 0 && (arr[j].score < key.score || (arr[j].score == key.score && arr[j].idx > key.idx))) {
                arr[j+1] = arr[j];
                --j;
            }
            arr[j+1] = key;
        }
    }

    template <typename T>
    __global__ static void multinomial_rows_kernel(
        int64_t B,
        int64_t K,
        int64_t num_samples,
        const T *bx,
        int64_t *br,
        uint64_t seed,
        uint64_t subseq0,
        mag_discrete_sample_pair_d *workspace
    ) {
        int64_t b = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        if (b >= B) return;
        mag_philox4x32_stream_t stream;
        mag_philox4x32_stream_seed(&stream, seed, subseq0 + static_cast<uint64_t>(b));
        const T *w = bx + b*K;
        int64_t *o = br + b*num_samples;
        float sumw = 0.f;
        int64_t nnz = 0;
        for (int64_t i=0; i < K; ++i) {
            float wi = d_mm_to_f32(w[i]);
            if (!isfinite(wi) || wi <= 0.f) wi = 0.f;
            sumw += wi;
            if (wi > 0.f) ++nnz;
        }
        mag_discrete_sample_pair_d *arr = workspace + b * K;
        if (!(sumw > 0.f) || nnz == 0) {
            for (int64_t s=0; s < num_samples; ++s) o[s] = -1;
            return;
        }
        int64_t kout = num_samples;
        if (kout > nnz) kout = nnz;
        if (kout <= 0) {
            for (int64_t s=0; s < num_samples; ++s) o[s] = -1;
            return;
        }
        int64_t m = 0;
        for (int64_t i=0; i < K; ++i) {
            float wi = d_mm_to_f32(w[i]);
            if (!isfinite(wi) || wi <= 0.f) continue;
            float u = mag_philox4x32_next_float32(&stream);
            u = fmaxf(u, 1e-37f);
            float g = -logf(-logf(u));
            arr[m].score = logf(wi) + g;
            arr[m].idx = i;
            ++m;
        }
        insertion_sort_pairs_desc(arr, m);
        for (int64_t s=0; s < kout; ++s) o[s] = arr[s].idx;
        for (int64_t s=kout; s < num_samples; ++s) o[s] = -1;
    }

    void misc_op_multinomial(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        mag_assert2(r->dtype == MAG_DTYPE_INT64);
        int64_t num_samples = mag_op_attr_unwrap_int64(cmd.attrs[0]);
        int64_t K = x->coords.shape[x->coords.rank-1];
        if (K <= 0) return;
        int64_t B = x->numel / K;
        if (B <= 0) return;
        size_t ws = static_cast<size_t>(B) * static_cast<size_t>(K) * sizeof(mag_discrete_sample_pair_d);
        void *d_ws = nullptr;
        cuda_check(cudaMalloc(&d_ws, ws), "multinomial cudaMalloc");
        uint64_t seed = global_seed.load(std::memory_order_relaxed);
        uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
        int64_t *br = reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(r));
        int64_t blocks = (B + MISC_BLOCK_SIZE - 1) / MISC_BLOCK_SIZE;
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32:
                multinomial_rows_kernel<float><<<blocks, MISC_BLOCK_SIZE>>>(B, K, num_samples, reinterpret_cast<const float *>(mag_tensor_data_ptr(x)), br, seed, subseq, reinterpret_cast<mag_discrete_sample_pair_d *>(d_ws));
                break;
            case MAG_DTYPE_FLOAT16:
                multinomial_rows_kernel<half><<<blocks, MISC_BLOCK_SIZE>>>(B, K, num_samples, reinterpret_cast<const half *>(mag_tensor_data_ptr(x)), br, seed, subseq, reinterpret_cast<mag_discrete_sample_pair_d *>(d_ws));
                break;
            case MAG_DTYPE_BFLOAT16:
                multinomial_rows_kernel<__nv_bfloat16><<<blocks, MISC_BLOCK_SIZE>>>(B, K, num_samples, reinterpret_cast<const __nv_bfloat16 *>(mag_tensor_data_ptr(x)), br, seed, subseq, reinterpret_cast<mag_discrete_sample_pair_d *>(d_ws));
                break;
            default: mag_assert(false, "multinomial: unsupported dtype");
        }
        cuda_check(cudaFree(d_ws), "multinomial cudaFree");
    }

} /* namespace mag */
