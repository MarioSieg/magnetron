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

static int mag_cmp_i64(const void *a, const void *b) {
    int64_t da = *(const int64_t *)a, db = *(const int64_t *)b;
    return (da > db) - (da < db);
}

#define mag_cpu_impl_reduce_axes(T, FUNC, ACC_T, INIT_EXPR, UPDATE_STMT, FINAL_STMT) \
    static void MAG_HOTPROC mag_##FUNC##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        int64_t nd = x->coords.rank; \
        int64_t rank = mag_op_attr_unwrap_i64(mag_cmd_attr(0)); \
        int64_t axes[MAG_MAX_DIMS]; \
        for (int64_t i=0; i<rank; ++i){ \
            int64_t a=mag_op_attr_unwrap_i64(mag_cmd_attr(2+i)); \
            if (a<0) a += nd; \
            axes[i]=a; \
        } \
        qsort(axes, (size_t)rank, sizeof(int64_t), &mag_cmp_i64); \
        int64_t rr=0; for(int64_t i=0;i<rank;++i) if(i==0||axes[i]!=axes[i-1]) axes[rr++]=axes[i]; \
        rank=rr; \
        int64_t keep_axes[MAG_MAX_DIMS]; \
        int64_t nk=0; \
        for (int64_t d=0; d < nd; ++d) { \
            bool red = false; \
            for (int64_t k=0; k < rank; ++k) \
                if (axes[k]==d) { red = true; break; } \
            if (!red) keep_axes[nk++] = d; \
        } \
        int64_t out_numel = r->numel; \
        int64_t red_prod = 1; (void)red_prod; \
        for (int64_t k=0; k < rank; ++k) \
            red_prod *= x->coords.shape[axes[k]]; \
        for (int64_t oi=0; oi<out_numel; ++oi) { \
            int64_t rem = oi, off = 0; \
            for (int64_t j=nk-1; j >= 0; --j) { \
                int64_t ax = keep_axes[j]; \
                int64_t sz = x->coords.shape[ax]; \
                int64_t idx = (sz > 1) ? rem % sz : 0; \
                if (sz > 1) rem /= sz; \
                off += idx*x->coords.strides[ax]; \
            } \
            ACC_T acc = (INIT_EXPR); \
            int64_t ctr[MAG_MAX_DIMS] = {0}; \
            int64_t cur = off; \
            for (;;) { \
                int64_t roff = cur; \
                mag_bnd_chk(bx+roff, bx, mag_tensor_get_data_size(x)); \
                { UPDATE_STMT } \
                int64_t k=0; \
                for (; k < rank; ++k) { \
                    int64_t ax = axes[k]; \
                    if (++ctr[k] < x->coords.shape[ax]) { cur += x->coords.strides[ax]; break; } \
                    cur -= x->coords.strides[ax]*(x->coords.shape[ax]-1); \
                    ctr[k]=0; \
                } \
                if (k == rank) break; \
            } \
            mag_##T##_t *outp = br + oi; \
            { FINAL_STMT } \
        } \
    }

mag_cpu_impl_reduce_axes( \
                          e8m23, sum, mag_e11m52_t, 0.0, \
                          acc += (mag_e11m52_t)bx[roff];, \
                          *outp = (mag_e8m23_t)acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, sum, mag_e8m23_t, 0.0f, \
                          acc += mag_e5m10_cvt_e8m23(bx[roff]);, \
                          *outp = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_impl_reduce_axes( \
                          e8m23, mean, mag_e11m52_t, 0.0, \
                          acc += (mag_e11m52_t)bx[roff];, \
                          acc /= (mag_e11m52_t)red_prod; *outp = (mag_e8m23_t)acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, mean, mag_e8m23_t, 0.0f, \
                          acc += mag_e5m10_cvt_e8m23(bx[roff]);, \
                          acc /= (mag_e8m23_t)red_prod; *outp = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_impl_reduce_axes( \
                          e8m23, min, mag_e8m23_t, INFINITY, \
                          acc = fminf(acc, bx[roff]);, \
                          *outp = acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, min, mag_e8m23_t, INFINITY, \
                          acc = fminf(acc, mag_e5m10_cvt_e8m23(bx[roff]));, \
                          *outp = mag_e8m23_cvt_e5m10(acc); )

mag_cpu_impl_reduce_axes( \
                          e8m23, max, mag_e8m23_t, -INFINITY, \
                          acc = fmaxf(acc, bx[roff]);, \
                          *outp = acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, max, mag_e8m23_t, -INFINITY, \
                          acc = fmaxf(acc, mag_e5m10_cvt_e8m23(bx[roff]));, \
                          *outp = mag_e8m23_cvt_e5m10(acc); )

#undef mag_cpu_impl_reduce_axes
