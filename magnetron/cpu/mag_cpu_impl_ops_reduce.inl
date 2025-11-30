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

#include <core/mag_reduce_plan.h>

#define mag_cpu_impl_reduce_axes(T, TF, FUNC, ACC_T, INIT_EXPR, UPDATE_STMT, FINAL_STMT) \
    static void MAG_HOTPROC mag_##FUNC##_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const T *bx = mag_tensor_get_data_ptr(x); \
        T *br =  mag_tensor_get_data_ptr(r); \
        mag_reduce_plan_t *plan = mag_op_attr_unwrap_ptr(mag_cmd_attr(0)); \
        int64_t out_numel = r->numel; \
        int64_t red_prod = plan->red_prod; \
        for (int64_t oi=0; oi < out_numel; ++oi) { \
            int64_t base = mag_reduce_plan_to_offset(plan, oi); \
            ACC_T acc = (INIT_EXPR); \
            for (int64_t ri=0; ri < red_prod; ++ri) { \
                int64_t tmp = ri; \
                int64_t roff = base; \
                for (int64_t k=plan->rank - 1; k >= 0; --k) { \
                    int64_t sz = plan->red_sizes[k]; \
                    int64_t idx = tmp % sz; \
                    tmp /= sz; \
                    roff += idx*plan->red_strides[k]; \
                } \
                mag_bnd_chk(bx + roff, bx, mag_tensor_get_data_size(x)); \
                { UPDATE_STMT } \
            } \
            T *outp = br + oi; \
            { FINAL_STMT } \
        } \
    }

mag_cpu_impl_reduce_axes( \
                          float, float32, sum, double, 0.0, \
                          acc += (double)bx[roff];, \
                          *outp = (float)acc; )
mag_cpu_impl_reduce_axes( \
                          mag_float16_t, float16, sum, float, 0.0f, \
                          acc += mag_float16_to_float32(bx[roff]);, \
                          *outp = mag_float32_to_float16(acc); )

mag_cpu_impl_reduce_axes( \
                          float, float32, prod, double, 1.0, \
                          acc *= (double)bx[roff];, \
                          *outp = (float)acc; )
mag_cpu_impl_reduce_axes( \
                          mag_float16_t, float16, prod, float, 1.0f, \
                          acc *= mag_float16_to_float32(bx[roff]);, \
                          *outp = mag_float32_to_float16(acc); )

mag_cpu_impl_reduce_axes( \
                          float, float32, mean, double, 0.0, \
                          acc += (double)bx[roff];, \
                          acc /= (double)red_prod; *outp = (float)acc; )
mag_cpu_impl_reduce_axes( \
                          mag_float16_t, float16, mean, float, 0.0f, \
                          acc += mag_float16_to_float32(bx[roff]);, \
                          acc /= (float)red_prod; *outp = mag_float32_to_float16(acc); )

mag_cpu_impl_reduce_axes( \
                          float, float32, min, float, INFINITY, \
                          acc = fminf(acc, bx[roff]);, \
                          *outp = acc; )
mag_cpu_impl_reduce_axes( \
                          mag_float16_t, float16, min, float, INFINITY, \
                          acc = fminf(acc, mag_float16_to_float32(bx[roff]));, \
                          *outp = mag_float32_to_float16(acc); )

mag_cpu_impl_reduce_axes( \
                          float, float32, max, float, -INFINITY, \
                          acc = fmaxf(acc, bx[roff]);, \
                          *outp = acc; )
mag_cpu_impl_reduce_axes( \
                          mag_float16_t, float16, max, float, -INFINITY, \
                          acc = fmaxf(acc, mag_float16_to_float32(bx[roff]));, \
                          *outp = mag_float32_to_float16(acc); )

#undef mag_cpu_impl_reduce_axes
