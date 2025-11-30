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

#define mag_cpu_impl_reduce_axes(T, FUNC, ACC_T, INIT_EXPR, UPDATE_STMT, FINAL_STMT) \
    static void MAG_HOTPROC mag_##FUNC##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
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
                          acc += mag_e5m10_to_e8m23(bx[roff]);, \
                          *outp = mag_e8m23_to_e5m10(acc); )

mag_cpu_impl_reduce_axes( \
                          e8m23, mean, mag_e11m52_t, 0.0, \
                          acc += (mag_e11m52_t)bx[roff];, \
                          acc /= (mag_e11m52_t)red_prod; *outp = (mag_e8m23_t)acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, mean, mag_e8m23_t, 0.0f, \
                          acc += mag_e5m10_to_e8m23(bx[roff]);, \
                          acc /= (mag_e8m23_t)red_prod; *outp = mag_e8m23_to_e5m10(acc); )

mag_cpu_impl_reduce_axes( \
                          e8m23, min, mag_e8m23_t, INFINITY, \
                          acc = fminf(acc, bx[roff]);, \
                          *outp = acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, min, mag_e8m23_t, INFINITY, \
                          acc = fminf(acc, mag_e5m10_to_e8m23(bx[roff]));, \
                          *outp = mag_e8m23_to_e5m10(acc); )

mag_cpu_impl_reduce_axes( \
                          e8m23, max, mag_e8m23_t, -INFINITY, \
                          acc = fmaxf(acc, bx[roff]);, \
                          *outp = acc; )
mag_cpu_impl_reduce_axes( \
                          e5m10, max, mag_e8m23_t, -INFINITY, \
                          acc = fmaxf(acc, mag_e5m10_to_e8m23(bx[roff]));, \
                          *outp = mag_e8m23_to_e5m10(acc); )

#undef mag_cpu_impl_reduce_axes
