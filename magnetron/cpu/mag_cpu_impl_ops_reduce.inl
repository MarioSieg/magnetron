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

typedef struct mag_var_acc_t { /* Variance accumulation state */
    double mean;
    double M2;
    int64_t n;
} mag_var_acc_t;

#define mag_cpu_impl_reduce_axes(T, OT, TF, FUNC, ACC_T, INIT_EXPR, UPDATE_STMT, FINAL_STMT) \
    static void MAG_HOTPROC mag_##FUNC##_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        OT *br = (OT *)mag_tensor_data_ptr_mut(r); \
        const T *bx = (const T *)mag_tensor_data_ptr(x); \
        mag_reduce_plan_t *plan = mag_op_attr_unwrap_ptr(mag_cmd_attr(0)); \
        int64_t numel = r->numel; \
        int64_t red_prod = plan->red_prod; \
        for (int64_t oi=0; oi < numel; ++oi) { \
            int64_t base = mag_reduce_plan_to_offset(plan, oi); \
            ACC_T acc = INIT_EXPR; \
            for (int64_t ri=0; ri < red_prod; ++ri) { \
                int64_t tmp = ri; \
                int64_t roff = base; \
                for (int64_t k=plan->rank - 1; k >= 0; --k) { \
                    int64_t sz = plan->red_sizes[k]; \
                    int64_t idx = tmp % sz; \
                    tmp /= sz; \
                    roff += idx*plan->red_strides[k]; \
                } \
                mag_bnd_chk(bx + roff, bx, mag_tensor_numbytes(x)); \
                { UPDATE_STMT } \
            } \
            OT *o = br + oi; \
            { FINAL_STMT } \
        } \
    }

mag_cpu_impl_reduce_axes(float, float, float32, mean, double, 0.0, acc += (double)bx[roff];, acc /= (double)red_prod; *o = (float)acc; )
mag_cpu_impl_reduce_axes(mag_float16_t, mag_float16_t, float16, mean, float, 0.0f, acc += mag_float16_to_float32(bx[roff]);, acc /= (float)red_prod; *o = mag_float32_to_float16(acc); )

mag_cpu_impl_reduce_axes(float, float, float32, sum, double, 0.0, acc += (double)bx[roff];, *o = (float)acc; )
mag_cpu_impl_reduce_axes(mag_float16_t, mag_float16_t, float16, sum, float, 0.0f,acc += mag_float16_to_float32(bx[roff]);,*o = mag_float32_to_float16(acc); )
mag_cpu_impl_reduce_axes(uint8_t, uint64_t, uint8, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int64_t, int8, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint64_t, uint16, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int64_t, int16, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint64_t, uint32, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int64_t, int32, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, sum, uint64_t, 0, acc += (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, sum, int64_t, 0, acc += (int64_t)bx[roff];, *o = acc; )

mag_cpu_impl_reduce_axes(float, float, float32, prod, double, 1.0, acc *= (double)bx[roff];, *o = (float)acc; )
mag_cpu_impl_reduce_axes(mag_float16_t, mag_float16_t, float16, prod, float, 1.0f, acc *= mag_float16_to_float32(bx[roff]);, *o = mag_float32_to_float16(acc); )
mag_cpu_impl_reduce_axes(uint8_t, uint64_t, uint8, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int64_t, int8, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint64_t, uint16, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int64_t, int16, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint64_t, uint32, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int64_t, int32, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, prod, uint64_t, 1, acc *= (uint64_t)bx[roff];, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, prod, int64_t, 1, acc *= (int64_t)bx[roff];, *o = acc; )

mag_cpu_impl_reduce_axes(float, float, float32, min, float, INFINITY, acc = fminf(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(mag_float16_t, mag_float16_t, float16, min, float, INFINITY, acc = fminf(acc, mag_float16_to_float32(bx[roff]));, *o = mag_float32_to_float16(acc); )
mag_cpu_impl_reduce_axes(uint8_t, uint8_t, uint8, min, uint8_t, UINT8_MAX, acc = mag_xmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int8_t, int8, min, int8_t, INT8_MAX, acc = mag_xmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint16_t, uint16, min, uint16_t, UINT16_MAX, acc = mag_xmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int16_t, int16, min, int16_t, INT16_MAX, acc = mag_xmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint32_t, uint32, min, uint32_t, UINT32_MAX, acc = mag_xmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int32_t, int32, min, int32_t, INT32_MAX, acc = mag_xmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, min, uint64_t, UINT64_MAX, acc = mag_xmin(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, min, int64_t, INT64_MAX, acc = mag_xmin(acc, bx[roff]);, *o = acc; )

mag_cpu_impl_reduce_axes(float, float, float32, max, float, -INFINITY, acc = fmaxf(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(mag_float16_t, mag_float16_t, float16, max, float, -INFINITY, acc = fmaxf(acc, mag_float16_to_float32(bx[roff]));, *o = mag_float32_to_float16(acc); )
mag_cpu_impl_reduce_axes(uint8_t, uint8_t, uint8, max, uint8_t, 0, acc = mag_xmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int8_t, int8_t, int8, max, int8_t, INT8_MIN, acc = mag_xmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint16_t, uint16_t, uint16, max, uint16_t, 0, acc = mag_xmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int16_t, int16_t, int16, max, int16_t, INT16_MIN, acc = mag_xmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint32_t, uint32_t, uint32, max, uint32_t, 0, acc = mag_xmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int32_t, int32_t, int32, max, int32_t, INT32_MIN, acc = mag_xmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(uint64_t, uint64_t, uint64, max, uint64_t, 0, acc = mag_xmax(acc, bx[roff]);, *o = acc; )
mag_cpu_impl_reduce_axes(int64_t, int64_t, int64, max, int64_t, INT64_MIN, acc = mag_xmax(acc, bx[roff]);, *o = acc; )

typedef struct mag_argmax_acc_f32_t {
    float val;
    int64_t idx;
    bool set;
} mag_argmax_acc_f32_t;

typedef struct mag_argmax_acc_i64_t {
    int64_t val;
    int64_t idx;
    bool set;
} mag_argmax_acc_i64_t;

mag_cpu_impl_reduce_axes(
    float,
    int64_t,
    float32,
    argmax,
    mag_argmax_acc_f32_t,
    {0},
    {
        float xv = bx[roff];
        if (!acc.set || xv > acc.val) {
            acc.val = xv;
            acc.idx = ri;
            acc.set = true;
        }
    },
    {
        *o = acc.idx;
    }
);

mag_cpu_impl_reduce_axes(
    float,
    int64_t,
    float32,
    argmin,
    mag_argmax_acc_f32_t,
    {0},
    {
        float xv = bx[roff];
        if (!acc.set || xv < acc.val) {
            acc.val = xv;
            acc.idx = ri;
            acc.set = true;
        }
    },
    {
        *o = acc.idx;
    }
);

mag_cpu_impl_reduce_axes(
    mag_float16_t,
    int64_t,
    float16,
    argmax,
    mag_argmax_acc_f32_t,
    {0},
    {
        float xv = mag_float16_to_float32(bx[roff]);
        if (!acc.set || xv > acc.val) {
            acc.val = xv;
            acc.idx = ri;
            acc.set = true;
        }
    },
    {
        *o = acc.idx;
    }
);

mag_cpu_impl_reduce_axes(
    mag_float16_t,
    int64_t,
    float16,
    argmin,
    mag_argmax_acc_f32_t,
    {0},
    {
        float xv = mag_float16_to_float32(bx[roff]);
        if (!acc.set || xv < acc.val) {
            acc.val = xv;
            acc.idx = ri;
            acc.set = true;
        }
    },
    {
        *o = acc.idx;
    }
);

#define mag_cpu_impl_argminmax_int(T, TF) \
    mag_cpu_impl_reduce_axes( \
        T, int64_t, TF, argmax, mag_argmax_acc_i64_t, \
        {0}, \
        { \
            int64_t xv = (int64_t)bx[roff]; \
            if (!acc.set || xv > acc.val) { \
                acc.val = xv; \
                acc.idx = ri; \
                acc.set = true; \
            } \
        }, \
        { *o = acc.idx; } \
    ); \
    mag_cpu_impl_reduce_axes( \
        T, int64_t, TF, argmin, mag_argmax_acc_i64_t, \
        {0}, \
        { \
            int64_t xv = (int64_t)bx[roff]; \
            if (!acc.set || xv < acc.val) { \
                acc.val = xv; \
                acc.idx = ri; \
                acc.set = true; \
            } \
        }, \
        { *o = acc.idx; } \
    )

mag_cpu_impl_argminmax_int(uint8_t,  uint8);
mag_cpu_impl_argminmax_int(int8_t,   int8);
mag_cpu_impl_argminmax_int(uint16_t, uint16);
mag_cpu_impl_argminmax_int(int16_t,  int16);
mag_cpu_impl_argminmax_int(uint32_t, uint32);
mag_cpu_impl_argminmax_int(int32_t,  int32);
mag_cpu_impl_argminmax_int(uint64_t, uint64);
mag_cpu_impl_argminmax_int(int64_t,  int64);

#undef mag_cpu_impl_argminmax_int

#undef mag_cpu_impl_reduce_axes

#define mag_cpu_impl_reduce_axes_logical(T, TF, FUNC, IDENTITY, UPDATE_STMT, BREAK_COND) \
    static void MAG_HOTPROC mag_##FUNC##_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        uint8_t *br = (uint8_t *)mag_tensor_data_ptr_mut(r); \
        const T *bx = (const T *)mag_tensor_data_ptr(x); \
        mag_reduce_plan_t *plan = mag_op_attr_unwrap_ptr(mag_cmd_attr(0)); \
        int64_t numel = r->numel; \
        int64_t red_prod = plan->red_prod; \
        for (int64_t oi=0; oi < numel; ++oi) { \
            uint8_t acc = (IDENTITY); \
            if (red_prod == 0) { \
                br[oi] = acc; \
                continue; \
            } \
            int64_t base = mag_reduce_plan_to_offset(plan, oi); \
            for (int64_t ri=0; ri < red_prod; ++ri) { \
                int64_t tmp = ri; \
                int64_t roff = base; \
                for (int64_t k=plan->rank-1; k >= 0; --k) { \
                    int64_t sz = plan->red_sizes[k]; \
                    int64_t idx = tmp % sz; \
                    tmp /= sz; \
                    roff += idx*plan->red_strides[k]; \
                } \
                mag_bnd_chk(bx + roff, bx, mag_tensor_numbytes(x)); \
                { UPDATE_STMT } \
                if (BREAK_COND) break; \
            } \
            br[oi] = acc; \
        } \
    }


#define mag_impl_logical_reduce_pair(T, TF, unpack) \
    mag_cpu_impl_reduce_axes_logical( \
        T, TF, any, \
        0, \
        { if (unpack(bx[roff]) != 0) acc = 1; }, \
        acc == 1 \
    ); \
    mag_cpu_impl_reduce_axes_logical( \
        T, TF, all, \
        1, \
        { if (unpack(bx[roff]) == 0) acc = 0; }, \
        acc == 0 \
    )

#define mag_unpack_nop(x) (x)
#define mag_unpack_packed(x) ((x).bits)

mag_impl_logical_reduce_pair(float, float32, mag_unpack_nop);
mag_impl_logical_reduce_pair(mag_float16_t, float16, mag_unpack_packed);
mag_impl_logical_reduce_pair(uint8_t, uint8, mag_unpack_nop);
mag_impl_logical_reduce_pair(int8_t, int8, mag_unpack_nop);
mag_impl_logical_reduce_pair(uint16_t, uint16, mag_unpack_nop);
mag_impl_logical_reduce_pair(int16_t, int16, mag_unpack_nop);
mag_impl_logical_reduce_pair(uint32_t, uint32, mag_unpack_nop);
mag_impl_logical_reduce_pair(int32_t, int32, mag_unpack_nop);
mag_impl_logical_reduce_pair(uint64_t, uint64, mag_unpack_nop);
mag_impl_logical_reduce_pair(int64_t, int64, mag_unpack_nop);

#undef mag_unpack_nop
#undef mag_unpack_packed

#undef mag_impl_logical_reduce_pair
