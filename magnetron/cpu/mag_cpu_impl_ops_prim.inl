/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#define mag_gen_stub_binop(T, TF, FUNC, OPF, CVT, RCVT) \
    static void MAG_HOTPROC mag_##FUNC##_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_tensor_t *y = mag_cmd_in(1); \
        T *br = (T *)mag_tensor_data_ptr_mut(r); \
        const T *bx = (const T *)mag_tensor_data_ptr(x); \
        const T *by = (const T *)mag_tensor_data_ptr(y); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t total = r->numel; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_full_cont3(r, x, y)) { \
            mag_v##FUNC##_##TF(rb-ra, br+ra, bx+ra, by+ra); \
            return; \
        } \
        mag_coords_iter_t cr, cx, cy; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        mag_coords_iter_init(&cy, &y->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, xi, yi; \
            mag_coords_iter_offset3(&cr, &cx, &cy, i, &ri, &xi, &yi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
            mag_bnd_chk(by+yi, by, mag_tensor_numbytes(y)); \
            mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
            br[ri] = RCVT(OPF(T, CVT(bx[xi]), CVT(by[yi]))); \
        } \
    }

#define mag_cvt_nop(x) (x)

#define mag_opf_add(T, x, y) ((x)+(y))
#define mag_opf_sub(T, x, y) ((x)-(y))
#define mag_opf_mul(T, x, y) ((x)*(y))
#define mag_opf_div(T, x, y) ((x)/(y))
#define mag_opf_ifloordiv(T, x, y) (mag_floordivi(x, y))
#define mag_opf_ufloordiv(T, x, y) (mag_floordivu(x, y))
#define mag_opf_ffloordiv(T, x, y) (mag_floordivf(x, y))
#define mag_opf_imod(T, x, y) (mag_remi(x, y))
#define mag_opf_fmod(T, x, y) (mag_remf(x, y))
#define mag_opf_umod(T, x, y) (mag_remu(x, y))
#define mag_opf_and(T, x, y) ((x)&(y))
#define mag_opf_or(T, x, y) ((x)|(y))
#define mag_opf_xor(T, x, y) ((x)^(y))
#define mag_opf_sal(T, x, y) mag_sal(x, y, sizeof(T)<<3)
#define mag_opf_sar(T, x, y) mag_sar(x, y, sizeof(T)<<3)
#define mag_opf_shl(T, x, y) mag_shl(x, y, sizeof(T)<<3)
#define mag_opf_shr(T, x, y) mag_shr(x, y, sizeof(T)<<3)

mag_gen_stub_binop(float, float32, add, mag_opf_add, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(mag_float16_t, float16, add, mag_opf_add, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_stub_binop(uint8_t, uint8, add, mag_opf_add, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, add, mag_opf_add, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, add, mag_opf_add, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, add, mag_opf_add, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, add, mag_opf_add, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, add, mag_opf_add, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, add, mag_opf_add, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, add, mag_opf_add, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(float, float32, sub, mag_opf_sub, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(mag_float16_t, float16, sub, mag_opf_sub, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_stub_binop(uint8_t, uint8, sub, mag_opf_sub, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, sub, mag_opf_sub, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, sub, mag_opf_sub, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, sub, mag_opf_sub, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, sub, mag_opf_sub, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, sub, mag_opf_sub, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, sub, mag_opf_sub, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, sub, mag_opf_sub, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(float, float32, mul, mag_opf_mul, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(mag_float16_t, float16, mul, mag_opf_mul, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_stub_binop(uint8_t, uint8, mul, mag_opf_mul, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, mul, mag_opf_mul, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, mul, mag_opf_mul, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, mul, mag_opf_mul, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, mul, mag_opf_mul, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, mul, mag_opf_mul, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, mul, mag_opf_mul, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, mul, mag_opf_mul, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(float, float32, div, mag_opf_div, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(mag_float16_t, float16, div, mag_opf_div, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_stub_binop(uint8_t, uint8, div, mag_opf_div, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, div, mag_opf_div, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, div, mag_opf_div, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, div, mag_opf_div, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, div, mag_opf_div, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, div, mag_opf_div, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, div, mag_opf_div, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, div, mag_opf_div, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(float, float32, floordiv, mag_opf_ffloordiv, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(mag_float16_t, float16, floordiv, mag_opf_ffloordiv, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_stub_binop(uint8_t, uint8, floordiv, mag_opf_ufloordiv, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, floordiv, mag_opf_ifloordiv, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, floordiv, mag_opf_ufloordiv, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, floordiv, mag_opf_ifloordiv, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, floordiv, mag_opf_ufloordiv, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, floordiv, mag_opf_ifloordiv, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, floordiv, mag_opf_ufloordiv, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, floordiv, mag_opf_ifloordiv, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(float, float32, mod, mag_opf_fmod, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(mag_float16_t, float16, mod, mag_opf_fmod, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_stub_binop(uint8_t, uint8, mod, mag_opf_umod, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, mod, mag_opf_imod, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, mod, mag_opf_umod, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, mod, mag_opf_imod, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, mod, mag_opf_umod, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, mod, mag_opf_imod, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, mod, mag_opf_umod, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, mod, mag_opf_imod, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(uint8_t, uint8, and, mag_opf_and, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, and, mag_opf_and, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, and, mag_opf_and, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, and, mag_opf_and, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, and, mag_opf_and, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, and, mag_opf_and, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, and, mag_opf_and, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, and, mag_opf_and, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(uint8_t, uint8, or, mag_opf_or, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, or, mag_opf_or, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, or, mag_opf_or, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, or, mag_opf_or, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, or, mag_opf_or, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, or, mag_opf_or, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, or, mag_opf_or, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, or, mag_opf_or, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(uint8_t, uint8, xor, mag_opf_xor, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, xor, mag_opf_xor, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, xor, mag_opf_xor, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, xor, mag_opf_xor, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, xor, mag_opf_xor, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, xor, mag_opf_xor, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, xor, mag_opf_xor, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, xor, mag_opf_xor, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(uint8_t, uint8, shl, mag_opf_shl, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, shl, mag_opf_sal, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, shl, mag_opf_shl, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, shl, mag_opf_sal, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, shl, mag_opf_shl, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, shl, mag_opf_sal, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, shl, mag_opf_shl, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, shl, mag_opf_sal, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(uint8_t, uint8, shr, mag_opf_shr, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int8_t, int8, shr, mag_opf_sar, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint16_t, uint16, shr, mag_opf_shr, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int16_t, int16, shr, mag_opf_sar, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint32_t, uint32, shr, mag_opf_shr, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int32_t, int32, shr, mag_opf_sar, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(uint64_t, uint64, shr, mag_opf_shr, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(int64_t, int64, shr, mag_opf_sar, mag_cvt_nop, mag_cvt_nop)

#undef mag_opf_add
#undef mag_opf_sub
#undef mag_opf_mul
#undef mag_opf_div
#undef mag_opf_mod
#undef mag_opf_fmod
#undef mag_opf_and
#undef mag_opf_or
#undef mag_opf_xor
#undef mag_opf_shl
#undef mag_opf_shr

#undef mag_gen_stub_binop

#define mag_gen_stub_cmp(FUNC, T, TF, OP, CVT) \
    static void MAG_HOTPROC mag_##FUNC##_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_tensor_t *y = mag_cmd_in(1); \
        uint8_t *br = (uint8_t *)mag_tensor_data_ptr_mut(r); \
        const T *bx = (const T *)mag_tensor_data_ptr(x); \
        const T *by = (const T *)mag_tensor_data_ptr(y); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t total = r->numel; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_full_cont3(r, x, y)) { \
            mag_v##FUNC##_##TF(rb-ra, br+ra, bx+ra, by+ra); \
            return; \
        } \
        mag_coords_iter_t cr, cx, cy; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        mag_coords_iter_init(&cy, &y->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, xi, yi; \
            mag_coords_iter_offset3(&cr, &cx, &cy, i, &ri, &xi, &yi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
            mag_bnd_chk(by+yi, by, mag_tensor_numbytes(y)); \
            mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
            br[ri] = CVT(bx[xi]) OP CVT(by[yi]); \
        } \
    }

mag_gen_stub_cmp(eq, float, float32, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, mag_float16_t, float16, ==, mag_float16_to_float32)
mag_gen_stub_cmp(eq, uint8_t, uint8, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, int8_t, int8, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, uint16_t, uint16, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, int16_t, int16, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, uint32_t, uint32, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, int32_t, int32, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, uint64_t, uint64, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, int64_t, int64, ==, mag_cvt_nop)

mag_gen_stub_cmp(ne, float, float32, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, mag_float16_t, float16, !=, mag_float16_to_float32)
mag_gen_stub_cmp(ne, uint8_t, uint8, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, int8_t, int8, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, uint16_t, uint16, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, int16_t, int16, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, uint32_t, uint32, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, int32_t, int32, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, uint64_t, uint64, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, int64_t, int64, !=, mag_cvt_nop)

mag_gen_stub_cmp(lt, float, float32, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, mag_float16_t, float16, <, mag_float16_to_float32)
mag_gen_stub_cmp(lt, uint8_t, uint8, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, int8_t, int8, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, uint16_t, uint16, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, int16_t, int16, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, uint32_t, uint32, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, int32_t, int32, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, uint64_t, uint64, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, int64_t, int64, <, mag_cvt_nop)

mag_gen_stub_cmp(gt, float, float32, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, mag_float16_t, float16, >, mag_float16_to_float32)
mag_gen_stub_cmp(gt, uint8_t, uint8, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, int8_t, int8, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, uint16_t, uint16, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, int16_t, int16, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, uint32_t, uint32, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, int32_t, int32, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, uint64_t, uint64, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, int64_t, int64, >, mag_cvt_nop)

mag_gen_stub_cmp(le, float, float32, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, mag_float16_t, float16, <=, mag_float16_to_float32)
mag_gen_stub_cmp(le, uint8_t, uint8, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, int8_t, int8, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, uint16_t, uint16, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, int16_t, int16, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, uint32_t, uint32, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, int32_t, int32, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, uint64_t, uint64, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, int64_t, int64, <=, mag_cvt_nop)

mag_gen_stub_cmp(ge, float, float32, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, mag_float16_t, float16, >=, mag_float16_to_float32)
mag_gen_stub_cmp(ge, uint8_t, uint8, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, int8_t, int8, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, uint16_t, uint16, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, int16_t, int16, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, uint32_t, uint32, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, int32_t, int32, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, uint64_t, uint64, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, int64_t, int64, >=, mag_cvt_nop)
