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

#define mag_gen_stub_binop(T, FUNC, OP, CVT, RCVT) \
    static void MAG_HOTPROC mag_##FUNC##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_tensor_t *y = mag_cmd_in(1); \
        mag_##T##_t *br = mag_##T##p_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        const mag_##T##_t *by = mag_##T##p(y); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t total = r->numel; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_full_cont3(r, x, y)) { \
            mag_v##FUNC##_##T(rb-ra, br+ra, bx+ra, by+ra); \
            return; \
        } \
        mag_coords_iter_t cr, cx, cy; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        mag_coords_iter_init(&cy, &y->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, xi, yi; \
            mag_coords_iter_offset3(&cr, &cx, &cy, i, &ri, &xi, &yi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y)); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = RCVT(CVT(bx[xi]) OP CVT(by[yi])); \
        } \
    }

#define mag_cvt_nop(x) (x)

mag_gen_stub_binop(e8m23, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e5m10, add, +, mag_e5m10_cvt_e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_binop(u8,  add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8,  add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, add, +, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e8m23, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e5m10, sub, -, mag_e5m10_cvt_e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_binop(u8,  sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8,  sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, sub, -, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e8m23, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e5m10, mul, *, mag_e5m10_cvt_e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_binop(u8,  mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8,  mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, mul, *, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e8m23, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(e5m10, div, /, mag_e5m10_cvt_e8m23, mag_e8m23_cvt_e5m10)
mag_gen_stub_binop(u8,  div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8,  div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, div, /, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, div, /, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(u8, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u8, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u8, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u8, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u8, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i8, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u16, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i16, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u32, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i32, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(u64, shr, >>, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, xor, ^, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, shl, <<, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(i64, shr, >>, mag_cvt_nop, mag_cvt_nop)

mag_gen_stub_binop(bool, and, &, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(bool, or, |, mag_cvt_nop, mag_cvt_nop)
mag_gen_stub_binop(bool, xor, ^, mag_cvt_nop, mag_cvt_nop)

#undef mag_gen_stub_binop

#define mag_gen_stub_cmp(FUNC, T, OP, CVT) \
    static void MAG_HOTPROC mag_##FUNC##_##T(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        const mag_tensor_t *y = mag_cmd_in(1); \
        mag_bool_t *br = mag_boolp_mut(r); \
        const mag_##T##_t *bx = mag_##T##p(x); \
        const mag_##T##_t *by = mag_##T##p(y); \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t total = r->numel; \
        int64_t chunk = (total + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, total); \
        if (mag_full_cont3(r, x, y)) { \
            mag_v##FUNC##_##T(rb-ra, br+ra, bx+ra, by+ra); \
            return; \
        } \
        mag_coords_iter_t cr, cx, cy; \
        mag_coords_iter_init(&cr, &r->coords); \
        mag_coords_iter_init(&cx, &x->coords); \
        mag_coords_iter_init(&cy, &y->coords); \
        for (int64_t i=ra; i < rb; ++i) { \
            int64_t ri, xi, yi; \
            mag_coords_iter_offset3(&cr, &cx, &cy, i, &ri, &xi, &yi); \
            mag_bnd_chk(bx+xi, bx, mag_tensor_get_data_size(x)); \
            mag_bnd_chk(by+yi, by, mag_tensor_get_data_size(y)); \
            mag_bnd_chk(br+ri, br, mag_tensor_get_data_size(r)); \
            br[ri] = CVT(bx[xi]) OP CVT(by[yi]); \
        } \
    }

mag_gen_stub_cmp(eq, e8m23, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, e5m10, ==, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(eq, u8,  ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, i8,  ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, u16, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, i16, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, u32, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, i32, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, u64, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, i64, ==, mag_cvt_nop)
mag_gen_stub_cmp(eq, bool, ==, mag_cvt_nop)
mag_gen_stub_cmp(ne, e8m23, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, e5m10, !=, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(ne, u8,  !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, i8,  !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, u16, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, i16, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, u32, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, i32, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, u64, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, i64, !=, mag_cvt_nop)
mag_gen_stub_cmp(ne, bool, !=, mag_cvt_nop)

mag_gen_stub_cmp(lt, e8m23, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, e5m10, <, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(lt, u8,  <, mag_cvt_nop)
mag_gen_stub_cmp(lt, i8,  <, mag_cvt_nop)
mag_gen_stub_cmp(lt, u16, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, i16, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, u32, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, i32, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, u64, <, mag_cvt_nop)
mag_gen_stub_cmp(lt, i64, <, mag_cvt_nop)
mag_gen_stub_cmp(gt, e8m23, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, e5m10, >, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(gt, u8,  >, mag_cvt_nop)
mag_gen_stub_cmp(gt, i8,  >, mag_cvt_nop)
mag_gen_stub_cmp(gt, u16, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, i16, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, u32, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, i32, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, u64, >, mag_cvt_nop)
mag_gen_stub_cmp(gt, i64, >, mag_cvt_nop)
mag_gen_stub_cmp(le, e8m23, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, e5m10, <=, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(le, u8,  <=, mag_cvt_nop)
mag_gen_stub_cmp(le, i8,  <=, mag_cvt_nop)
mag_gen_stub_cmp(le, u16, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, i16, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, u32, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, i32, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, u64, <=, mag_cvt_nop)
mag_gen_stub_cmp(le, i64, <=, mag_cvt_nop)
mag_gen_stub_cmp(ge, e8m23, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, e5m10, >=, mag_e5m10_cvt_e8m23)
mag_gen_stub_cmp(ge, u8,  >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, i8,  >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, u16, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, i16, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, u32, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, i32, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, u64, >=, mag_cvt_nop)
mag_gen_stub_cmp(ge, i64, >=, mag_cvt_nop)
