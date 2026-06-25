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

#define mag_gen_stub_index_add(T, TF, CVT, ACC_T, MUL, FROM_ACC) \
  static MAG_HOTPROC mag_status_t mag_index_add_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)payload; \
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *src = mag_cmd_in(1); \
    const mag_tensor_t *idx = mag_cmd_in(2); \
    T *bs = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(src); \
    const int64_t *bi = (const int64_t *)mag_tensor_data_ptr(idx); \
    int64_t ax = mag_op_attr_unwrap_int64(mag_cmd_attr(0)); \
    double alpha = mag_op_attr_unwrap_float64(mag_cmd_attr(1)); \
    if (ax < 0) ax += r->coords.rank; \
    mag_assert2(ax >= 0 && ax < r->coords.rank); \
    int64_t ra = r->coords.rank; \
    int64_t rra = r->coords.shape[ax]; \
    int64_t total = src->numel; \
    if (payload->thread_idx != 0) return MAG_STATUS_OK; \
    for (int64_t flat=0; flat < total; ++flat) { \
      int64_t tmp = flat; \
      int64_t sc[MAG_MAX_DIMS]; \
      for (int64_t dx = ra-1; dx >= 0; --dx) { \
        sc[dx] = tmp % src->coords.shape[dx]; \
        tmp /= src->coords.shape[dx]; \
      } \
      int64_t j = sc[ax]; \
      int64_t idx_off = j*idx->coords.strides[0]; \
      int64_t g = bi[idx_off]; \
      if (g < 0) g += rra; \
      mag_contract(err, ERR_KERNEL_FAILURE, {}, g >= 0 && g < rra, "index_add_: idx %" PRIi64 " is out of range [0, %" PRIi64 ").", g, rra); \
      int64_t src_off = 0; \
      for (int64_t dx=0; dx < ra; ++dx) src_off += sc[dx]*src->coords.strides[dx]; \
      sc[ax] = g; \
      int64_t dst_off = 0; \
      for (int64_t dx=0; dx < ra; ++dx) dst_off += sc[dx]*r->coords.strides[dx]; \
      mag_bnd_chk(bs+dst_off, bs, mag_tensor_numbytes(r)); \
      mag_bnd_chk(bx+src_off, bx, mag_tensor_numbytes(src)); \
      bs[dst_off] = FROM_ACC((ACC_T)(CVT(bs[dst_off])) + (ACC_T)(MUL(CVT(bx[src_off]), alpha))); \
    } \
    return MAG_STATUS_OK; \
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
