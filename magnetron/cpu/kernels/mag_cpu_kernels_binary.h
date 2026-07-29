/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

static MAG_AINLINE float mag_fn_add_f32(float x, float y) { return x+y; }
static MAG_AINLINE float mag_fn_sub_f32(float x, float y) { return x-y; }
static MAG_AINLINE float mag_fn_mul_f32(float x, float y) { return x*y; }
static MAG_AINLINE float mag_fn_div_f32(float x, float y) { return x/y; }
static MAG_AINLINE float mag_fn_floordiv_f32(float x, float y) { return floorf(x/y); }
static MAG_AINLINE float mag_fn_mod_f32(float x, float y) { return mag_remf(x,y); }
static MAG_AINLINE float mag_fn_pow_f32(float x, float y) { return powf(x,y); }
static MAG_AINLINE float mag_fn_min_f32(float x, float y) { return fminf(x,y); }
static MAG_AINLINE float mag_fn_max_f32(float x, float y) { return fmaxf(x,y); }

#define mag_def_float_bin_wrappers(name) \
  static MAG_AINLINE mag_float16_t mag_fn_##name##_f16(mag_float16_t x, mag_float16_t y) { return mag_float32_to_float16(mag_fn_##name##_f32(mag_float16_to_float32(x), mag_float16_to_float32(y))); } \
  static MAG_AINLINE mag_bfloat16_t mag_fn_##name##_bf16(mag_bfloat16_t x, mag_bfloat16_t y) { return mag_float32_to_bfloat16(mag_fn_##name##_f32(mag_bfloat16_to_float32(x), mag_bfloat16_to_float32(y))); } \
  static MAG_AINLINE mag_float8_e4m3fn_t mag_fn_##name##_f8_e4m3fn(mag_float8_e4m3fn_t x, mag_float8_e4m3fn_t y) { return mag_float32_to_float8_e4m3fn(mag_fn_##name##_f32(mag_float8_e4m3fn_to_float32(x), mag_float8_e4m3fn_to_float32(y))); }

mag_def_float_bin_wrappers(add)
mag_def_float_bin_wrappers(sub)
mag_def_float_bin_wrappers(mul)
mag_def_float_bin_wrappers(div)
mag_def_float_bin_wrappers(floordiv)
mag_def_float_bin_wrappers(mod)
mag_def_float_bin_wrappers(pow)
mag_def_float_bin_wrappers(min)
mag_def_float_bin_wrappers(max)

#undef mag_def_float_bin_wrappers

static MAG_AINLINE mag_vf32_t mag_vec_add_f32(mag_vf32_t x, mag_vf32_t y) { return mag_vf32_add(x,y); }
static MAG_AINLINE mag_vf32_t mag_vec_sub_f32(mag_vf32_t x, mag_vf32_t y) { return mag_vf32_sub(x,y); }
static MAG_AINLINE mag_vf32_t mag_vec_mul_f32(mag_vf32_t x, mag_vf32_t y) { return mag_vf32_mul(x,y); }
static MAG_AINLINE mag_vf32_t mag_vec_div_f32(mag_vf32_t x, mag_vf32_t y) { return mag_vf32_div(x,y); }

#define mag_fn_add_int(x,y) ((x)+(y))
#define mag_fn_sub_int(x,y) ((x)-(y))
#define mag_fn_mul_int(x,y) ((x)*(y))
#define mag_fn_div_int(x,y) ((x)/(y))
#define mag_fn_and_int(x,y) ((x)&(y))
#define mag_fn_or_int(x,y) ((x)|(y))
#define mag_fn_xor_int(x,y) ((x)^(y))
#define mag_fn_floordiv_i(x,y) ((((x) - mag_remi((x), (y)))/(y)))
#define mag_fn_floordiv_u(x,y) ((x)/(y))
#define mag_fn_mod_i(x,y) mag_remi((x),(y))
#define mag_fn_mod_u(x,y) ((x)%(y))/* Unsigned remainder is the same as in C */
#define mag_fn_pow_i(x,y) mag_powi((x),(y))
#define mag_fn_pow_u(x,y) mag_powu((x),(y))
#define mag_fn_shl_i(x,y,T) (mag_unlikely((y)<0 || (y)>=(sizeof(T)<<3)) ? 0 : (x)<<(y))
#define mag_fn_shl_u(x,y,T) (mag_unlikely((y)<0 || (y)>=(sizeof(T)<<3)) ? 0 : (x)<<(y))
#define mag_fn_shr_i(x,y,T) (mag_unlikely((y)<0 || (y)>=(sizeof(T)<<3)) ? 0 : (x)>>(y))
#define mag_fn_shr_u(x,y,T) (mag_unlikely((y)<0 || (y)>=(sizeof(T)<<3)) ? 0 : (x)>>(y))
#define mag_fn_min_int(x,y) ((x)<(y)?(x):(y))
#define mag_fn_max_int(x,y) ((x)>(y)?(x):(y))

#define mag_gen_bin_scalar(T, TF, name, suffix) \
  static mag_status_t MAG_HOTPROC mag_##name##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *y = payload->cmd->in[1]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *by = (const T *)mag_tensor_data_ptr(y); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra+chunk,total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[3]){r,x,y},3)) { \
      for (int64_t i=ra; i < rb; ++i) br[i] = mag_fn_##name##_##suffix(bx[i],by[i]); \
      return MAG_OK; \
    } \
    mag_coords_iter_t cr,cx,cy; \
    mag_coords_iter_init(&cr,&r->meta.coords); \
    mag_coords_iter_init(&cx,&x->meta.coords); \
    mag_coords_iter_init(&cy,&y->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri,xi,yi; \
      mag_coords_iter_offset3(&cr,&cx,&cy,i,&ri,&xi,&yi); \
      br[ri] = mag_fn_##name##_##suffix(bx[xi],by[yi]); \
    } \
    return MAG_OK; \
  }

#define mag_gen_bin_simd(T, TF, suffix, LOAD, STORE, name) \
  static mag_status_t MAG_HOTPROC mag_##name##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *y = payload->cmd->in[1]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *by = (const T *)mag_tensor_data_ptr(y); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra+chunk,total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[3]){r, x, y}, 3)) { \
      int64_t i=ra; \
      for (; i+MAG_VF32_LANES <= rb; i += MAG_VF32_LANES) { \
        mag_vf32_t vx = LOAD(bx+i); \
        mag_vf32_t vy = LOAD(by+i); \
        mag_vf32_t vr = mag_vec_##name##_f32(vx,vy); \
        STORE(br+i,vr); \
      } \
      for (; i < rb; ++i) br[i] = mag_fn_##name##_##suffix(bx[i],by[i]); \
      return MAG_OK; \
    } \
    mag_coords_iter_t cr,cx,cy; \
    mag_coords_iter_init(&cr,&r->meta.coords); \
    mag_coords_iter_init(&cx,&x->meta.coords); \
    mag_coords_iter_init(&cy,&y->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri,xi,yi; \
      mag_coords_iter_offset3(&cr,&cx,&cy,i,&ri,&xi,&yi); \
      br[ri] = mag_fn_##name##_##suffix(bx[xi],by[yi]); \
    } \
    return MAG_OK; \
  }

#define mag_gen_float_bin_scalar(name) \
  mag_gen_bin_scalar(float,float32,name,f32) \
  mag_gen_bin_scalar(mag_float16_t,float16,name,f16) \
  mag_gen_bin_scalar(mag_bfloat16_t,bfloat16,name,bf16) \
  mag_gen_bin_scalar(mag_float8_e4m3fn_t,float8_e4m3fn,name,f8_e4m3fn)

#define mag_gen_float_bin_simd(name) \
  mag_gen_bin_simd(float,float32,f32,mag_vf32_loadu_f32,mag_vf32_storeu_f32,name) \
  mag_gen_bin_simd(mag_float16_t,float16,f16,mag_vf32_loadu_f16,mag_vf32_storeu_f16,name) \
  mag_gen_bin_simd(mag_bfloat16_t,bfloat16,bf16,mag_vf32_loadu_bf16,mag_vf32_storeu_bf16,name) \
  mag_gen_bin_simd(mag_float8_e4m3fn_t,float8_e4m3fn,f8_e4m3fn,mag_vf32_loadu_float8_e4m3fn,mag_vf32_storeu_float8_e4m3fn,name)

mag_gen_float_bin_simd(add)
mag_gen_float_bin_simd(sub)
mag_gen_float_bin_simd(mul)
mag_gen_float_bin_simd(div)
mag_gen_float_bin_scalar(floordiv)
mag_gen_float_bin_scalar(mod)
mag_gen_float_bin_scalar(pow)
mag_gen_float_bin_scalar(min)
mag_gen_float_bin_scalar(max)

#define mag_gen_int_bin_common(name) \
  mag_gen_bin_scalar(uint8_t,uint8,name,int) \
  mag_gen_bin_scalar(int8_t,int8,name,int) \
  mag_gen_bin_scalar(uint16_t,uint16,name,int) \
  mag_gen_bin_scalar(int16_t,int16,name,int) \
  mag_gen_bin_scalar(uint32_t,uint32,name,int) \
  mag_gen_bin_scalar(int32_t,int32,name,int) \
  mag_gen_bin_scalar(uint64_t,uint64,name,int) \
  mag_gen_bin_scalar(int64_t,int64,name,int)

mag_gen_int_bin_common(add)
mag_gen_int_bin_common(sub)
mag_gen_int_bin_common(mul)
mag_gen_int_bin_common(div)
mag_gen_int_bin_common(and)
mag_gen_int_bin_common(or)
mag_gen_int_bin_common(xor)
mag_gen_int_bin_common(min)
mag_gen_int_bin_common(max)

#define mag_gen_int_signed_unsigned(name) \
  mag_gen_bin_scalar(uint8_t,uint8,name,u) \
  mag_gen_bin_scalar(int8_t,int8,name,i) \
  mag_gen_bin_scalar(uint16_t,uint16,name,u) \
  mag_gen_bin_scalar(int16_t,int16,name,i) \
  mag_gen_bin_scalar(uint32_t,uint32,name,u) \
  mag_gen_bin_scalar(int32_t,int32,name,i) \
  mag_gen_bin_scalar(uint64_t,uint64,name,u) \
  mag_gen_bin_scalar(int64_t,int64,name,i)

mag_gen_int_signed_unsigned(floordiv)
mag_gen_int_signed_unsigned(mod)
mag_gen_int_signed_unsigned(pow)

#define mag_gen_shift(T, TF, name, sign) \
  static mag_status_t MAG_HOTPROC mag_##name##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *y = payload->cmd->in[1]; \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *by = (const T *)mag_tensor_data_ptr(y); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra+chunk,total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[3]){r,x,y},3)) { \
      for (int64_t i=ra; i < rb; ++i) br[i] = mag_fn_##name##_##sign(bx[i],by[i],T); \
      return MAG_OK; \
    } \
    mag_coords_iter_t cr,cx,cy; \
    mag_coords_iter_init(&cr,&r->meta.coords); \
    mag_coords_iter_init(&cx,&x->meta.coords); \
    mag_coords_iter_init(&cy,&y->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri,xi,yi; \
      mag_coords_iter_offset3(&cr,&cx,&cy,i,&ri,&xi,&yi); \
      br[ri] = mag_fn_##name##_##sign(bx[xi],by[yi],T); \
    } \
    return MAG_OK; \
  }

#define mag_gen_shift_all(name) \
  mag_gen_shift(uint8_t,uint8,name,u) \
  mag_gen_shift(int8_t,int8,name,i) \
  mag_gen_shift(uint16_t,uint16,name,u) \
  mag_gen_shift(int16_t,int16,name,i) \
  mag_gen_shift(uint32_t,uint32,name,u) \
  mag_gen_shift(int32_t,int32,name,i) \
  mag_gen_shift(uint64_t,uint64,name,u) \
  mag_gen_shift(int64_t,int64,name,i)

mag_gen_shift_all(shl)
mag_gen_shift_all(shr)

#define mag_gen_cmp(T, TF, name, OP, CVT) \
  static mag_status_t MAG_HOTPROC mag_##name##_##TF(mag_error_t *err, const mag_kernel_payload_t *payload) { \
    (void)err; \
    mag_tensor_t *r = payload->cmd->out[0]; \
    const mag_tensor_t *x = payload->cmd->in[0]; \
    const mag_tensor_t *y = payload->cmd->in[1]; \
    uint8_t *br = (uint8_t *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    const T *by = (const T *)mag_tensor_data_ptr(y); \
    int64_t total = r->meta.numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra+chunk,total); \
    if (mag_unlikely(rb <= ra)) return MAG_OK; \
    if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[3]){r,x,y},3)) { \
      for (int64_t i=ra; i < rb; ++i) br[i] = CVT(bx[i]) OP CVT(by[i]); \
      return MAG_OK; \
    } \
    mag_coords_iter_t cr,cx,cy; \
    mag_coords_iter_init(&cr,&r->meta.coords); \
    mag_coords_iter_init(&cx,&x->meta.coords); \
    mag_coords_iter_init(&cy,&y->meta.coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri,xi,yi; \
      mag_coords_iter_offset3(&cr,&cx,&cy,i,&ri,&xi,&yi); \
      br[ri] = CVT(bx[xi]) OP CVT(by[yi]); \
    } \
    return MAG_OK; \
  }

#define mag_cvt_nop(x) (x)

#define mag_gen_cmp_all(name, OP) \
  mag_gen_cmp(float,float32,name,OP,mag_cvt_nop) \
  mag_gen_cmp(mag_float16_t,float16,name,OP,mag_float16_to_float32) \
  mag_gen_cmp(mag_bfloat16_t,bfloat16,name,OP,mag_bfloat16_to_float32) \
  mag_gen_cmp(mag_float8_e4m3fn_t,float8_e4m3fn,name,OP,mag_float8_e4m3fn_to_float32) \
  mag_gen_cmp(uint8_t,uint8,name,OP,mag_cvt_nop) \
  mag_gen_cmp(int8_t,int8,name,OP,mag_cvt_nop) \
  mag_gen_cmp(uint16_t,uint16,name,OP,mag_cvt_nop) \
  mag_gen_cmp(int16_t,int16,name,OP,mag_cvt_nop) \
  mag_gen_cmp(uint32_t,uint32,name,OP,mag_cvt_nop) \
  mag_gen_cmp(int32_t,int32,name,OP,mag_cvt_nop) \
  mag_gen_cmp(uint64_t,uint64,name,OP,mag_cvt_nop) \
  mag_gen_cmp(int64_t,int64,name,OP,mag_cvt_nop)

mag_gen_cmp_all(eq, ==)
mag_gen_cmp_all(ne, !=)
mag_gen_cmp_all(lt, <)
mag_gen_cmp_all(gt, >)
mag_gen_cmp_all(le, <=)
mag_gen_cmp_all(ge, >=)

#undef mag_gen_cmp_all
#undef mag_gen_cmp
#undef mag_cvt_nop
#undef mag_gen_shift_all
#undef mag_gen_shift
#undef mag_gen_int_signed_unsigned
#undef mag_gen_int_bin_common
#undef mag_gen_float_bin_simd
#undef mag_gen_float_bin_scalar
#undef mag_gen_bin_simd
#undef mag_gen_bin_scalar

#undef mag_fn_add_int
#undef mag_fn_sub_int
#undef mag_fn_mul_int
#undef mag_fn_div_int
#undef mag_fn_and_int
#undef mag_fn_or_int
#undef mag_fn_xor_int
#undef mag_fn_floordiv_i
#undef mag_fn_floordiv_u
#undef mag_fn_mod_i
#undef mag_fn_mod_u
#undef mag_fn_pow_i
#undef mag_fn_pow_u
#undef mag_fn_shl_i
#undef mag_fn_shl_u
#undef mag_fn_shr_i
#undef mag_fn_shr_u
