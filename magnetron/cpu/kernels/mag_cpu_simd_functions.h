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

static MAG_AINLINE mag_vf32_t mag_vf32_exp(mag_vf32_t x) {
  mag_vf32_t r = mag_vf32_splat(0x1.8p23f);
  mag_vf32_t z = mag_vf32_fmadd(x, mag_vf32_splat(0x1.715476p+0f), r);
  mag_vf32_t n = mag_vf32_sub(z, r);
  mag_vf32_t b = mag_vf32_fnmadd(n, mag_vf32_splat(0x1.7f7d1cp-20f), mag_vf32_fnmadd(n, mag_vf32_splat(0x1.62e4p-1f), x));
  mag_vi32_t e = mag_vi32_slli(mag_vi32_reinterpret_from_vf32(z), 23);
  mag_vf32_t k = mag_vf32_reinterpret_from_vi32(mag_vi32_add(e, mag_vi32_reinterpret_from_vf32(mag_vf32_splat(1.0f))));
  mag_vmask32_t c = mag_vf32_cmpgt(mag_vf32_abs(n), mag_vf32_splat(126.0f));
  mag_vf32_t u = mag_vf32_mul(b, b);
  mag_vf32_t j = mag_vf32_fmadd(
    mag_vf32_mul(mag_vf32_splat(0x1.ffffecp-1f), b),
    mag_vf32_fmadd(mag_vf32_fmadd(mag_vf32_splat(0x1.fffdb6p-2f), mag_vf32_splat(0x1.555e66p-3f), b), mag_vf32_fmadd(mag_vf32_splat(0x1.573e2ep-5f), mag_vf32_splat(0x1.0e4020p-7f), b),u
    ), u
  );
  mag_vf32_t fast = mag_vf32_fmadd(k, j, k);
  if (!mag_vmask32_any(c)) return fast;
  mag_vi32_t d = mag_vi32_and(mag_vi32_from_vmask32(mag_vf32_cmple(n, mag_vf32_zero())), mag_vi32_splat((int32_t)0x82000000u));
  mag_vf32_t s1 = mag_vf32_reinterpret_from_vi32(mag_vi32_add(d, mag_vi32_splat((int32_t)0x7f000000u)));
  mag_vf32_t s2 = mag_vf32_reinterpret_from_vi32(mag_vi32_sub(e, d));
  mag_vmask32_t extreme = mag_vf32_cmpgt(mag_vf32_abs(n), mag_vf32_splat(192.0f));
  mag_vf32_t medium = mag_vf32_mul(mag_vf32_fmadd(s2, s2, j), s1);
  mag_vf32_t huge = mag_vf32_mul(s1, s1);
  return mag_vf32_blend(extreme, huge, mag_vf32_blend(c, medium, fast));
}

static MAG_AINLINE mag_vf32_t mag_vf32_tanh(mag_vf32_t x) {
  mag_vf32_t k1 = mag_vf32_splat(1.0f);
  mag_vf32_t k2 = mag_vf32_splat(2.0f);
  mag_vf32_t a = mag_vf32_mul(mag_vf32_splat(-2.f), x);
  mag_vf32_t b = mag_vf32_exp(a);
  mag_vf32_t c = mag_vf32_add(k1, b);
  mag_vf32_t inv = mag_vf32_rcp_approx(c);
  inv = mag_vf32_rcp_refine_step(c, inv);
  inv = mag_vf32_rcp_refine_step(c, inv);
  return mag_vf32_sub(mag_vf32_mul(k2, inv), k1);
}

static MAG_AINLINE void mag_vf32_sincos(mag_vf32_t x, mag_vf32_t *s, mag_vf32_t *c) {
  mag_vi32_t v2 = mag_vi32_splat(2), v4 = mag_vi32_splat(4);
  mag_vf32_t sign_bit_sin = x;
  x = mag_vf32_and(x, mag_vf32_reinterpret_from_vi32(mag_vi32_splat((int32_t)~0x80000000u)));
  sign_bit_sin = mag_vf32_and(sign_bit_sin, mag_vf32_reinterpret_from_vi32(mag_vi32_splat((int32_t)0x80000000u)));
  mag_vf32_t y = mag_vf32_mul(x, mag_vf32_splat(1.27323954473516f));
  mag_vi32_t imm2 = mag_vf32_trunc_to_vi32(y);
  imm2 = mag_vi32_add(imm2, mag_vi32_splat(1));
  imm2 = mag_vi32_and(imm2, mag_vi32_splat(~1));
  y = mag_vi32_to_f32(imm2);
  mag_vi32_t imm4 = imm2;
  mag_vi32_t imm0 = mag_vi32_slli(mag_vi32_and(imm2, v4), 29);
  imm2 = mag_vi32_and(imm2, v2);
  mag_vmask32_t poly_mask = mag_vi32_cmpeq(imm2, mag_vi32_zero());
  mag_vf32_t swap_sign_bit_sin = mag_vf32_reinterpret_from_vi32(imm0);
  x = mag_vf32_fmadd(y, mag_vf32_splat(-0.78515625f), x);
  x = mag_vf32_fmadd(y, mag_vf32_splat(-2.4187564849853515625e-4f), x);
  x = mag_vf32_fmadd(y, mag_vf32_splat(-3.77489497744594108e-8f), x);
  imm4 = mag_vi32_slli(mag_vi32_andnot(mag_vi32_sub(imm4, v2), v4), 29);
  mag_vf32_t sign_bit_cos = mag_vf32_reinterpret_from_vi32(imm4);
  sign_bit_sin = mag_vf32_xor(sign_bit_sin, swap_sign_bit_sin);
  mag_vf32_t z = mag_vf32_mul(x, x);
  y = mag_vf32_splat(2.443315711809948e-005f);
  y = mag_vf32_fmadd(y, z, mag_vf32_splat(-1.388731625493765e-003f));
  y = mag_vf32_fmadd(y, z, mag_vf32_splat(4.166664568298827e-002f));
  mag_vf32_t t = mag_vf32_mul(y, mag_vf32_mul(z, z));
  y = mag_vf32_add(mag_vf32_fnmadd(mag_vf32_splat(0.5f), z, t), mag_vf32_splat(1.0f));
  mag_vf32_t y2 = mag_vf32_splat(-1.9515295891e-4f);
  y2 = mag_vf32_fmadd(y2, z, mag_vf32_splat(8.3321608736e-3f));
  y2 = mag_vf32_fmadd(y2, z, mag_vf32_splat(-1.6666654611e-1f));
  y2 = mag_vf32_mul(y2, z);
  y2 = mag_vf32_fmadd(y2, x, x);
  mag_vf32_t ss = mag_vf32_blend(poly_mask, y2, y);
  mag_vf32_t cc = mag_vf32_blend(poly_mask, y, y2);
  *s = mag_vf32_xor(ss, sign_bit_sin);
  *c = mag_vf32_xor(cc, sign_bit_cos);
}

static MAG_AINLINE mag_vf32_t mag_vf32_log(mag_vf32_t x) {
  mag_vmask32_t invalid = mag_vf32_cmple(x, mag_vf32_zero());
  x = mag_vf32_max(x, mag_vf32_reinterpret_from_vi32(mag_vi32_splat((int32_t)0x00800000u)));
  mag_vi32_t ix = mag_vi32_reinterpret_from_vf32(x);
  mag_vi32_t e = mag_vi32_sub(mag_vi32_srli(ix, 23), mag_vi32_splat(0x7f));
  x = mag_vf32_reinterpret_from_vi32(mag_vi32_or(mag_vi32_and(ix, mag_vi32_splat((int32_t)~0x7f800000u)),mag_vi32_splat((int32_t)0x3f000000u)));
  mag_vf32_t ef = mag_vf32_add(mag_vi32_to_f32(e), mag_vf32_splat(1.0f));
  mag_vmask32_t m = mag_vf32_cmplt(x, mag_vf32_splat(0.707106781186547524f));
  mag_vf32_t mf = mag_vf32_reinterpret_from_vi32(mag_vi32_from_vmask32(m));
  mag_vf32_t tmp = mag_vf32_and(x, mf);
  x = mag_vf32_sub(x, mag_vf32_splat(1.0f));
  ef = mag_vf32_sub(ef, mag_vf32_and(mag_vf32_splat(1.0f), mf));
  x = mag_vf32_add(x, tmp);
  mag_vf32_t z = mag_vf32_mul(x, x);
  mag_vf32_t y = mag_vf32_splat(7.0376836292e-2f);
  y = mag_vf32_fmadd(y, x, mag_vf32_splat(-1.1514610310e-1f));
  y = mag_vf32_fmadd(y, x, mag_vf32_splat( 1.1676998740e-1f));
  y = mag_vf32_fmadd(y, x, mag_vf32_splat(-1.2420140846e-1f));
  y = mag_vf32_fmadd(y, x, mag_vf32_splat( 1.4249322787e-1f));
  y = mag_vf32_fmadd(y, x, mag_vf32_splat(-1.6668057665e-1f));
  y = mag_vf32_fmadd(y, x, mag_vf32_splat( 2.0000714765e-1f));
  y = mag_vf32_fmadd(y, x, mag_vf32_splat(-2.4999993993e-1f));
  y = mag_vf32_fmadd(y, x, mag_vf32_splat( 3.3333331174e-1f));
  y = mag_vf32_mul(mag_vf32_mul(y, x), z);
  y = mag_vf32_fmadd(ef, mag_vf32_splat(-2.12194440e-4f), y);
  y = mag_vf32_sub(y, mag_vf32_mul(z, mag_vf32_splat(0.5f)));
  x = mag_vf32_add(x, y);
  x = mag_vf32_fmadd(ef, mag_vf32_splat(0.693359375f), x);
  return mag_vf32_or(x, mag_vf32_reinterpret_from_vi32(mag_vi32_from_vmask32(invalid)));

}

static MAG_AINLINE mag_vf32_t mag_vf32_sigmoid(mag_vf32_t x) {
  mag_vf32_t one = mag_vf32_splat(1.0f);
  mag_vf32_t ex = mag_vf32_exp(mag_vf32_sub(mag_vf32_zero(), x));
  mag_vf32_t d = mag_vf32_add(one, ex);
  mag_vf32_t r = mag_vf32_rcp_approx(d);
  r = mag_vf32_rcp_refine_step(d, r);
  r = mag_vf32_rcp_refine_step(d, r);
  return r;
}

static MAG_AINLINE mag_vf32_t mag_vf32_silu(mag_vf32_t x) {
  return mag_vf32_mul(x, mag_vf32_sigmoid(x));
}

static MAG_AINLINE mag_vf32_t mag_vf32_gelu_approx(mag_vf32_t x) {
  mag_vf32_t half = mag_vf32_splat(0.5f);
  mag_vf32_t one = mag_vf32_splat(1.0f);
  mag_vf32_t k = mag_vf32_splat(MAG_INVSQRT2);
  mag_vf32_t c = mag_vf32_splat(MAG_GELU_COEFF);
  mag_vf32_t x2 = mag_vf32_mul(x, x);
  mag_vf32_t x3 = mag_vf32_mul(x2, x);
  mag_vf32_t inner = mag_vf32_mul(k, mag_vf32_fmadd(c, x3, x));
  mag_vf32_t th = mag_vf32_tanh(inner);
  return mag_vf32_mul(mag_vf32_mul(half, x), mag_vf32_add(one, th));
}

static MAG_AINLINE mag_vf32_t mag_vf32_relu(mag_vf32_t x) {
  return mag_vf32_max(x, mag_vf32_zero());
}

static MAG_AINLINE mag_vf32_t mag_vf32_step(mag_vf32_t x) {
  return mag_vf32_blend(mag_vf32_cmpgt(x, mag_vf32_zero()), mag_vf32_splat(1.0f), mag_vf32_zero());
}

static MAG_AINLINE mag_vf32_t mag_vf32_hard_sigmoid(mag_vf32_t x) {
  mag_vf32_t y = mag_vf32_mul(mag_vf32_add(x, mag_vf32_splat(3.0f)), mag_vf32_splat(1.0f / 6.0f));
  return mag_vf32_min(mag_vf32_splat(1.0f), mag_vf32_max(mag_vf32_zero(), y));
}

static MAG_AINLINE mag_vf32_t mag_vf32_tan(mag_vf32_t x) {
  mag_vf32_t s, c;
  mag_vf32_sincos(x, &s, &c);
  return mag_vf32_div(s, c);
}

static MAG_AINLINE mag_vf32_t mag_vf32_sin(mag_vf32_t x) {
  mag_vf32_t s, c;
  mag_vf32_sincos(x, &s, &c);
  return s;
}
static MAG_AINLINE mag_vf32_t mag_vf32_cos(mag_vf32_t x) {
  mag_vf32_t s, c;
  mag_vf32_sincos(x, &s, &c);
  return c;
}
