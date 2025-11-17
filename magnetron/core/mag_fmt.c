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

/*
** Dragonbox floating-point to decimal conversion.
** Original implementation by Junekey Jeon.
** Paper: https://github.com/jk-jeon/dragonbox/blob/master/other_files/Dragonbox.pdf
**
** Licensed under the Apache License Version 2.0 with LLVM Exceptions:
**   https://llvm.org/foundation/relicensing/LICENSE.txt
** or the Apache 2.0 license included with this project.
**
** C 99 Port and modifications for Magnetron by Mario Sieg.
*/

#include "mag_fmt.h"
#include "mag_u128.h"

static uint32_t mag_rol32(uint32_t n, unsigned r) { r &= 31; return (n>>r) | (n<<((32-r) & 31)); }

static uint64_t rotr64(uint64_t n, unsigned r) { r &= 63; return (n>>r) | (n<<((64-r) & 63)); }

static int32_t mag_floorlog2(uint64_t n) {
    int32_t c;
    for (c=-1; n!=0; ++c, n >>= 1);
    return c;
}

static int32_t mag_floor_log10_pow2(int32_t e) { mag_assert2(-2620 <= e && e <= 2620); return (e*315653)>>20; }

static int32_t mag_floor_log2_pow10(int32_t e) { mag_assert2(-1233 <= e && e <= 1233); return (e*1741647)>>19; }

static int32_t mag_floor_log10_pow2_minus_log10_4_over_3(int32_t e) { mag_assert2(-2985 <= e && e <= 2936); return (e*631305 - 261663)>>21; }

static int32_t mag_floor_log5_pow2(int32_t e) { mag_assert2(-1831 <= e && e <= 1831); return (e*225799)>>19; }

static int32_t mag_floor_log5_pow2_minus_log5_3(int32_t e) { mag_assert2(-3543 <= e && e <= 2427); return (e*451597 - 715764)>>20; }

static int32_t mag_n_fact(int32_t a, uint32_t n) {
    int32_t c=0;
    for (c=0; n % a == 0; n /= a, ++c);
    return c;
}

static const uint32_t mag_div_magic[2] = {6554, 656};

static const uint64_t mag_fmt_cache[78] = {
    UINT64_C(0x81ceb32c4b43fcf5), UINT64_C(0xa2425ff75e14fc32),
    UINT64_C(0xcad2f7f5359a3b3f), UINT64_C(0xfd87b5f28300ca0e),
    UINT64_C(0x9e74d1b791e07e49), UINT64_C(0xc612062576589ddb),
    UINT64_C(0xf79687aed3eec552), UINT64_C(0x9abe14cd44753b53),
    UINT64_C(0xc16d9a0095928a28), UINT64_C(0xf1c90080baf72cb2),
    UINT64_C(0x971da05074da7bef), UINT64_C(0xbce5086492111aeb),
    UINT64_C(0xec1e4a7db69561a6), UINT64_C(0x9392ee8e921d5d08),
    UINT64_C(0xb877aa3236a4b44a), UINT64_C(0xe69594bec44de15c),
    UINT64_C(0x901d7cf73ab0acda), UINT64_C(0xb424dc35095cd810),
    UINT64_C(0xe12e13424bb40e14), UINT64_C(0x8cbccc096f5088cc),
    UINT64_C(0xafebff0bcb24aaff), UINT64_C(0xdbe6fecebdedd5bf),
    UINT64_C(0x89705f4136b4a598), UINT64_C(0xabcc77118461cefd),
    UINT64_C(0xd6bf94d5e57a42bd), UINT64_C(0x8637bd05af6c69b6),
    UINT64_C(0xa7c5ac471b478424), UINT64_C(0xd1b71758e219652c),
    UINT64_C(0x83126e978d4fdf3c), UINT64_C(0xa3d70a3d70a3d70b),
    UINT64_C(0xcccccccccccccccd), UINT64_C(0x8000000000000000),
    UINT64_C(0xa000000000000000), UINT64_C(0xc800000000000000),
    UINT64_C(0xfa00000000000000), UINT64_C(0x9c40000000000000),
    UINT64_C(0xc350000000000000), UINT64_C(0xf424000000000000),
    UINT64_C(0x9896800000000000), UINT64_C(0xbebc200000000000),
    UINT64_C(0xee6b280000000000), UINT64_C(0x9502f90000000000),
    UINT64_C(0xba43b74000000000), UINT64_C(0xe8d4a51000000000),
    UINT64_C(0x9184e72a00000000), UINT64_C(0xb5e620f480000000),
    UINT64_C(0xe35fa931a0000000), UINT64_C(0x8e1bc9bf04000000),
    UINT64_C(0xb1a2bc2ec5000000), UINT64_C(0xde0b6b3a76400000),
    UINT64_C(0x8ac7230489e80000), UINT64_C(0xad78ebc5ac620000),
    UINT64_C(0xd8d726b7177a8000), UINT64_C(0x878678326eac9000),
    UINT64_C(0xa968163f0a57b400), UINT64_C(0xd3c21bcecceda100),
    UINT64_C(0x84595161401484a0), UINT64_C(0xa56fa5b99019a5c8),
    UINT64_C(0xcecb8f27f4200f3a), UINT64_C(0x813f3978f8940985),
    UINT64_C(0xa18f07d736b90be6), UINT64_C(0xc9f2c9cd04674edf),
    UINT64_C(0xfc6f7c4045812297), UINT64_C(0x9dc5ada82b70b59e),
    UINT64_C(0xc5371912364ce306), UINT64_C(0xf684df56c3e01bc7),
    UINT64_C(0x9a130b963a6c115d), UINT64_C(0xc097ce7bc90715b4),
    UINT64_C(0xf0bdc21abb48db21), UINT64_C(0x96769950b50d88f5),
    UINT64_C(0xbc143fa4e250eb32), UINT64_C(0xeb194f8e1ae525fe),
    UINT64_C(0x92efd1b8d0cf37bf), UINT64_C(0xb7abc627050305ae),
    UINT64_C(0xe596b7b0c643c71a), UINT64_C(0x8f7e32ce7bea5c70),
    UINT64_C(0xb35dbf821ae4f38c), UINT64_C(0xe0352f62a19e306f)
};

static void mag_trim_trailing_zeros(uint32_t *sig, int32_t *exp) {
    uint32_t r = mag_rol32(*sig * UINT32_C(184254097), 4);
    uint32_t b = r < UINT32_C(429497);
    size_t s = b;
    *sig = b ? r : *sig;
    r = mag_rol32(*sig * UINT32_C(42949673), 2);
    b = r < UINT32_C(42949673);
    s = s * 2 + b;
    *sig = b ? r : *sig;
    r = mag_rol32(*sig * UINT32_C(1288490189), 1);
    b = r < UINT32_C(429496730);
    s = s * 2 + b;
    *sig = b ? r : *sig;
    *exp += (int32_t)(s);
}

static uint32_t mag_ipow(int32_t k, uint32_t a) {
    int32_t e = k;
    uint32_t p=1;
    while (e) {
        if (e & 1) p *= a;
        e >>= 1;
        a *= a;
    }
    return p;
}

typedef struct mag_mul_t {
    uint32_t integer_part;
    bool is_integer;
} mag_mul_t;

typedef struct mag_parity_t {
    bool parity;
    bool is_integer;
} mag_parity_t;

static mag_mul_t mag_compute_mul(uint32_t u, uint64_t cache) {
    uint64_t r = mag_uint128_mulhi96(u, cache);
    return (mag_mul_t){(uint32_t)(r>>32), (uint32_t)r == 0};
}

static uint32_t mag_compute_delta(uint64_t cache, int32_t beta) {
    return (uint32_t)(cache >> (64 - 1 - beta));
}

static mag_parity_t mag_compute_mul_parity(uint32_t two_f, uint64_t cache, int32_t beta) {
    mag_assert2(beta >= 1);
    mag_assert2(beta <= 32);
    uint64_t r = mag_uint128_mullo96(two_f, cache);
    return (mag_parity_t){((r >> (64 - beta)) & 1) != 0, (UINT32_C(0xffffffff) & (r >> (32 - beta))) == 0};
}

static uint32_t mag_left_lo(uint64_t cache, int32_t beta) {
    return (uint32_t)((cache - (cache >> (23 + 2))) >> (64 - 23 - 1 - beta));
}

static uint32_t mag_right_hi(uint64_t cache, int32_t beta) {
    return (uint32_t)((cache + (cache >> (23 + 1))) >> (64 - 23 - 1 - beta));
}

static uint32_t mag_rndup(uint64_t cache, int32_t beta) {
    return ((uint32_t)(cache >> (64 - 23 - 2 - beta)) + 1) / 2;
}

static uint32_t mag_div_pow10(int32_t x, uint32_t n_max, uint32_t n) {
    if (x == 1 && n_max <= UINT32_C(1073741828)) return (uint32_t)(mag_uint128_mul64(n, UINT32_C(429496730))>>32);
    if (x == 2) return (uint32_t)(mag_uint128_mul64(n, UINT32_C(1374389535))>>37);
    return n / mag_ipow(x, 10);
}

static bool check_divisibility_and_mag_div_pow10(int32_t N, uint32_t *n) {
    mag_assert2(*n <= mag_ipow(N+1, 10));
    uint32_t m = mag_div_magic[N - 1];
    uint32_t prod = *n*m;
    bool r = (prod & ((1u<<16)-1)) < m;
    *n = prod>>16;
    return r;
}

typedef struct mag_binfp_t {
    uint32_t sig;
    int32_t exp;
    bool is_neg;
} mag_binfp_t;

static mag_binfp_t mag_binfp_decompose(mag_e8m23_t x) {
    union castor {
        uint32_t u;
        mag_e8m23_t f;
    } c = {.f=x};
    return (mag_binfp_t){(c.u&((1u<<23)-1)), (int32_t)(c.u>>23&((1u<<8)-1)), (bool)(c.u>>(23+8))};
}

#define mag_is_fin(e) ((e) != (1u<<8)-1)

typedef struct mag_decfp_t {
    uint32_t sig;
    int32_t exp;
    bool is_neg;
} mag_decfp_t;

static mag_decfp_t mag_decfp_to_dec_dragonbox(uint32_t binmt, int32_t bexp, bool is_neg) {
    int32_t kappa = mag_floor_log10_pow2(32 - 23 - 2) - 1;
    bool is_even = (binmt & 1) == 0;
    uint32_t two_fc = binmt<<1;
    if (bexp) {
        bexp += -127 - 23;
        if (!two_fc) {
            int32_t minus_k = mag_floor_log10_pow2_minus_log10_4_over_3(bexp);
            int32_t beta = bexp + mag_floor_log2_pow10(-minus_k);
            uint64_t cache = mag_fmt_cache[-minus_k - -31];
            uint32_t xi = mag_left_lo(cache, beta);
            uint32_t zi = mag_right_hi(cache, beta);
            int32_t csi = 2 + mag_floorlog2(mag_ipow(mag_n_fact(5, (1u<<(23+2))-1)+1, 10/3));
            if (!(bexp >= 2 && bexp <= csi)) ++xi;
            uint32_t dsi = mag_div_pow10(1, (((2u<<23)+1)/3+1)*20, zi);
            if (dsi * 10 >= xi) {
                int32_t dexp = minus_k + 1;
                mag_trim_trailing_zeros(&dsi, &dexp);
                return (mag_decfp_t){dsi, dexp, is_neg};
            }
            dsi = mag_rndup(cache, beta);
            int32_t lo = -mag_floor_log5_pow2_minus_log5_3(23+4)-2-23;
            int32_t hi = -mag_floor_log5_pow2(23+2)-2-23;
            if ((dsi & 1) != 0 &&
                bexp >= lo &&
                bexp <= hi) {
                --dsi;
            }
            else if (dsi < xi) {
                ++dsi;
            }
            return (mag_decfp_t){dsi, minus_k, is_neg};
        }
        two_fc |= 1u<<(23+1);
    }
    else bexp = -126-23;
    int32_t minus_k = mag_floor_log10_pow2(bexp) - kappa;
    uint64_t cache = mag_fmt_cache[-minus_k - -31];
    int32_t beta = bexp + mag_floor_log2_pow10(-minus_k);
    uint32_t deltai = mag_compute_delta(cache, beta);
    mag_mul_t zr = mag_compute_mul((two_fc | 1) << beta, cache);
    uint32_t bd = mag_ipow(kappa + 1, 10);
    uint32_t sd = mag_ipow(kappa, 10);
    uint32_t dsi = mag_div_pow10(kappa + 1, (2u<<23)*bd - 1, zr.integer_part);
    uint32_t r = zr.integer_part - bd * dsi;
    do {
        if (r < deltai) {
            if ((r|(uint32_t)!zr.is_integer|(uint32_t)(is_even)) == 0) {
                --dsi;
                r = bd;
                break;
            }
        }
        else if (r > deltai) break;
        else {
            mag_parity_t x_result = mag_compute_mul_parity(two_fc - 1, cache, beta);
            if (!(x_result.parity|(x_result.is_integer & is_even))) break;
        }
        int32_t dexp = minus_k + kappa + 1;
        mag_trim_trailing_zeros(&dsi, &dexp);
        return (mag_decfp_t){dsi, dexp, is_neg};
    } while (0);
    dsi *= 10;
    uint32_t dist = r - (deltai>>1) + (sd>>1);
    bool approx_y_parity = ((dist ^ (sd>>1))&1) != 0;
    bool small_div = check_divisibility_and_mag_div_pow10(kappa, &dist);
    dsi += dist;
    if (small_div) {
        mag_parity_t y_result = mag_compute_mul_parity(two_fc, cache, beta);
        if (y_result.parity != approx_y_parity) --dsi;
        else if ((dsi&1) & y_result.is_integer) --dsi;
    }
    return (mag_decfp_t){dsi, minus_k + kappa, is_neg};
}

static char *mag_fmt_dec(char *o, const char *d, int32_t nd, int32_t dec_exp) {
    int32_t point = nd + dec_exp;
    if (point >= nd) {
        for (int32_t i=0; i < nd; ++i) *o++ = d[i];
        for (int32_t i=nd; i < point; ++i)*o++ = '0';
        return o;
    }
    if (point > 0) {
        char *start = o;
        for (int32_t i=0; i < point; ++i) *o++ = d[i];
        *o++ = '.';
        for (int32_t i=point; i < nd; ++i) *o++ = d[i];
        char *end = o;
        while (end > start && end[-1] == '0') --end;
        if (end > start && end[-1] == '.') --end;
        return end;
    }
    *o++ = '0';
    *o++ = '.';
    for (int32_t i=0; i < -point; ++i) *o++ = '0';
    for (int32_t i=0; i < nd; ++i) *o++ = d[i];
    return o;
}

static char *mag_fmt_scientific(char *o, const char *d, int32_t nd, int32_t dec_exp) {
    int32_t sci_exp = dec_exp + (nd - 1);
    char *mant_start = o;
    *o++ = d[0];
    if (nd > 1) {
        *o++ = '.';
        for (int32_t i = 1; i < nd; ++i) {
            *o++ = d[i];
        }
        char *end = o;
        while (end > mant_start && end[-1] == '0') --end;
        if (end > mant_start && end[-1] == '.') --end;
        o = end;
    }
    *o++ = 'e';
    if (sci_exp < 0) {
        *o++ = '-';
        sci_exp = -sci_exp;
    } else *o++ = '+';
    char tmp[8];
    int32_t tlen = 0;
    do {
        tmp[tlen++] = (char)('0' + (sci_exp % 10));
        sci_exp /= 10;
    } while (sci_exp);
    for (int32_t i=tlen - 1; i >= 0; --i) *o++ = tmp[i];
    return o;
}

static char *mag_fmt_to_chars(mag_e8m23_t x, char *o, int32_t prec) {
    if (prec <= 0) prec = 1;
    mag_binfp_t const decomposed = mag_binfp_decompose(x);
    if (!mag_is_fin(decomposed.exp)) {
        if (!decomposed.sig) {
            if (decomposed.is_neg) {
                *o++ = '-';
            }
            memcpy(o, "Inf", 3);
            return o+3;
        }
        memcpy(o, "NaN", 3);
        return o+3;
    }
    if (decomposed.is_neg) *o++ = '-';
    if (decomposed.sig == 0 && decomposed.exp == 0) {
        *o++ = '0';
        return o;
    }
    mag_decfp_t fp = mag_decfp_to_dec_dragonbox(decomposed.sig, decomposed.exp, decomposed.is_neg);
    uint32_t dec_sig = fp.sig;
    char digits[16];
    int32_t nd = 0;
    do {
        digits[nd++] = (char)('0' + (dec_sig % 10));
        dec_sig /= 10;
    } while (dec_sig != 0);
    for (int32_t i = 0; i < nd>>1; ++i) {
        char tmp = digits[i];
        digits[i] = digits[nd-1-i];
        digits[nd-1-i] = tmp;
    }
    int32_t dec_exp = fp.exp;
    int32_t k = dec_exp+(nd-1);
    if (k >= -4 && k < prec) return mag_fmt_dec(o, digits, nd, dec_exp);
    return mag_fmt_scientific(o, digits, nd, dec_exp);
}

char *mag_fmt_e8m23(char (*p)[MAG_E8M23_FMT_BUF_SIZE], mag_e8m23_t n){
    char *e = mag_fmt_to_chars(n, *p, 6);
    *e = '\0';
    mag_assert2(e - *p < MAG_E8M23_FMT_BUF_SIZE);
    return e;
}
