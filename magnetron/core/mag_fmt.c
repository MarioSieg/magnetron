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
** This file contains code derived from LuaJIT's floating-point number formatting implementation.
** LuaJIT Project Readme:
**
**      README for LuaJIT 2.1
**      ---------------------
**
**      LuaJIT is a Just-In-Time (JIT) compiler for the Lua programming language.
**
**      Project Homepage: https://luajit.org/
**
**      LuaJIT is Copyright (C) 2005-2025 Mike Pall.
**      LuaJIT is free software, released under the MIT license.
**      See full Copyright Notice in the COPYRIGHT file or in luajit.h.
**
**      Documentation for LuaJIT is available in HTML format.
**      Please point your favorite browser to:
**
**     doc/luajit.html
**
** Original Source File Notice:
**      String formatting for floating-point numbers.
**      Copyright (C) 2005-2025 Mike Pall. See Copyright Notice in luajit.h
**      Contributed by Peter Cawley.
**
** Modifications:
**   This file includes modifications by Mario Sieg for the
**   Magnetron project. All modifications are licensed under the
**   Apache License, Version 2.0.
*/

#include "mag_fmt.h"
#include "mag_sstream.h"
#include "mag_float16.h"
#include "mag_coords_iter.h"
#include "mag_tensor.h"
#include "mag_backend.h"
#include "mag_alloc.h"

#define wint_r(x, sh, sc) { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }
static char* mag_wuint9(char *p, uint32_t u) {
    uint32_t v = u / 10000, w;
    u -= v * 10000;
    w = v / 10000;
    v -= w * 10000;
    *p++ = (char)('0'+w);
    wint_r(v, 23, 1000)
    wint_r(v, 12, 100)
    wint_r(v, 10, 10)
    *p++ = (char)('0'+v);
    wint_r(u, 23, 1000)
    wint_r(u, 12, 100)
    wint_r(u, 10, 10)
    *p++ = (char)('0'+u);
    return p;
}
#undef wint_r

#define wint_r(x, sh, sc) { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }
static char* mag_wint(char *p, int32_t k) {
    uint32_t u = (uint32_t)k;
    if (k < 0) { u = ~u+1u; *p++ = '-'; }
    if (u < 10000) {
        if (u < 10) goto dig1;
        if (u < 100) goto dig2;
        if (u < 1000) goto dig3;
    } else {
        uint32_t v = u / 10000; u -= v * 10000;
        if (v < 10000) {
            if (v < 10) goto dig5;
            if (v < 100) goto dig6;
            if (v < 1000) goto dig7;
        } else {
            uint32_t w = v / 10000; v -= w * 10000;
            if (w >= 10) wint_r(w, 10, 10)
            *p++ = (char)('0'+w);
        }
        wint_r(v, 23, 1000)
        dig7: wint_r(v, 12, 100)
        dig6: wint_r(v, 10, 10)
        dig5: *p++ = (char)('0'+v);
    }
    wint_r(u, 23, 1000)
    dig3: wint_r(u, 12, 100)
    dig2: wint_r(u, 10, 10)
    dig1: *p++ = (char)('0'+u);
    return p;
}
#undef wint_r

static char *mag_uint64_to_str(char *p, uint64_t v) {
    char tmp[32];
    int n = 0;
    if (v == 0) {
        *p++ = '0';
        return p;
    }
    while (v) {
        uint64_t q = v / 10;
        tmp[n++] = (char)('0' + (uint32_t)(v - q*10));
        v = q;
    }
    while (n--) *p++ = tmp[n];
    return p;
}

static char *mag_int64_to_str(char *p, int64_t v) {
    if (v < 0) {
        *p++ = '-';
        // safe for INT64_MIN
        uint64_t uv = (uint64_t)(-(v + 1)) + 1u;
        return mag_uint64_to_str(p, uv);
    }
    return mag_uint64_to_str(p, (uint64_t)v);
}

char *mag_fmt_int64(char *p, int64_t n) {
    if (n >= INT32_MIN && n <= INT32_MAX)
        return mag_wint(p, (int32_t)n);
    return mag_int64_to_str(p, n);
}

char *mag_fmt_uint64(char *p, uint64_t n) {
    if (n <= UINT32_MAX)
        return mag_wint(p, (int32_t)n);
    return mag_uint64_to_str(p, n);
}

/* Rescale factors to push the exponent of a number towards zero. */
#define rescale_exponents(P, N) \
  P(308), P(289), P(270), P(250), P(231), P(212), P(193), P(173), P(154), \
  P(135), P(115), P(96), P(77), P(58), P(38), P(0), P(0), P(0), N(39), N(58), \
  N(77), N(96), N(116), N(135), N(154), N(174), N(193), N(212), N(231), \
  N(251), N(270), N(289)
#define one_e_p(X) 1e+0 ## X
#define one_e_n(X) 1e-0 ## X
static const int16_t mag_rescale_e[] = { rescale_exponents(-, +) };
static const double mag_rescale_n[] = { rescale_exponents(one_e_p, one_e_n) };
#undef one_e_n
#undef one_e_p

/*
** For p in range -70 through 57, this table encodes pairs (m, e) such that
** 4*2^p <= (uint8_t)m*10^e, and is the smallest value for which this holds.
*/
static const int8_t mag_four_ulp_m_e[] = {
    34, -21, 68, -21, 14, -20, 28, -20, 55, -20, 2, -19, 3, -19, 5, -19, 9, -19,
    -82, -18, 35, -18, 7, -17, -117, -17, 28, -17, 56, -17, 112, -16, -33, -16,
    45, -16, 89, -16, -78, -15, 36, -15, 72, -15, -113, -14, 29, -14, 57, -14,
    114, -13, -28, -13, 46, -13, 91, -12, -74, -12, 37, -12, 73, -12, 15, -11, 3,
    -11, 59, -11, 2, -10, 3, -10, 5, -10, 1, -9, -69, -9, 38, -9, 75, -9, 15, -7,
    3, -7, 6, -7, 12, -6, -17, -7, 48, -7, 96, -7, -65, -6, 39, -6, 77, -6, -103,
    -5, 31, -5, 62, -5, 123, -4, -11, -4, 49, -4, 98, -4, -60, -3, 4, -2, 79, -3,
    16, -2, 32, -2, 63, -2, 2, -1, 25, 0, 5, 1, 1, 2, 2, 2, 4, 2, 8, 2, 16, 2,
    32, 2, 64, 2, -128, 2, 26, 2, 52, 2, 103, 3, -51, 3, 41, 4, 82, 4, -92, 4,
    33, 4, 66, 4, -124, 5, 27, 5, 53, 5, 105, 6, 21, 6, 42, 6, 84, 6, 17, 7, 34,
    7, 68, 7, 2, 8, 3, 8, 6, 8, 108, 9, -41, 9, 43, 10, 86, 9, -84, 10, 35, 10,
    69, 10, -118, 11, 28, 11, 55, 12, 11, 13, 22, 13, 44, 13, 88, 13, -80, 13,
    36, 13, 71, 13, -115, 14, 29, 14, 57, 14, 113, 15, -30, 15, 46, 15, 91, 15,
    19, 16, 37, 16, 73, 16, 2, 17, 3, 17, 6, 17
};

/* min(2^32-1, 10^e-1) for e in range 0 through 10 */
static const uint32_t mag_ndigits_dec_threshold[] = {
    0, 9U, 99U, 999U, 9999U, 99999U, 999999U,
    9999999U, 99999999U, 999999999U, 0xffffffffU
};

/* Compute the number of digits in the decimal representation of x. */
static size_t mag_ndigits_dec(uint32_t x) {
    size_t t = ((mag_fls(x | 1) * 77) >> 8) + 1; /* 2^8/77 is roughly log2(10) */
    return t + (x > mag_ndigits_dec_threshold[t]);
}

/* -- Extended precision arithmetic --------------------------------------- */

/*
** The "nd" format is a fixed-precision decimal representation for numbers. It
** consists of up to 64 uint32_t values, with each uint32_t storing a value
** in the range [0, 1e9). A number in "nd" format consists of three variables:
**
**  uint32_t nd[64];
**  uint32_t ndlo;
**  uint32_t ndhi;
**
** The integral part of the number is stored in nd[0 ... ndhi], the value of
** which is sum{i in [0, ndhi] | nd[i] * 10^(9*i)}. If the fractional part of
** the number is zero, ndlo is zero. Otherwise, the fractional part is stored
** in nd[ndlo ... 63], the value of which is taken to be
** sum{i in [ndlo, 63] | nd[i] * 10^(9*(i-64))}.
**
** If the array part had 128 elements rather than 64, then every double would
** have an exact representation in "nd" format. With 64 elements, all integral
** doubles have an exact representation, and all non-integral doubles have
** enough digits to make both %.99e and %.99f do the right thing.
*/
#define MAG_ND_MUL2K_MAX_SHIFT 29
#define MAG_ND_MUL2K_DIV1E9(val) ((uint32_t)((val) / 1000000000))

/* Multiply nd by 2^k and add carry_in (ndlo is assumed to be zero). */
static uint32_t nd_mul2k(uint32_t *nd, uint32_t ndhi, uint32_t k, uint32_t carry_in, mag_format_flags_t sf) {
    uint32_t i, ndlo = 0, start = 1;
    /* Performance hacks. */
    if (k > MAG_ND_MUL2K_MAX_SHIFT*2 && MAG_FMT_FP(sf) != MAG_FMT_FP(MAG_FMT_T_FP_F)) {
        start = ndhi - (MAG_FMT_PREC(sf) + 17) / 8;
    }
    /* Real logic. */
    while (k >= MAG_ND_MUL2K_MAX_SHIFT) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << MAG_ND_MUL2K_MAX_SHIFT) | carry_in;
            carry_in = MAG_ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) {
            nd[++ndhi] = carry_in; carry_in = 0;
            if (start++ == ndlo) ++ndlo;
        }
        k -= MAG_ND_MUL2K_MAX_SHIFT;
    }
    if (k) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << k) | carry_in;
            carry_in = MAG_ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) nd[++ndhi] = carry_in;
    }
    return ndhi;
}

/* Divide nd by 2^k (ndlo is assumed to be zero). */
static uint32_t nd_div2k(uint32_t *nd, uint32_t ndhi, uint32_t k, mag_format_flags_t sf) {
    uint32_t ndlo = 0, stop1 = ~0, stop2 = ~0;
    /* Performance hacks. */
    if (!ndhi) {
        if (!nd[0]) {
            return 0;
        } else {
            uint32_t s = mag_ffs(nd[0]);
            if (s >= k) { nd[0] >>= k; return 0; }
            nd[0] >>= s; k -= s;
        }
    }
    if (k > 18) {
        if (MAG_FMT_FP(sf) == MAG_FMT_FP(MAG_FMT_T_FP_F)) {
            stop1 = 63 - (int32_t)MAG_FMT_PREC(sf) / 9;
        } else {
            int32_t floorlog2 = ndhi * 29 + mag_fls(nd[ndhi]) - k;
            int32_t floorlog10 = (int32_t)(floorlog2 * 0.30102999566398114);
            stop1 = 62 + (floorlog10 - (int32_t)MAG_FMT_PREC(sf)) / 9;
            stop2 = 61 + ndhi - (int32_t)MAG_FMT_PREC(sf) / 8;
        }
    }
    /* Real logic. */
    while (k >= 9) {
        uint32_t i = ndhi, carry = 0;
        for (;;) {
            uint32_t val = nd[i];
            nd[i] = (val >> 9) + carry;
            carry = (val & 0x1ff) * 1953125;
            if (i == ndlo) break;
            i = (i - 1) & 0x3f;
        }
        if (ndlo != stop1 && ndlo != stop2) {
            if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
            if (!nd[ndhi]) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
        } else if (!nd[ndhi]) {
            if (ndhi != ndlo) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
            else return ndlo;
        }
        k -= 9;
    }
    if (k) {
        uint32_t mask = (1U << k) - 1, mul = 1000000000 >> k, i = ndhi, carry = 0;
        for (;;) {
            uint32_t val = nd[i];
            nd[i] = (val >> k) + carry;
            carry = (val & mask) * mul;
            if (i == ndlo) break;
            i = (i - 1) & 0x3f;
        }
        if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
    }
    return ndlo;
}

/* Add m*10^e to nd (assumes ndlo <= e/9 <= ndhi and 0 <= m <= 9). */
static uint32_t nd_add_m10e(uint32_t *nd, uint32_t ndhi, uint8_t m, int32_t e) {
    uint32_t i, carry;
    if (e >= 0) {
        i = (uint32_t)e/9;
        carry = m * (mag_ndigits_dec_threshold[e - (int32_t)i*9] + 1);
    } else {
        int32_t f = (e-8)/9;
        i = (uint32_t)(64 + f);
        carry = m * (mag_ndigits_dec_threshold[e - f*9] + 1);
    }
    for (;;) {
        uint32_t val = nd[i] + carry;
        if (mag_unlikely(val >= 1000000000)) {
            val -= 1000000000;
            nd[i] = val;
            if (mag_unlikely(i == ndhi)) {
                ndhi = (ndhi + 1) & 0x3f;
                nd[ndhi] = 1;
                break;
            }
            carry = 1;
            i = (i + 1) & 0x3f;
        } else {
            nd[i] = val;
            break;
        }
    }
    return ndhi;
}

static bool nd_similar(uint32_t *nd, uint32_t ndhi, uint32_t* ref, size_t hilen, size_t prec) {
    char nd9[9], ref9[9];
    if (hilen <= prec) {
        if (mag_unlikely(nd[ndhi] != *ref)) return 0;
        prec -= hilen; ref--; ndhi = (ndhi - 1) & 0x3f;
        if (prec >= 9) {
            if (mag_unlikely(nd[ndhi] != *ref)) return 0;
            prec -= 9; ref--; ndhi = (ndhi - 1) & 0x3f;
        }
    } else {
        prec -= hilen - 9;
    }
    mag_assert(prec < 9, "bad precision %zu", prec);
    mag_wuint9(nd9, nd[ndhi]);
    mag_wuint9(ref9, *ref);
    return !memcmp(nd9, ref9, prec) && (nd9[prec] < '5') == (ref9[prec] < '5');
}

char *mag_fmt_e11m52(char *p, double n, mag_format_flags_t sf) {
    size_t width = MAG_FMT_WIDTH(sf), prec = MAG_FMT_PREC(sf), len;
    union {
        uint64_t u64;
        double n;
        struct { /* TODO: make endian aware */
            uint32_t lo, hi;
        } u32;
    } t = {.n = n};
    if (mag_unlikely((t.u32.hi << 1) >= 0xffe00000)) {
        /* Handle non-finite values uniformly for %a, %e, %f, %g. */
        int32_t prefix = 0, ch = (sf & MAG_FMT_F_UPPER) ? 0x202020 : 0;
        if (((t.u32.hi & 0x000fffff) | t.u32.lo) != 0) {
            ch ^= ('n' << 16) | ('a' << 8) | 'n';
            if ((sf & MAG_FMT_F_SPACE)) prefix = ' ';
        } else {
            ch ^= ('i' << 16) | ('n' << 8) | 'f';
            if ((t.u32.hi & 0x80000000)) prefix = '-';
            else if ((sf & MAG_FMT_F_PLUS)) prefix = '+';
            else if ((sf & MAG_FMT_F_SPACE)) prefix = ' ';
        }
        len = 3 + (prefix != 0);
        if (!(sf & MAG_FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
        if (prefix) *p++ = prefix;
        *p++ = (char)(ch >> 16); *p++ = (char)(ch >> 8); *p++ = (char)ch;
    } else if (MAG_FMT_FP(sf) == MAG_FMT_FP(MAG_FMT_T_FP_A)) {
        /* %a */
        const char* hexdig = (sf & MAG_FMT_F_UPPER) ? "0123456789ABCDEFPX" : "0123456789abcdefpx";
        int32_t e = (t.u32.hi >> 20) & 0x7ff;
        char prefix = 0, eprefix = '+';
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & MAG_FMT_F_PLUS)) prefix = '+';
        else if ((sf & MAG_FMT_F_SPACE)) prefix = ' ';
        t.u32.hi &= 0xfffff;
        if (e) {
            t.u32.hi |= 0x100000;
            e -= 1023;
        } else if (t.u32.lo | t.u32.hi) {
            /* Non-zero denormal - normalise it. */
            uint32_t shift = t.u32.hi ? 20-mag_fls(t.u32.hi) : 52-mag_fls(t.u32.lo);
            e = -1022 - shift;
            t.u64 <<= shift;
        }
        /* abs(n) == t.uint64_t * 2^(e - 52) */
        /* If n != 0, bit 52 of t.uint64_t is set, and is the highest set bit. */
        if ((int32_t)prec < 0) {
            /* Default precision: use smallest precision giving exact result. */
            prec = t.u32.lo ? 13-mag_ffs(t.u32.lo)/4 : 5-mag_ffs(t.u32.hi|0x100000)/4;
        } else if (prec < 13) {
            /* Precision is sufficiently low as to maybe require rounding. */
            t.u64 += (((uint64_t)1) << (51 - prec*4));
        }
        if (e < 0) {
            eprefix = '-';
            e = -e;
        }
        len = 5 + mag_ndigits_dec((uint32_t)e) + prec + (prefix != 0)
              + ((prec | (sf & MAG_FMT_F_ALT)) != 0);
        if (!(sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO))) {
            while (width-- > len) *p++ = ' ';
        }
        if (prefix) *p++ = prefix;
        *p++ = '0';
        *p++ = hexdig[17]; /* x or X */
        if ((sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO)) == MAG_FMT_F_ZERO) {
            while (width-- > len) *p++ = '0';
        }
        *p++ = '0' + (t.u32.hi >> 20); /* Usually '1', sometimes '0' or '2'. */
        if ((prec | (sf & MAG_FMT_F_ALT))) {
            /* Emit fractional part. */
            char* q = p + 1 + prec;
            *p = '.';
            if (prec < 13) t.u64 >>= (52 - prec*4);
            else while (prec > 13) p[prec--] = '0';
            while (prec) { p[prec--] = hexdig[t.u64 & 15]; t.u64 >>= 4; }
            p = q;
        }
        *p++ = hexdig[16]; /* p or P */
        *p++ = eprefix; /* + or - */
        p = mag_wint(p, e);
    } else {
        /* %e or %f or %g - begin by converting n to "nd" format. */
        uint32_t nd[64];
        uint32_t ndhi = 0, ndlo, i;
        int32_t e = (int32_t)(t.u32.hi >> 20) & 0x7ff, ndebias = 0;
        char prefix = 0, *q;
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & MAG_FMT_F_PLUS)) prefix = '+';
        else if ((sf & MAG_FMT_F_SPACE)) prefix = ' ';
        prec += ((int32_t)prec >> 31) & 7; /* Default precision is 6. */
        if (MAG_FMT_FP(sf) == MAG_FMT_FP(MAG_FMT_T_FP_G)) {
            /* %g - decrement precision if non-zero (to make it like %e). */
            prec--;
            prec ^= (uint32_t)((int32_t)prec >> 31);
        }
        if ((sf & MAG_FMT_T_FP_E) && prec < 14 && n != 0) {
            /* Precision is sufficiently low that rescaling will probably work. */
            if ((ndebias = mag_rescale_e[e >> 6])) {
                t.n = n * mag_rescale_n[e >> 6];
                if (mag_unlikely(!e)) t.n *= 1e10, ndebias -= 10;
                t.u64 -= 2; /* Convert 2ulp below (later we convert 2ulp above). */
                nd[0] = 0x100000 | (t.u32.hi & 0xfffff);
                e = ((int32_t)(t.u32.hi >> 20) & 0x7ff) - 1075 - (MAG_ND_MUL2K_MAX_SHIFT < 29);
                goto load_t_lo; rescale_failed:
                t.n = n;
                e = (int32_t)(t.u32.hi >> 20) & 0x7ff;
                ndebias = 0;
                ndhi = 0;
            }
        }
        nd[0] = t.u32.hi & 0xfffff;
        if (e == 0) e++; else nd[0] |= 0x100000;
        e -= 1043;
        if (t.u32.lo) {
            e -= 32 + (MAG_ND_MUL2K_MAX_SHIFT < 29); load_t_lo:
#if MAG_ND_MUL2K_MAX_SHIFT >= 29
            nd[0] = (nd[0]<<3) | (t.u32.lo>>29);
            ndhi = nd_mul2k(nd, ndhi, 29, t.u32.lo & 0x1fffffff, sf);
#elif MAG_ND_MUL2K_MAX_SHIFT >= 11
            ndhi = nd_mul2k(nd, ndhi, 11, t.u32.lo>>21, sf);
            ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo>>10) & 0x7ff, sf);
            ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo<<1) & 0x7ff, sf);
#else
#error "MAG_ND_MUL2K_MAX_SHIFT not big enough"
#endif
        }
        if (e >= 0) {
            ndhi = nd_mul2k(nd, ndhi, (uint32_t)e, 0, sf);
            ndlo = 0;
        } else {
            ndlo = nd_div2k(nd, ndhi, (uint32_t)-e, sf);
            if (ndhi && !nd[ndhi]) ndhi--;
        }
        /* |n| == nd * 10^ndebias (for slightly loose interpretation of ==) */
        if ((sf & MAG_FMT_T_FP_E)) {
            /* %e or %g - assume %e and start by calculating nd's exponent (nde). */
            char eprefix = '+';
            int32_t nde = -1;
            size_t hilen;
            if (ndlo && !nd[ndhi]) {
                ndhi = 64; do {} while (!nd[--ndhi]);
                nde -= 64 * 9;
            }
            hilen = mag_ndigits_dec(nd[ndhi]);
            nde += (int32_t)(ndhi * 9 + hilen);
            if (ndebias) {
                /*
                ** Rescaling was performed, but this introduced some error, and might
                ** have pushed us across a rounding boundary. We check whether this
                ** error affected the result by introducing even more error (2ulp in
                ** either direction), and seeing whether a rounding boundary was
                ** crossed. Having already converted the -2ulp case, we save off its
                ** most significant digits, convert the +2ulp case, and compare them.
                */
                int32_t eidx = e + 70 + (MAG_ND_MUL2K_MAX_SHIFT < 29)
                               + (t.u32.lo >= 0xfffffffe && !(~t.u32.hi << 12));
                const int8_t *m_e = mag_four_ulp_m_e + eidx * 2;
                mag_assert(0 <= eidx && eidx < 128, "bad eidx %d", eidx);
                nd[33] = nd[ndhi];
                nd[32] = nd[(ndhi - 1) & 0x3f];
                nd[31] = nd[(ndhi - 2) & 0x3f];
                nd_add_m10e(nd, ndhi, (uint8_t)*m_e, m_e[1]);
                if (mag_unlikely(!nd_similar(nd, ndhi, nd + 33, hilen, prec + 1))) {
                    goto rescale_failed;
                }
            }
            if ((int32_t)(prec - nde) < (0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, (int32_t)nde - prec - 1);
                nde += (hilen != mag_ndigits_dec(nd[ndhi]));
            }
            nde += ndebias;
            if ((sf & MAG_FMT_T_FP_F)) {
                /* %g */
                if ((int32_t)prec >= nde && nde >= -4) {
                    if (nde < 0) ndhi = 0;
                    prec -= nde;
                    goto g_format_like_f;
                } else if (!(sf & MAG_FMT_F_ALT) && prec && width > 5) {
                    /* Decrease precision in order to strip trailing zeroes. */
                    char tail[9];
                    uint32_t maxprec = hilen - 1 + ((ndhi - ndlo) & 0x3f) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = (ndhi - (((int32_t)(prec - hilen) + 9) / 9)) & 0x3f;
                    i = prec - hilen - (((ndhi - ndlo) & 0x3f) * 9) + 10;
                    mag_wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == ndhi) { prec = 0; break; }
                            ndlo = (ndlo + 1) & 0x3f;
                            mag_wuint9(tail, nd[ndlo]);
                            i = 9;
                        }
                    }
                }
            }
            if (nde < 0) {
                /* Make nde non-negative. */
                eprefix = '-';
                nde = -nde;
            }
            len = 3 + prec + (prefix != 0) + mag_ndigits_dec((uint32_t)nde) + (nde < 10)
                  + ((prec | (sf & MAG_FMT_F_ALT)) != 0);
            if (!(sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO)) == MAG_FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            q = mag_wint(p + 1, nd[ndhi]);
            p[0] = p[1]; /* Put leading digit in the correct place. */
            if ((prec | (sf & MAG_FMT_F_ALT))) {
                /* Emit fractional part. */
                p[1] = '.'; p += 2;
                prec -= (size_t)(q - p); p = q; /* Account for digits already emitted. */
                /* Then emit chunks of 9 digits (this may emit 8 digits too many). */
                for (i = ndhi; (int32_t)prec > 0 && i != ndlo; prec -= 9) {
                    i = (i - 1) & 0x3f;
                    p = mag_wuint9(p, nd[i]);
                }
                if ((sf & MAG_FMT_T_FP_F) && !(sf & MAG_FMT_F_ALT)) {
                    /* %g (and not %#g) - strip trailing zeroes. */
                    p += (int32_t)prec & ((int32_t)prec >> 31);
                    while (p[-1] == '0') p--;
                    if (p[-1] == '.') p--;
                } else {
                    /* %e (or %#g) - emit trailing zeroes. */
                    while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
                    p += (int32_t)prec;
                }
            } else {
                p++;
            }
            *p++ = (sf & MAG_FMT_F_UPPER) ? 'E' : 'e';
            *p++ = eprefix; /* + or - */
            if (nde < 10) *p++ = '0'; /* Always at least two digits of exponent. */
            p = mag_wint(p, nde);
        } else {
            /* %f (or, shortly, %g in %f style) */
            if (prec < (size_t)(0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, 0 - prec - 1);
            }
            g_format_like_f:
            if ((sf & MAG_FMT_T_FP_E) && !(sf & MAG_FMT_F_ALT) && prec && width) {
                /* Decrease precision in order to strip trailing zeroes. */
                if (ndlo) {
                    /* nd has a fractional part; we need to look at its digits. */
                    char tail[9];
                    uint32_t maxprec = (64 - ndlo) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = 64 - (prec + 8) / 9;
                    i = prec - ((63 - ndlo) * 9);
                    mag_wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == 63) { prec = 0; break; }
                            mag_wuint9(tail, nd[++ndlo]);
                            i = 9;
                        }
                    }
                } else {
                    /* nd has no fractional part, so precision goes straight to zero. */
                    prec = 0;
                }
            }
            len = ndhi * 9 + mag_ndigits_dec(nd[ndhi]) + prec + (prefix != 0)
                  + ((prec | (sf & MAG_FMT_F_ALT)) != 0);
            if (!(sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (MAG_FMT_F_LEFT | MAG_FMT_F_ZERO)) == MAG_FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            /* Emit integer part. */
            p = mag_wint(p, nd[ndhi]);
            i = ndhi;
            while (i) p = mag_wuint9(p, nd[--i]);
            if ((prec | (sf & MAG_FMT_F_ALT))) {
                /* Emit fractional part. */
                *p++ = '.';
                /* Emit chunks of 9 digits (this may emit 8 digits too many). */
                while ((int32_t)prec > 0 && i != ndlo) {
                    i = (i - 1) & 0x3f;
                    p = mag_wuint9(p, nd[i]);
                    prec -= 9;
                }
                if ((sf & MAG_FMT_T_FP_E) && !(sf & MAG_FMT_F_ALT)) {
                    /* %g (and not %#g) - strip trailing zeroes. */
                    p += (int32_t)prec & ((int32_t)prec >> 31);
                    while (p[-1] == '0') p--;
                    if (p[-1] == '.') p--;
                } else {
                    /* %f (or %#g) - emit trailing zeroes. */
                    while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
                    p += (int32_t)prec;
                }
            }
        }
    }
    if ((sf & MAG_FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
    return p;
}

typedef struct mag_tensor_format_context_t {
    mag_sstream_t *ss;
    const void *buf;
    mag_dtype_t dtype;
    const mag_coords_iter_t *iter;
    int64_t idx[MAG_MAX_DIMS];
    int64_t head;
    int64_t tail;
    bool trunc;
    size_t pad;
    size_t linewidth;
    size_t col;
} mag_tensor_format_context_t;

static void mag_fmt_putc(mag_tensor_format_context_t *fmt, char c) {
    mag_sstream_putc(fmt->ss, c);
    if (c == '\n') fmt->col = 0;
    else ++fmt->col;
}

static void mag_fmt_indent(mag_tensor_format_context_t *fmt, int depth) {
    for (size_t i=0; i < fmt->pad; ++i) mag_fmt_putc(fmt, ' ');
    for (int i=0; i <= depth; ++i) mag_fmt_putc(fmt, ' ');
}

static char *mag_fmt_scalar(char (*fmt)[MAG_FMT_BUF_MAX], const void *buf, int64_t i, mag_dtype_t type) {
    int64_t nb = (int64_t)mag_type_trait(type)->size;
    const void *val = (const uint8_t *)buf + i*nb; /* Pointer to the value */
    if (type == MAG_DTYPE_BOOLEAN) {
        int n = snprintf(*fmt, sizeof(*fmt), "%s", *(const uint8_t *)val ? "True" : "False");
        return *fmt + n;
    }
    switch (type) {
        case MAG_DTYPE_FLOAT32: return mag_fmt_e11m52(*fmt, *(const float *)val, MAG_FMT_G5);
        case MAG_DTYPE_FLOAT16: return mag_fmt_e11m52(*fmt, mag_float16_to_float32_soft_fp(*(const mag_float16_t *)val), MAG_FMT_G5);
        case MAG_DTYPE_UINT8: return mag_fmt_uint64(*fmt, *(const uint8_t *)val);
        case MAG_DTYPE_INT8: return mag_fmt_int64(*fmt, *(const int8_t *)val);
        case MAG_DTYPE_UINT16: return mag_fmt_uint64(*fmt, *(const uint16_t *)val);
        case MAG_DTYPE_INT16: return mag_fmt_int64(*fmt, *(const int16_t *)val);
        case MAG_DTYPE_UINT32: return mag_fmt_uint64(*fmt, *(const uint32_t *)val);
        case MAG_DTYPE_INT32: return mag_fmt_int64(*fmt, *(const int32_t *)val);
        case MAG_DTYPE_UINT64: return mag_fmt_uint64(*fmt, *(const uint64_t *)val);
        case MAG_DTYPE_INT64: return mag_fmt_int64(*fmt, *(const int64_t *)val);
        default: mag_panic("Unknown dtype for formatting: %d", type);
    }
}

static bool mag_fmt_lastdim_elem(
    mag_tensor_format_context_t *fmt,
    int depth,
    int64_t k,
    int64_t heads,
    int64_t tails,
    int64_t dim_size,
    bool use_ellipsis,
    char (*tmp)[MAG_FMT_BUF_MAX],
    bool *out_ellipsis,
    size_t *out_elen
) {
    const mag_coords_iter_t *iter = fmt->iter;
    bool ellipsis = use_ellipsis && (k == heads);
    *out_ellipsis = ellipsis;
    if (ellipsis) {
        *out_elen = 3; /* "..." */
        return true;
    }
    fmt->idx[depth] = k < heads ? k : dim_size - tails + (k - heads - (use_ellipsis ? 1 : 0));
    int64_t off = mag_coords_iter_offset_at(iter, fmt->idx);
    char *e = mag_fmt_scalar(tmp, fmt->buf, off, fmt->dtype);
    if (mag_unlikely(!e)) return false;
    ptrdiff_t len = e-*tmp;
    if (mag_unlikely(len <= 0)) {
        *out_elen = 0;
        return false;
    }
    *out_elen = (size_t)len;
    return true;
}

static void mag_tensor_fmt_recursive(mag_tensor_format_context_t *fmt, int depth) {
    const mag_coords_iter_t *iter = fmt->iter;
    char tmp[MAG_FMT_BUF_MAX];
    if (depth == iter->rank) { /* scalar leaf */
        int64_t off = mag_coords_iter_offset_at(iter, fmt->idx);
        char *e = mag_fmt_scalar(&tmp, fmt->buf, off, fmt->dtype);
        if (mag_unlikely(!e)) return;
        ptrdiff_t len = e - tmp;
        if (mag_likely(len > 0)) mag_sstream_append_strn(fmt->ss, tmp, (size_t)len);
        return;
    }
    int64_t dim = iter->shape[depth];
    bool last_dim = iter->rank - depth == 1;
    int64_t head = fmt->head;
    int64_t tail = fmt->tail;
    if (!fmt->trunc || head + tail >= dim) {
        head = dim;
        tail = 0;
    }
    bool use_ellipsis = head + tail < dim;
    int64_t tails = use_ellipsis ? tail : 0;
    int64_t total = head + tails + (use_ellipsis ? 1 : 0);
    mag_fmt_putc(fmt, '[');
    if (last_dim) {  /* Pass 1: compute max element width (including numbers and "...") */
        size_t max_width = 0;
        for (int64_t k=0; k < total; ++k) {
            bool ellipsis;
            size_t elen = 0;
            if (mag_unlikely(!mag_fmt_lastdim_elem(fmt, depth, k, head, tails, dim, use_ellipsis, &tmp, &ellipsis, &elen)))
                continue;
            max_width = mag_xmax(max_width, elen);
        }
        for (int64_t k=0; k < total; ++k) { /* Pass 2: actually print, with linewidth + right alignment */
            bool ellipsis;
            size_t elen = 0;
            if (mag_unlikely(!mag_fmt_lastdim_elem(fmt, depth, k, head, tails, dim, use_ellipsis, &tmp, &ellipsis, &elen)))
                continue;
            if (k > 0) {
                if (fmt->linewidth > 0 && fmt->col+2 + max_width > fmt->linewidth) {
                    mag_fmt_putc(fmt, ',');
                    mag_fmt_putc(fmt, '\n');
                    mag_fmt_indent(fmt, depth);
                } else {
                    mag_fmt_putc(fmt, ',');
                    mag_fmt_putc(fmt, ' ');
                }
            }
            size_t pad = max_width > elen ? max_width - elen : 0;
            for (size_t i=0; i < pad; ++i)
                mag_fmt_putc(fmt, ' ');
            if (ellipsis) {
                mag_fmt_putc(fmt, '.');
                mag_fmt_putc(fmt, '.');
                mag_fmt_putc(fmt, '.');
            } else {
                for (size_t i=0; i < elen; ++i)
                    mag_fmt_putc(fmt, tmp[i]);
            }
        }
    } else {
        for (int64_t k=0; k < total; ++k) {
            if (use_ellipsis && k == head) mag_sstream_append(fmt->ss, "...");
            else {
                fmt->idx[depth] = k < head ? k : dim - tails + (k - head - (use_ellipsis ? 1 : 0));
                mag_tensor_fmt_recursive(fmt, depth + 1);
            }
            if (k != total - 1) {
                mag_fmt_putc(fmt, ',');
                if (iter->rank - depth > 1) {
                    mag_fmt_putc(fmt, '\n');
                    mag_fmt_indent(fmt, depth);
                } else mag_fmt_putc(fmt, ' ');
            }
        }
    }
    mag_fmt_putc(fmt, ']');
}

char *mag_tensor_to_string(mag_tensor_t *tensor, int64_t head, int64_t tail, int64_t threshold) {
    mag_assert(mag_device_is(tensor->storage->device, "cpu"), "Tensor must be on CPU to convert to string.");
    head = head < 0 ? MAG_FMT_TENSOR_DEFAULT_HEAD_ELEMS : head;
    tail = tail < 0 ? MAG_FMT_TENSOR_DEFAULT_TAIL_ELEMS : tail;
    threshold = threshold < 0 ? MAG_FMT_TENSOR_DEFAULT_THRESHOLD : threshold;
    mag_sstream_t ss;
    mag_sstream_init(&ss);
    const char *prefix = "Tensor(";
    size_t pad = strlen(prefix);
    mag_sstream_append(&ss, prefix);
    mag_coords_iter_t iter;
    mag_coords_iter_init(&iter, &tensor->coords);
    mag_tensor_format_context_t fmt = {
        .ss = &ss,
        .buf = (const void *)mag_tensor_data_ptr(tensor),
        .dtype = tensor->dtype,
        .iter = &iter,
        .idx = {0},
        .head = head,
        .tail = tail,
        .trunc = tensor->numel > threshold,
        .pad = pad,
        .linewidth = MAG_FMT_TENSOR_DEFAULT_LINE_WIDTH,
        .col = pad
    };
    memset(fmt.idx, 0, sizeof(fmt.idx));
    mag_tensor_fmt_recursive(&fmt, 0); /* Recursive format */
    mag_sstream_append(&ss, ", dtype=%s, device=%s)", mag_type_trait(tensor->dtype)->name, tensor->storage->device->id);
    return ss.buf; /* Return the string, must be freed with mag_tensor_to_string_free_data. */
}

void mag_tensor_to_string_free_data(char *ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}
