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

/*
** Fast CRC32 Castagnoli implementation using hardware acceleration if available.
** Fallback to a software implementation using a lookup table otherwise.
** Based on generations by: https://github.com/corsix/fast-crc32/.
** See MIT license at the end of this file.
*/

#if defined(__SSE4_2__) && defined(__CRC32__) && (defined(__PCLMUL__) || defined(__PCLMULQDQ__))

static uint32_t mag_xnmodp(uint64_t n) {
    uint64_t stack = ~(uint64_t)1;
    uint32_t acc, low;
    for (; n > 191; n = (n >> 1) - 16) {
        stack = (stack << 1) + (n & 1);
    }
    stack = ~stack;
    acc = 0x80000000u >> (n & 31);
    for (n >>= 5; n; --n) {
        acc = _mm_crc32_u32(acc, 0);
    }
    while ((low = stack & 1), stack >>= 1) {
        __m128i x = _mm_cvtsi32_si128(acc);
        uint64_t y = _mm_cvtsi128_si64(_mm_clmulepi64_si128(x, x, 0));
        acc = _mm_crc32_u64(0, y << low);
    }
    return acc;
}

MAG_AINLINE static __m128i mag_cmuls(uint32_t a, uint32_t b) {
    return _mm_clmulepi64_si128(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b), 0);
}

MAG_AINLINE static __m128i mag_crcshift(uint32_t crc, size_t nbytes) {
    return mag_cmuls(crc, mag_xnmodp(nbytes*8 - 33));
}

#endif

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__VPCLMULQDQ__) && defined(__CRC32__)

#define mag_clmullo(a, b) (_mm512_clmulepi64_epi128((a), (b), 0))
#define mag_clmulhi(a, b) (_mm512_clmulepi64_epi128((a), (b), 17))

static uint32_t mag_crc32c(const void *buffer, size_t len) {
    const uint8_t *buf = buffer;
    uint32_t crc0 = ~0u;
    for (; len && ((uintptr_t)buf & 7); --len) {
        crc0 = _mm_crc32_u8(crc0, *buf++);
    }
    while (((uintptr_t)buf & 56) && len >= 8) {
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
        buf += 8;
        len -= 8;
    }
    if (len >= 208) {
        size_t blk = (len - 8)/200;
        size_t klen = blk*8;
        const uint8_t *buf2 = buf + 0;
        uint64_t vc;
        __m128i z0;
        __m512i x0 = _mm512_loadu_si512((const void *)buf2), y0;
        __m512i x1 = _mm512_loadu_si512((const void *)(buf2 + 64)), y1;
        __m512i x2 = _mm512_loadu_si512((const void *)(buf2 + 128)), y2;
        __m512i k;
        k = _mm512_broadcast_i32x4(_mm_setr_epi32(0xa87ab8a8, 0, 0xab7aff2a, 0));
        x0 = _mm512_xor_si512(_mm512_zextsi128_si512(_mm_cvtsi32_si128(crc0)), x0);
        crc0 = 0;
        buf2 += 192;
        len -= 200;
        buf += blk*192;
        while (len >= 208) {
            y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
            y1 = mag_clmullo(x1, k), x1 = mag_clmulhi(x1, k);
            y2 = mag_clmullo(x2, k), x2 = mag_clmulhi(x2, k);
            x0 = _mm512_ternarylogic_epi64(x0, y0, _mm512_loadu_si512((const void *)buf2), 0x96);
            x1 = _mm512_ternarylogic_epi64(x1, y1, _mm512_loadu_si512((const void *)(buf2 + 64)), 0x96);
            x2 = _mm512_ternarylogic_epi64(x2, y2, _mm512_loadu_si512((const void *)(buf2 + 128)), 0x96);
            crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
            buf += 8;
            buf2 += 192;
            len -= 200;
        }
        k = _mm512_broadcast_i32x4(_mm_setr_epi32(0x740eef02, 0, 0x9e4addf8, 0));
        y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
        x0 = _mm512_ternarylogic_epi64(x0, y0, x1, 0x96);
        x1 = x2;
        y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
        x0 = _mm512_ternarylogic_epi64(x0, y0, x1, 0x96);
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
        buf += 8;
        vc = 0;
        k = _mm512_setr_epi32(0x1c291d04, 0, 0xddc0152b, 0, 0x3da6d0cb, 0, 0xba4fc28e, 0, 0xf20c0dfe, 0, 0x493c7d27, 0, 0, 0, 0, 0);
        y0 = mag_clmullo(x0, k), k = mag_clmulhi(x0, k);
        y0 = _mm512_xor_si512(y0, k);
        z0 = _mm_ternarylogic_epi64(_mm512_castsi512_si128(y0), _mm512_extracti32x4_epi32(y0, 1), _mm512_extracti32x4_epi32(y0, 2), 0x96);
        z0 = _mm_xor_si128(z0, _mm512_extracti32x4_epi32(x0, 3));
        vc ^= _mm_extract_epi64(mag_crcshift(_mm_crc32_u64(_mm_crc32_u64(0, _mm_extract_epi64(z0, 0)), _mm_extract_epi64(z0, 1)), klen*1 + 8), 0);
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf ^ vc), buf += 8;
        len -= 8;
    }
    if (len >= 32) {
        size_t klen = ((len - 8)/24)*8;
        uint32_t crc1 = 0;
        uint32_t crc2 = 0;
        __m128i vc0;
        __m128i vc1;
        uint64_t vc;
        do {
            crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
            crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen));
            crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2));
            buf += 8;
            len -= 24;
        } while (len >= 32);
        vc0 = mag_crcshift(crc0, klen*2 + 8);
        vc1 = mag_crcshift(crc1, klen + 8);
        vc = _mm_extract_epi64(_mm_xor_si128(vc0, vc1), 0);
        buf += klen*2;
        crc0 = crc2;
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf ^ vc), buf += 8;
        len -= 8;
    }
    for (; len >= 8; buf += 8, len -= 8) {
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
    }
    for (; len; --len) {
        crc0 = _mm_crc32_u8(crc0, *buf++);
    }
    return ~crc0;
}

#elif defined(__AVX512F__) && defined(__AVX512VL__) && defined(__CRC32__)

#define mag_clmullo(a, b) (_mm_clmulepi64_si128((a), (b), 0))
#define mag_clmulhi(a, b) (_mm_clmulepi64_si128((a), (b), 17))

static uint32_t mag_crc32c(const void *buffer, size_t len) {
    const uint8_t *buf = buffer;
    uint32_t crc0 = ~0u;
    for (; len && (uintptr_t)buf & 7; --len) crc0 = _mm_crc32_u8(crc0, *buf++);
    if (((uintptr_t)buf & 8) && len >= 8) {
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
        buf += 8;
        len -= 8;
    }
    if (len >= 240) {
        const uint8_t *end = buf + len;
        size_t blk = (len - 0)/240;
        size_t klen = blk*32;
        const uint8_t *buf2 = buf + klen*3;
        const uint8_t *limit = buf + klen - 64;
        uint32_t crc1 = 0;
        uint32_t crc2 = 0;
        __m128i vc0;
        __m128i vc1;
        __m128i vc2;
        uint64_t vc;
        __m128i x0 = _mm_loadu_si128((const __m128i *)buf2), y0;
        __m128i x1 = _mm_loadu_si128((const __m128i *)(buf2 + 16)), y1;
        __m128i x2 = _mm_loadu_si128((const __m128i *)(buf2 + 32)), y2;
        __m128i x3 = _mm_loadu_si128((const __m128i *)(buf2 + 48)), y3;
        __m128i x4 = _mm_loadu_si128((const __m128i *)(buf2 + 64)), y4;
        __m128i x5 = _mm_loadu_si128((const __m128i *)(buf2 + 80)), y5;
        __m128i x6 = _mm_loadu_si128((const __m128i *)(buf2 + 96)), y6;
        __m128i x7 = _mm_loadu_si128((const __m128i *)(buf2 + 112)), y7;
        __m128i x8 = _mm_loadu_si128((const __m128i *)(buf2 + 128)), y8;
        __m128i k;
        k = _mm_setr_epi32(0x7e908048, 0, 0xc96cfdc0, 0);
        buf2 += 144;
        while (buf <= limit) {
            y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
            y1 = mag_clmullo(x1, k), x1 = mag_clmulhi(x1, k);
            y2 = mag_clmullo(x2, k), x2 = mag_clmulhi(x2, k);
            y3 = mag_clmullo(x3, k), x3 = mag_clmulhi(x3, k);
            y4 = mag_clmullo(x4, k), x4 = mag_clmulhi(x4, k);
            y5 = mag_clmullo(x5, k), x5 = mag_clmulhi(x5, k);
            y6 = mag_clmullo(x6, k), x6 = mag_clmulhi(x6, k);
            y7 = mag_clmullo(x7, k), x7 = mag_clmulhi(x7, k);
            y8 = mag_clmullo(x8, k), x8 = mag_clmulhi(x8, k);
            x0 = _mm_ternarylogic_epi64(x0, y0, _mm_loadu_si128((const __m128i *)buf2), 0x96);
            x1 = _mm_ternarylogic_epi64(x1, y1, _mm_loadu_si128((const __m128i *)(buf2 + 16)), 0x96);
            x2 = _mm_ternarylogic_epi64(x2, y2, _mm_loadu_si128((const __m128i *)(buf2 + 32)), 0x96);
            x3 = _mm_ternarylogic_epi64(x3, y3, _mm_loadu_si128((const __m128i *)(buf2 + 48)), 0x96);
            x4 = _mm_ternarylogic_epi64(x4, y4, _mm_loadu_si128((const __m128i *)(buf2 + 64)), 0x96);
            x5 = _mm_ternarylogic_epi64(x5, y5, _mm_loadu_si128((const __m128i *)(buf2 + 80)), 0x96);
            x6 = _mm_ternarylogic_epi64(x6, y6, _mm_loadu_si128((const __m128i *)(buf2 + 96)), 0x96);
            x7 = _mm_ternarylogic_epi64(x7, y7, _mm_loadu_si128((const __m128i *)(buf2 + 112)), 0x96);
            x8 = _mm_ternarylogic_epi64(x8, y8, _mm_loadu_si128((const __m128i *)(buf2 + 128)), 0x96);
            crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
            crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen));
            crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2));
            crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 8));
            crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 8));
            crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 8));
            crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 16));
            crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 16));
            crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 16));
            crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 24));
            crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 24));
            crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 24));
            buf += 32;
            buf2 += 144;
        }
        k = _mm_setr_epi32(0xf20c0dfe, 0, 0x493c7d27, 0);
        y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
        x0 = _mm_ternarylogic_epi64(x0, y0, x1, 0x96);
        x1 = x2, x2 = x3, x3 = x4, x4 = x5, x5 = x6, x6 = x7, x7 = x8;
        y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
        y2 = mag_clmullo(x2, k), x2 = mag_clmulhi(x2, k);
        y4 = mag_clmullo(x4, k), x4 = mag_clmulhi(x4, k);
        y6 = mag_clmullo(x6, k), x6 = mag_clmulhi(x6, k);
        x0 = _mm_ternarylogic_epi64(x0, y0, x1, 0x96);
        x2 = _mm_ternarylogic_epi64(x2, y2, x3, 0x96);
        x4 = _mm_ternarylogic_epi64(x4, y4, x5, 0x96);
        x6 = _mm_ternarylogic_epi64(x6, y6, x7, 0x96);
        k = _mm_setr_epi32(0x3da6d0cb, 0, 0xba4fc28e, 0);
        y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
        y4 = mag_clmullo(x4, k), x4 = mag_clmulhi(x4, k);
        x0 = _mm_ternarylogic_epi64(x0, y0, x2, 0x96);
        x4 = _mm_ternarylogic_epi64(x4, y4, x6, 0x96);
        k = _mm_setr_epi32(0x740eef02, 0, 0x9e4addf8, 0);
        y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
        x0 = _mm_ternarylogic_epi64(x0, y0, x4, 0x96);
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
        crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen));
        crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2));
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 8));
        crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 8));
        crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 8));
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 16));
        crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 16));
        crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 16));
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 24));
        crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 24));
        crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 24));
        vc0 = mag_crcshift(crc0, klen*2 + blk*144);
        vc1 = mag_crcshift(crc1, klen + blk*144);
        vc2 = mag_crcshift(crc2, 0 + blk*144);
        vc = _mm_extract_epi64(_mm_ternarylogic_epi64(vc0, vc1, vc2, 0x96), 0);
        crc0 = _mm_crc32_u64(0, _mm_extract_epi64(x0, 0));
        crc0 = _mm_crc32_u64(crc0, vc ^ _mm_extract_epi64(x0, 1));
        buf = buf2;
        len = end - buf;
    }
    for (; len >= 8; buf += 8, len -= 8) crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
    for (; len; --len) crc0 = _mm_crc32_u8(crc0, *buf++);
    return ~crc0;
}

#elif defined(__SSE4_2__) &&  (defined(__PCLMUL__) || defined(__PCLMULQDQ__)) && defined(__CRC32__)

#define mag_clmullo(a, b) (_mm_clmulepi64_si128((a), (b), 0))
#define mag_clmulhi(a, b) (_mm_clmulepi64_si128((a), (b), 17))

static uint32_t mag_crc32c(const void *buffer, size_t len) {
    const uint8_t *buf = buffer;
    uint32_t crc0 = ~0u;
    for (; len && ((uintptr_t)buf & 7); --len) {
        crc0 = _mm_crc32_u8(crc0, *buf++);
    }
    if (((uintptr_t)buf & 8) && len >= 8) {
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
        buf += 8;
        len -= 8;
    }
    if (len >= 208) {
        size_t blk = (len - 8)/200;
        size_t klen = blk*24;
        const uint8_t *buf2 = buf + 0;
        uint32_t crc1 = 0;
        uint32_t crc2 = 0;
        __m128i vc0;
        __m128i vc1;
        uint64_t vc;
        __m128i x0 = _mm_loadu_si128((const __m128i *)buf2), y0;
        __m128i x1 = _mm_loadu_si128((const __m128i *)(buf2 + 16)), y1;
        __m128i x2 = _mm_loadu_si128((const __m128i *)(buf2 + 32)), y2;
        __m128i x3 = _mm_loadu_si128((const __m128i *)(buf2 + 48)), y3;
        __m128i x4 = _mm_loadu_si128((const __m128i *)(buf2 + 64)), y4;
        __m128i x5 = _mm_loadu_si128((const __m128i *)(buf2 + 80)), y5;
        __m128i x6 = _mm_loadu_si128((const __m128i *)(buf2 + 96)), y6;
        __m128i x7 = _mm_loadu_si128((const __m128i *)(buf2 + 112)), y7;
        __m128i k;
        k = _mm_setr_epi32(0x6992cea2, 0, 0x0d3b6092, 0);
        x0 = _mm_xor_si128(_mm_cvtsi32_si128(crc0), x0);
        crc0 = 0;
        buf2 += 128;
        len -= 200;
        buf += blk*128;
        while (len >= 208) {
            y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
            y1 = mag_clmullo(x1, k), x1 = mag_clmulhi(x1, k);
            y2 = mag_clmullo(x2, k), x2 = mag_clmulhi(x2, k);
            y3 = mag_clmullo(x3, k), x3 = mag_clmulhi(x3, k);
            y4 = mag_clmullo(x4, k), x4 = mag_clmulhi(x4, k);
            y5 = mag_clmullo(x5, k), x5 = mag_clmulhi(x5, k);
            y6 = mag_clmullo(x6, k), x6 = mag_clmulhi(x6, k);
            y7 = mag_clmullo(x7, k), x7 = mag_clmulhi(x7, k);
            y0 = _mm_xor_si128(y0, _mm_loadu_si128((const __m128i *)buf2)), x0 = _mm_xor_si128(x0, y0);
            y1 = _mm_xor_si128(y1, _mm_loadu_si128((const __m128i *)(buf2 + 16))), x1 = _mm_xor_si128(x1, y1);
            y2 = _mm_xor_si128(y2, _mm_loadu_si128((const __m128i *)(buf2 + 32))), x2 = _mm_xor_si128(x2, y2);
            y3 = _mm_xor_si128(y3, _mm_loadu_si128((const __m128i *)(buf2 + 48))), x3 = _mm_xor_si128(x3, y3);
            y4 = _mm_xor_si128(y4, _mm_loadu_si128((const __m128i *)(buf2 + 64))), x4 = _mm_xor_si128(x4, y4);
            y5 = _mm_xor_si128(y5, _mm_loadu_si128((const __m128i *)(buf2 + 80))), x5 = _mm_xor_si128(x5, y5);
            y6 = _mm_xor_si128(y6, _mm_loadu_si128((const __m128i *)(buf2 + 96))), x6 = _mm_xor_si128(x6, y6);
            y7 = _mm_xor_si128(y7, _mm_loadu_si128((const __m128i *)(buf2 + 112))), x7 = _mm_xor_si128(x7, y7);
            crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
            crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen));
            crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2));
            crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 8));
            crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 8));
            crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 8));
            crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 16));
            crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 16));
            crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 16));
            buf += 24;
            buf2 += 128;
            len -= 200;
        }
        k = _mm_setr_epi32(0xf20c0dfe, 0, 0x493c7d27, 0);
        y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
        y2 = mag_clmullo(x2, k), x2 = mag_clmulhi(x2, k);
        y4 = mag_clmullo(x4, k), x4 = mag_clmulhi(x4, k);
        y6 = mag_clmullo(x6, k), x6 = mag_clmulhi(x6, k);
        y0 = _mm_xor_si128(y0, x1), x0 = _mm_xor_si128(x0, y0);
        y2 = _mm_xor_si128(y2, x3), x2 = _mm_xor_si128(x2, y2);
        y4 = _mm_xor_si128(y4, x5), x4 = _mm_xor_si128(x4, y4);
        y6 = _mm_xor_si128(y6, x7), x6 = _mm_xor_si128(x6, y6);
        k = _mm_setr_epi32(0x3da6d0cb, 0, 0xba4fc28e, 0);
        y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
        y4 = mag_clmullo(x4, k), x4 = mag_clmulhi(x4, k);
        y0 = _mm_xor_si128(y0, x2), x0 = _mm_xor_si128(x0, y0);
        y4 = _mm_xor_si128(y4, x6), x4 = _mm_xor_si128(x4, y4);
        k = _mm_setr_epi32(0x740eef02, 0, 0x9e4addf8, 0);
        y0 = mag_clmullo(x0, k), x0 = mag_clmulhi(x0, k);
        y0 = _mm_xor_si128(y0, x4), x0 = _mm_xor_si128(x0, y0);
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
        crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen));
        crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2));
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 8));
        crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 8));
        crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 8));
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)(buf + 16));
        crc1 = _mm_crc32_u64(crc1, *(const uint64_t *)(buf + klen + 16));
        crc2 = _mm_crc32_u64(crc2, *(const uint64_t *)(buf + klen*2 + 16));
        buf += 24;
        vc0 = mag_crcshift(crc0, klen*2 + 8);
        vc1 = mag_crcshift(crc1, klen + 8);
        vc = _mm_extract_epi64(_mm_xor_si128(vc0, vc1), 0);
        vc ^= _mm_extract_epi64(mag_crcshift(_mm_crc32_u64(_mm_crc32_u64(0, _mm_extract_epi64(x0, 0)), _mm_extract_epi64(x0, 1)), klen*3 + 8), 0);
        buf += klen*2;
        crc0 = crc2;
        crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf ^ vc), buf += 8;
        len -= 8;
    }
    for (; len >= 8; buf += 8, len -= 8) crc0 = _mm_crc32_u64(crc0, *(const uint64_t *)buf);
    for (; len; --len) crc0 = _mm_crc32_u8(crc0, *buf++);
    return ~crc0;
}


#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_CRC32) && defined(__ARM_FEATURE_CRYPTO)

MAG_AINLINE static uint64x2_t mag_clmullo(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
    uint64x2_t r;
    __asm__ __volatile__(
        "pmull %0.1q, %2.1d, %3.1d\n"
        "eor %0.16b, %0.16b, %1.16b\n"
        : "=w"(r), "+w"(c)
        : "w"(a), "w"(b)
    );
    return r;
}

MAG_AINLINE static uint64x2_t mag_clmulhi(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
    uint64x2_t r;
    __asm__ __volatile__(
        "pmull2 %0.1q, %2.2d, %3.2d\n"
        "eor %0.16b, %0.16b, %1.16b\n"
        : "=w"(r), "+w"(c)
        : "w"(a), "w"(b)
    );
    return r;
}

MAG_AINLINE static uint64x2_t mag_clmuls(uint32_t a, uint32_t b) {
    uint64x2_t r;
    __asm__ __volatile__(
        "pmull %0.1q, %1.1d, %2.1d\n"
        : "=w"(r)
        : "w"(vmovq_n_u64(a)),
        "w"(vmovq_n_u64(b))
    );
    return r;
}

static uint32_t mag_xnmodp(uint64_t n) {
    uint64_t stack = ~(uint64_t)1;
    uint32_t acc, low;
    for (; n > 191; n = (n >> 1) - 16) {
        stack = (stack << 1) + (n & 1);
    }
    stack = ~stack;
    acc = 0x80000000u >> (n & 31);
    for (n >>= 5; n; --n) {
        acc = __crc32cw(acc, 0);
    }
    while ((low = stack & 1), stack >>= 1) {
        poly8x8_t x = vreinterpret_p8_u64(vmov_n_u64(acc));
        uint64_t y = vgetq_lane_u64(vreinterpretq_u64_p16(vmull_p8(x, x)), 0);
        acc = __crc32cd(0, y << low);
    }
    return acc;
}

MAG_AINLINE static uint64x2_t mag_crc_shift(uint32_t crc, size_t nbytes) {
    return mag_clmuls(crc, mag_xnmodp(nbytes * 8 - 33));
}

static uint32_t mag_crc32c(const void *buffer, size_t len) {
    const uint8_t *buf = buffer;
    uint32_t crc0 = ~0u;
    for (; len && ((uintptr_t)buf & 7); --len) crc0 = __crc32cb(crc0, *buf++);
    if (((uintptr_t)buf & 8) && len >= 8) {
        crc0 = __crc32cd(crc0, *(const uint64_t *)buf);
        buf += 8;
        len -= 8;
    }
    if (len >= 112) {
        const uint8_t *end = buf + len;
        size_t blk = (len - 0) / 112;
        size_t klen = blk * 16;
        const uint8_t *buf2 = buf + klen * 4;
        const uint8_t *limit = buf + klen - 32;
        uint32_t crc1 = 0;
        uint32_t crc2 = 0;
        uint32_t crc3 = 0;
        uint64x2_t vc0;
        uint64x2_t vc1;
        uint64x2_t vc2;
        uint64x2_t vc3;
        uint64_t vc;
        uint64x2_t x0 = vld1q_u64((const uint64_t *)buf2), y0;
        uint64x2_t x1 = vld1q_u64((const uint64_t *)(buf2 + 16)), y1;
        uint64x2_t x2 = vld1q_u64((const uint64_t *)(buf2 + 32)), y2;
        uint64x2_t k;
        { static const uint64_t mag_alignas(16) k_[] = {0x1c291d04, 0xddc0152b}; k = vld1q_u64(k_); }
        buf2 += 48;
        while (buf <= limit) {
            y0 = mag_clmullo(x0, k, vld1q_u64((const uint64_t *)buf2)), x0 = mag_clmulhi(x0, k, y0);
            y1 = mag_clmullo(x1, k, vld1q_u64((const uint64_t *)(buf2 + 16))), x1 = mag_clmulhi(x1, k, y1);
            y2 = mag_clmullo(x2, k, vld1q_u64((const uint64_t *)(buf2 + 32))), x2 = mag_clmulhi(x2, k, y2);
            crc0 = __crc32cd(crc0, *(const uint64_t *)buf);
            crc1 = __crc32cd(crc1, *(const uint64_t *)(buf + klen));
            crc2 = __crc32cd(crc2, *(const uint64_t *)(buf + klen * 2));
            crc3 = __crc32cd(crc3, *(const uint64_t *)(buf + klen * 3));
            crc0 = __crc32cd(crc0, *(const uint64_t *)(buf + 8));
            crc1 = __crc32cd(crc1, *(const uint64_t *)(buf + klen + 8));
            crc2 = __crc32cd(crc2, *(const uint64_t *)(buf + klen * 2 + 8));
            crc3 = __crc32cd(crc3, *(const uint64_t *)(buf + klen * 3 + 8));
            buf += 16;
            buf2 += 48;
        }
        { static const uint64_t mag_alignas(16) k_[] = {0xf20c0dfe, 0x493c7d27}; k = vld1q_u64(k_); }
        y0 = mag_clmullo(x0, k, x1), x0 = mag_clmulhi(x0, k, y0);
        x1 = x2;
        y0 = mag_clmullo(x0, k, x1), x0 = mag_clmulhi(x0, k, y0);
        crc0 = __crc32cd(crc0, *(const uint64_t *)buf);
        crc1 = __crc32cd(crc1, *(const uint64_t *)(buf + klen));
        crc2 = __crc32cd(crc2, *(const uint64_t *)(buf + klen * 2));
        crc3 = __crc32cd(crc3, *(const uint64_t *)(buf + klen * 3));
        crc0 = __crc32cd(crc0, *(const uint64_t *)(buf + 8));
        crc1 = __crc32cd(crc1, *(const uint64_t *)(buf + klen + 8));
        crc2 = __crc32cd(crc2, *(const uint64_t *)(buf + klen * 2 + 8));
        crc3 = __crc32cd(crc3, *(const uint64_t *)(buf + klen * 3 + 8));
        vc0 = mag_crc_shift(crc0, klen * 3 + blk * 48);
        vc1 = mag_crc_shift(crc1, klen * 2 + blk * 48);
        vc2 = mag_crc_shift(crc2, klen + blk * 48);
        vc3 = mag_crc_shift(crc3, 0 + blk * 48);
        vc = vgetq_lane_u64(veorq_u64(veorq_u64(vc0, vc1), veorq_u64(vc2, vc3)), 0);
        crc0 = __crc32cd(0, vgetq_lane_u64(x0, 0));
        crc0 = __crc32cd(crc0, vc ^ vgetq_lane_u64(x0, 1));
        buf = buf2;
        len = end - buf;
    }
    if (len >= 32) {
        uint64x2_t x0 = vld1q_u64((const uint64_t *)buf), y0;
        uint64x2_t x1 = vld1q_u64((const uint64_t *)(buf + 16)), y1;
        uint64x2_t k;
        { static const uint64_t mag_alignas(16) k_[] = {0x3da6d0cb, 0xba4fc28e}; k = vld1q_u64(k_); }
        x0 = veorq_u64((uint64x2_t){crc0, 0}, x0);
        buf += 32;
        len -= 32;
        while (len >= 32) {
            y0 = mag_clmullo(x0, k, vld1q_u64((const uint64_t *)buf)), x0 = mag_clmulhi(x0, k, y0);
            y1 = mag_clmullo(x1, k, vld1q_u64((const uint64_t *)(buf + 16))), x1 = mag_clmulhi(x1, k, y1);
            buf += 32;
            len -= 32;
        }
        { static const uint64_t mag_alignas(16) k_[] = {0xf20c0dfe, 0x493c7d27}; k = vld1q_u64(k_); }
        y0 = mag_clmullo(x0, k, x1), x0 = mag_clmulhi(x0, k, y0);
        crc0 = __crc32cd(0, vgetq_lane_u64(x0, 0));
        crc0 = __crc32cd(crc0, vgetq_lane_u64(x0, 1));
    }
    for (; len >= 8; buf += 8, len -= 8) crc0 = __crc32cd(crc0, *(const uint64_t *)buf);
    for (; len; --len) crc0 = __crc32cb(crc0, *buf++);
    return ~crc0;
}

#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)

static uint32_t mag_crc32c(const void *buffer, size_t len) {
    const uint8_t *buf = buffer;
    uint32_t crc = ~0u;
    for (; len && ((uintptr_t)buf & 7); --len) crc = __crc32cb(crc, *buf++);
    for (; len >= 8; buf += 8, len -= 8) crc = __crc32cd(crc, *(const uint64_t*)buf);
    for (; len; --len) crc = __crc32cb(crc, *buf++);
    return ~crc;
}

#else

static uint32_t mag_crc32c(const void *buffer, size_t len) {
    const uint8_t *buf = buffer;
    uint32_t crc0 = ~0u;
    extern const uint32_t mag_crc32c_lut[256];
    for (size_t i=0; i < len; ++i)
        crc0 = (crc0>>8) ^ mag_crc32c_lut[buf[i] ^ (crc0&0xff)];
    return ~crc0;
}

#endif

/*
**  Copyright (c) 2023 Peter Cawley
**
**  Permission is hereby granted, free of charge, to any person obtaining a copy
**  of this software and associated documentation files (the "Software"), to deal
**  in the Software without restriction, including without limitation the rights
**  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
**  copies of the Software, and to permit persons to whom the Software is
**  furnished to do so, subject to the following conditions:
**
**  The above copyright notice and this permission notice shall be included in all
**  copies or substantial portions of the Software.
**
**  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
**  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
**  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
**  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
**  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
**  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
**  SOFTWARE.
*/
