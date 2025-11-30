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

#ifndef MAG_CPU_PRNG_PHILOX_H
#define MAG_CPU_PRNG_PHILOX_H

#include <core/mag_def.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_PHILOX_ROUNDS 10

typedef struct mag_philox4x32_ctr_t { uint32_t v[4]; } mag_philox4x32_ctr_t;
typedef struct mag_philox4x32_key_t { uint32_t v[2]; } mag_philox4x32_key_t;
typedef struct mag_philox4x32_uint32x4_t { uint32_t v[4]; } mag_philox4x32_uint32x4_t;
typedef struct mag_philox4x32_float32x4_t { float v[4]; } mag_philox4x32_float32x4_t;
typedef struct mag_philox4x32_stream_t {
    mag_philox4x32_key_t key;
    mag_philox4x32_ctr_t ctr;
    mag_philox4x32_uint32x4_t cache;
    int idx;
} mag_philox4x32_stream_t;

static MAG_CUDA_DEVICE MAG_AINLINE void mag_philox4x32_stream_seed(mag_philox4x32_stream_t *stream, uint64_t seed, uint64_t subseq) {
    stream->key.v[0] = (uint32_t)seed;
    stream->key.v[1] = (uint32_t)(seed>>32);
    stream->ctr.v[0] = 0;
    stream->ctr.v[1] = 0;
    stream->ctr.v[2] = (uint32_t)subseq;
    stream->ctr.v[3] = (uint32_t)(subseq>>32);
    stream->idx = 4; /* Cache is empty */
}

static MAG_CUDA_DEVICE MAG_AINLINE mag_philox4x32_uint32x4_t mag_philox4x32_next_uint32x4(mag_philox4x32_stream_t *stream) {
    mag_philox4x32_key_t key = stream->key;
    mag_philox4x32_ctr_t ctr = stream->ctr;
    for (int i=0; i < MAG_PHILOX_ROUNDS; ++i) {
        uint32_t hi0, hi1;
        uint32_t lo0 = mag_mulhilo32(0xd2511f53u, ctr.v[0], &hi0);
        uint32_t lo1 = mag_mulhilo32(0xcd9e8d57u, ctr.v[2], &hi1);
        ctr.v[0] = hi1^ctr.v[1]^key.v[0];
        ctr.v[1] = lo1;
        ctr.v[2] = hi0^ctr.v[3]^key.v[1];
        ctr.v[3] = lo0;
        key.v[0] += 0x9e3779B9u;
        key.v[1] += 0xbb67ae85u;
    }
    /* Treat control as 128-bit integral and increment */
    #if !defined(__CUDA_ARCH__) && defined(__SIZEOF_INT128__) && defined(MAG_LE)
        unsigned __int128 x;
        memcpy(&x, stream->ctr.v, sizeof(x));
        ++x;
        memcpy(stream->ctr.v, &x, sizeof(x));
    #else /* Carry cascade */
        if (!++stream->ctr.v[0])
            if (!++stream->ctr.v[1])
                if (!++stream->ctr.v[2])
                    ++stream->ctr.v[3];
    #endif
    mag_philox4x32_uint32x4_t r;
    memcpy(&r, &ctr, sizeof(r));
    return r;
}

static MAG_CUDA_DEVICE MAG_AINLINE uint32_t mag_philox4x32_next_uint32(mag_philox4x32_stream_t *stream) {
    int *i = &stream->idx;
    if (*i >= 4) {
        stream->cache = mag_philox4x32_next_uint32x4(stream);
        *i = 0;
    }
    return stream->cache.v[(*i)++];
}

static MAG_CUDA_DEVICE MAG_AINLINE uint64_t mag_philox4x32_next_uint64(mag_philox4x32_stream_t *stream) {
    return (uint64_t)mag_philox4x32_next_uint32(stream)<<32 | mag_philox4x32_next_uint32(stream);
}

static MAG_CUDA_DEVICE MAG_AINLINE mag_philox4x32_float32x4_t mag_philox4x32_next_float32x4(mag_philox4x32_stream_t *stream) {
    mag_philox4x32_uint32x4_t u32x4 = mag_philox4x32_next_uint32x4(stream);
    mag_philox4x32_float32x4_t r;
    r.v[0] = 1.f/0x1.0p23f*((float)(u32x4.v[0]>>9) + 0.5f);
    r.v[1] = 1.f/0x1.0p23f*((float)(u32x4.v[1]>>9) + 0.5f);
    r.v[2] = 1.f/0x1.0p23f*((float)(u32x4.v[2]>>9) + 0.5f);
    r.v[3] = 1.f/0x1.0p23f*((float)(u32x4.v[3]>>9) + 0.5f);
    return r;
}

static MAG_CUDA_DEVICE MAG_AINLINE float mag_philox4x32_next_float32(mag_philox4x32_stream_t *stream) {
    return 1.f/0x1.0p23f*((float)(mag_philox4x32_next_uint32(stream)>>9) + 0.5f);
}

static MAG_CUDA_DEVICE MAG_AINLINE mag_philox4x32_float32x4_t mag_philox4x32_next_float32x4_uniform(mag_philox4x32_stream_t *stream, float min, float max) {
    float scale = max-min;
    mag_philox4x32_uint32x4_t u32x4 = mag_philox4x32_next_uint32x4(stream);
    mag_philox4x32_float32x4_t r;
    r.v[0] = fmaf(1.f/0x1.0p23f*((float)(u32x4.v[0]>>9) + 0.5f), scale, min);
    r.v[1] = fmaf(1.f/0x1.0p23f*((float)(u32x4.v[1]>>9) + 0.5f), scale, min);
    r.v[2] = fmaf(1.f/0x1.0p23f*((float)(u32x4.v[2]>>9) + 0.5f), scale, min);
    r.v[3] = fmaf(1.f/0x1.0p23f*((float)(u32x4.v[3]>>9) + 0.5f), scale, min);
    return r;
}

static MAG_CUDA_DEVICE MAG_AINLINE float mag_philox4x32_next_float32_uniform(mag_philox4x32_stream_t *stream, float min, float max) {
    return fmaf(1.f/0x1.0p23f*((float)(mag_philox4x32_next_uint32(stream)>>9) + 0.5f), max-min, min);
}

static MAG_CUDA_DEVICE MAG_AINLINE mag_philox4x32_float32x4_t mag_philox4x32_next_float32x4_normal(mag_philox4x32_stream_t *stream, float mean, float std) {
    mag_philox4x32_uint32x4_t u32x4 = mag_philox4x32_next_uint32x4(stream);
    mag_philox4x32_float32x4_t r;
    r.v[0] = 1.f/0x1.0p23f*((float)(u32x4.v[0]>>9) + 0.5f);
    r.v[1] = 1.f/0x1.0p23f*((float)(u32x4.v[1]>>9) + 0.5f);
    r.v[2] = 1.f/0x1.0p23f*((float)(u32x4.v[2]>>9) + 0.5f);
    r.v[3] = 1.f/0x1.0p23f*((float)(u32x4.v[3]>>9) + 0.5f);
    r.v[0] = fmaxf(r.v[0], 1e-37f);
    r.v[2] = fmaxf(r.v[2], 1e-37f);
    float rho0 = sqrtf(-2.f*logf(r.v[0]));
    float theta0 = MAG_TAU*r.v[1];
    float z0 = rho0*cosf(theta0);
    float z1 = rho0*sinf(theta0);
    float rho1 = sqrtf(-2.f*logf(r.v[2]));
    float theta1 = MAG_TAU*r.v[3];
    float z2 = rho1*cosf(theta1);
    float z3 = rho1*sinf(theta1);
    r.v[0] = fmaf(z0, std, mean);
    r.v[1] = fmaf(z1, std, mean);
    r.v[2] = fmaf(z2, std, mean);
    r.v[3] = fmaf(z3, std, mean);
    return r;
}

static MAG_CUDA_DEVICE MAG_AINLINE float mag_philox4x32_next_float32_normal(mag_philox4x32_stream_t *stream, float mean, float std) {
    float u0 = 1.f/0x1.0p23f*((float)(mag_philox4x32_next_uint32(stream)>>9) + 0.5f);
    float u1 = 1.f/0x1.0p23f*((float)(mag_philox4x32_next_uint32(stream)>>9) + 0.5f);
    return fmaf(sqrtf(-2.0f*logf(fmaxf(u0, 1e-37f)))*cosf(MAG_TAU*u1), std, mean);
}

#ifdef __cplusplus
}
#endif

#endif
