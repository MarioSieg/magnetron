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

#ifndef _MSC_VER
#if !defined(__SSE__) \
    || !defined(__SSE2__) \
    || !defined(__SSE3__) \
    || !defined(__SSSE3__) \
    || !defined(__SSE4_1__) \
    || !defined(__SSE4_2__) \
    || !defined(__AVX__) \
    || !defined(__AVX2__) \
    || !defined(__F16C__) \
    || !defined(__FMA__) \
    || !defined(__AVX512F__) \
    || !defined(__AVX512BW__) \
    || !defined(__AVX512DQ__) \
    || !defined(__AVX512VL__)
#pragma message ("Current compiler lacks modern optimization flags - upgrade GCC/Clang to enable better optimizations!")
#endif
#else
#pragma message("MSVC does not allow to fine tune CPU architecture level, usine clang-cl or mingw-w64 for best performance!")
#endif
#ifdef __AVX512VNNI__
#error "BLAS specialization feature too high"
#endif

#define MAG_BLAS_SPECIALIZATION mag_cpu_blas_specialization_amd64_v4
#define MAG_BLAS_SPECIALIZATION_FEAT_REQUEST mag_cpu_blas_specialization_amd64_v4_features

#include "magnetron_cpu_blas.inl"
