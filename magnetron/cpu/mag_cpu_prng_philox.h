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
typedef struct mag_philox4x32_ctr_t {
    uint32_t v[4];
} mag_philox4x32_ctr_t;

typedef struct mag_philox4x32_key_t {
    uint32_t v[2];
} mag_philox4x32_key_t;

typedef struct mag_philox4x32_stream_t {
    mag_philox4x32_ctr_t ctr;
    mag_philox4x32_key_t key;
} mag_philox4x32_stream_t;

#ifdef __cplusplus
}
#endif

#endif
