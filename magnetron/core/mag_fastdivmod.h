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

#ifndef MAG_FASTDIVMOD_H
#define MAG_FASTDIVMOD_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_fastdiv_t {
    uint64_t magic;
    uint8_t flags;
} mag_fastdiv_t;

extern MAG_EXPORT mag_fastdiv_t mag_fastdiv_init(uint64_t d);

static MAG_AINLINE uint64_t mag_fastdiv_eval(uint64_t numer, const mag_fastdiv_t *denom) {
    uint64_t q = mag_mulhilo64(numer, denom->magic);
    return (((numer-q)>>1)+q)>>denom->flags;
}

#ifdef __cplusplus
}
#endif

#endif
