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

#ifndef MAG_FMT_H
#define MAG_FMT_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_E8M23_FMT_BUF_SIZE 128

extern char *mag_fmt_e8m23(char (*p)[MAG_E8M23_FMT_BUF_SIZE], mag_e8m23_t n);

#ifdef __cplusplus
}
#endif

#endif
